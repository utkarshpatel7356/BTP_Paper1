"""
src/models/train.py
-------------------
Training loop for SparseIndexGNN.

Rolling-window training:
  - For each training day t (after initial burn-in):
      * Input  : x[:, t-W:t, :]   (N+1, W, F)
      * Target : y[t]              scalar index return
  - Optimise MSE loss with Adam
  - Save best checkpoint by validation loss

Memory-efficient: uses a lazy dataset that slices windows on the fly
instead of pre-materializing all windows (which would use ~66GB).
"""

import gc
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Tuple

from src.models.gat_model import SparseIndexGNN


def _embedding_diversity_loss(
    embeddings: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Penalise embedding collapse by encouraging low mean cosine
    similarity among node embeddings.

    Parameters
    ----------
    embeddings : (N+1, hidden) node embeddings from one forward pass
    margin     : target maximum mean cosine similarity

    Returns
    -------
    loss : scalar tensor — ReLU(mean_cosine_sim - margin)
    """
    normed = torch.nn.functional.normalize(embeddings, dim=-1)
    sim_matrix = normed @ normed.T  # (N+1, N+1)
    mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
    mean_sim = sim_matrix[mask].mean()
    return torch.nn.functional.relu(mean_sim - margin)


class WindowDataset(Dataset):
    """
    Lazy sliding-window dataset. Slices windows on-the-fly from the
    original tensors, avoiding pre-materialization of ~66GB data.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, window: int, stride: int = 1):
        """
        x : (N+1, T, F) full node features
        y : (T,) target index returns
        window : temporal window size
        stride : step between consecutive windows
        """
        self.x = x
        self.y = y
        self.window = window
        # Valid window end indices
        self.indices = list(range(window, x.shape[1], stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x_window = self.x[:, t - self.window : t, :]  # (N+1, W, F) — sliced, not copied
        y_target = self.y[t]                            # scalar
        return x_window, y_target


def train_model(
    graph_data,          # PyG Data from build_graph
    cfg: dict,
    checkpoint_dir: str = "outputs",
) -> Tuple[SparseIndexGNN, np.ndarray, np.ndarray]:
    """
    Train SparseIndexGNN on the training split.

    Returns
    -------
    model        : trained SparseIndexGNN
    train_losses : (epochs,) array
    val_losses   : (epochs,) array
    """
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device(mcfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    x = graph_data.x                                    # (N+1, T, F)
    y = graph_data.y                                    # (T,)
    edge_index = graph_data.edge_index.to(device)       # (2, E)
    edge_weight = graph_data.edge_weight.to(device)

    T = x.shape[1]
    W = mcfg["gru_window"]

    # Train/val split (last 10% of training data as val)
    train_frac = 0.9
    T_train = int(T * train_frac)

    x_train = x[:, :T_train, :]
    y_train = y[:T_train]
    x_val = x[:, T_train:, :]
    y_val = y[T_train:]

    # Build lazy datasets
    train_dataset = WindowDataset(x_train, y_train, W)
    val_dataset = WindowDataset(x_val, y_val, W)

    print(f"Training samples: {len(train_dataset)}  Validation: {len(val_dataset)}")

    F_in = x.shape[2]
    model = SparseIndexGNN(
        in_features=F_in,
        hidden=mcfg["gnn_hidden"],
        heads=mcfg["gnn_heads"],
        num_layers=mcfg["gnn_layers"],
        gru_window=W,
        dropout=mcfg["dropout"],
    ).to(device)

    optimiser = Adam(model.parameters(), lr=mcfg["lr"])
    scheduler = ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    train_losses, val_losses = [], []
    bs = mcfg["batch_size"]

    # Use DataLoader for shuffling and batching
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)

    # Mixed precision scaler for GPU memory efficiency
    use_amp = (device.type == 'cuda')
    scaler = GradScaler('cuda') if use_amp else None

    for epoch in range(1, mcfg["epochs"] + 1):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in train_loader:
            # batch_x: (bs, N+1, W, F),  batch_y: (bs,)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimiser.zero_grad()

            with autocast('cuda', enabled=use_amp):
                preds = []
                all_embeddings = []
                for b in range(len(batch_x)):
                    y_hat, embs, _, _ = model(batch_x[b], edge_index, edge_weight)
                    preds.append(y_hat)
                    all_embeddings.append(embs)
                preds = torch.stack(preds)
                loss = criterion(preds, batch_y)

                # Embedding diversity regularisation
                div_lambda = cfg["model"].get("diversity_lambda", 0.0)
                if div_lambda > 0 and len(all_embeddings) > 0:
                    div_loss = _embedding_diversity_loss(all_embeddings[-1])
                    loss = loss + div_lambda * div_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                # Capture the norm here
                raw_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                # Capture the norm here
                raw_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            
            if (n_samples // len(batch_x)) % 10 == 0:
                print(f"  [Monitor] Loss: {loss.item():.4f} | Raw Grad Norm: {raw_grad_norm.item():.4f}")

            epoch_loss += loss.item() * len(batch_x)
            n_samples += len(batch_x)

            # Free GPU memory between batches
            del batch_x, batch_y, preds, loss
            if use_amp:
                torch.cuda.empty_cache()

        epoch_loss /= max(n_samples, 1)

        # --- val ---
        model.eval()
        val_loss_accum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                with autocast('cuda', enabled=use_amp):
                    preds = []
                    for b in range(len(batch_x)):
                        y_hat, _, _, _ = model(batch_x[b], edge_index, edge_weight)
                        preds.append(y_hat)
                    preds = torch.stack(preds)
                val_loss_accum += criterion(preds, batch_y).item() * len(batch_x)
                val_count += len(batch_x)
                del batch_x, batch_y, preds
        val_loss = val_loss_accum / max(val_count, 1)

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_gnn.pt"))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{mcfg['epochs']}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

    # Load best weights
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_gnn.pt"), map_location=device))
    print(f"\nTraining complete. Best val loss: {best_val:.6f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, np.array(train_losses), np.array(val_losses)
