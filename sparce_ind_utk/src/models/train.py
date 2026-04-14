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
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Tuple

from src.models.gat_model import SparseIndexGNN


def make_windows(
    x: torch.Tensor,   # (N+1, T, F)
    y: torch.Tensor,   # (T,)
    window: int,
    stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice into overlapping windows. Returns (B, N+1, W, F) and (B,)."""
    T = x.shape[1]
    xs, ys = [], []
    for t in range(window, T, stride):
        xs.append(x[:, t - window : t, :])
        ys.append(y[t])
    return torch.stack(xs), torch.stack(ys)


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

    x = graph_data.x.to(device)                  # (N+1, T, F)
    y = graph_data.y.to(device)                   # (T,)
    edge_index = graph_data.edge_index.to(device) # (2, E)
    edge_weight = graph_data.edge_weight.to(device)

    T = x.shape[1]
    W = mcfg["gru_window"]
    F = x.shape[2]

    # Train/val split (last 10% of training data as val)
    train_frac = 0.9
    T_train = int(T * train_frac)

    x_train = x[:, :T_train, :]
    y_train = y[:T_train]
    x_val = x[:, T_train:, :]
    y_val = y[T_train:]

    # Build windows
    X_train, Y_train = make_windows(x_train, y_train, W)  # (B_tr, N+1, W, F), (B_tr,)
    X_val, Y_val = make_windows(x_val, y_val, W)

    print(f"Training samples: {len(X_train)}  Validation: {len(X_val)}")

    model = SparseIndexGNN(
        in_features=F,
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

    for epoch in range(1, mcfg["epochs"] + 1):
        # --- train ---
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        for i in range(0, len(X_train), bs):
            idx = perm[i : i + bs]
            batch_x = X_train[idx]          # (bs, N+1, W, F)
            batch_y = Y_train[idx]           # (bs,)
            preds = []
            for b in range(len(batch_x)):
                y_hat, _, _ = model(batch_x[b], edge_index, edge_weight)
                preds.append(y_hat)
            preds = torch.stack(preds)
            loss = criterion(preds, batch_y)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= len(X_train)

        # --- val ---
        model.eval()
        with torch.no_grad():
            val_preds = [model(X_val[i], edge_index, edge_weight)[0] for i in range(len(X_val))]
            val_preds = torch.stack(val_preds)
            val_loss = criterion(val_preds, Y_val).item()

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_gnn.pt"))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{mcfg['epochs']}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

    # Load best weights
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_gnn.pt")))
    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    return model, np.array(train_losses), np.array(val_losses)
