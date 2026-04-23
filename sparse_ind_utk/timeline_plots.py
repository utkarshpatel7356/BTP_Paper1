"""
timeline_plots.py — All plotting functions for the timeline ablation study.
"""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch
from itertools import combinations

COLORS = ["#2563EB","#D97706","#059669","#8B5CF6","#DC2626","#0891B2","#374151"]
NAMES = ["RF+SHAP Only","GNN Influence Only","Emb-SHAP Only",
         "Hybrid v1","Hybrid v2","Hybrid v2 + Attn QP"]

def _setup():
    plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,
        "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.3,
        "grid.linestyle":"--","figure.dpi":150})

def plot_score_distributions(shap_rf, influence, shap_emb, d):
    _setup()
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax,scores,name,c in zip(axes,[shap_rf,influence,shap_emb],
            ["RF-SHAP","GNN Influence","Emb-SHAP"],COLORS[:3]):
        ax.hist(scores,bins=50,color=c,alpha=0.7,edgecolor="none")
        ax.set_title(f"{name} score distribution",fontsize=11)
        ax.set_xlabel("Score"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(d,"score_distributions.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/score_distributions.png")

def plot_top30_bars(shap_rf, influence, shap_emb, tickers, d):
    _setup()
    fig, axes = plt.subplots(1,3,figsize=(18,8))
    for ax,scores,name,c in zip(axes,[shap_rf,influence,shap_emb],
            ["RF-SHAP","GNN Influence","Emb-SHAP"],COLORS[:3]):
        order = np.argsort(scores)[::-1][:30]
        ax.barh(range(30),[scores[i] for i in order][::-1],color=c,edgecolor="none")
        ax.set_yticks(range(30))
        ax.set_yticklabels([tickers[i] for i in order][::-1],fontsize=7)
        ax.set_title(f"Top-30 by {name}",fontsize=11)
        ax.set_xlabel("Score")
    fig.tight_layout()
    fig.savefig(os.path.join(d,"top30_per_signal.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/top30_per_signal.png")

def plot_selection_overlap(all_selected, d):
    _setup()
    n = len(all_selected)
    jaccard = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            si,sj = set(all_selected[i]),set(all_selected[j])
            jaccard[i,j] = len(si&sj)/len(si|sj) if len(si|sj)>0 else 0
    fig,ax = plt.subplots(figsize=(8,7))
    sns.heatmap(jaccard,annot=True,fmt=".2f",xticklabels=NAMES,
                yticklabels=NAMES,cmap="YlGnBu",ax=ax,vmin=0,vmax=1)
    ax.set_title("Selection overlap (Jaccard similarity)",fontsize=13)
    plt.xticks(rotation=35,ha="right",fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d,"selection_overlap.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/selection_overlap.png")

def plot_cumulative_ladder(dates, idx_ret, all_port_rets, d):
    _setup()
    fig,ax = plt.subplots(figsize=(13,6))
    cum_idx = np.cumprod(1+idx_ret)*100
    ax.plot(dates,cum_idx,color="#374151",lw=2.5,label="S&P 500",zorder=10)
    for i,(ret,name) in enumerate(zip(all_port_rets,NAMES)):
        cum = np.cumprod(1+ret)*100
        ax.plot(dates,cum,color=COLORS[i],lw=1.5,label=name,
                linestyle=["-","--","-.","-","--","-."][i])
    ax.set_title("Cumulative Returns — Ablation Progression",fontsize=14,pad=12)
    ax.set_ylabel("Cumulative return (rebased to 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=9,framealpha=0.9,ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(d,"cumulative_returns_ladder.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/cumulative_returns_ladder.png")

def plot_rolling_te_ladder(dates, idx_ret, all_port_rets, d, window=30):
    _setup()
    fig,ax = plt.subplots(figsize=(13,5))
    for i,(ret,name) in enumerate(zip(all_port_rets,NAMES)):
        excess = ret - idx_ret
        te = pd.Series(excess).rolling(window).std().values*np.sqrt(252)*100
        ax.plot(dates,te,color=COLORS[i],lw=1.3,label=name,
                linestyle=["-","--","-.","-","--","-."][i])
    ax.set_title(f"{window}-day Rolling Tracking Error — All Strategies",fontsize=13)
    ax.set_ylabel("Tracking error (ann. %)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(d,"rolling_te_ladder.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/rolling_te_ladder.png")

def plot_metrics_bars(all_metrics, d):
    _setup()
    keys = ["tracking_error_pct","sharpe","beta","max_drawdown_pct"]
    titles = ["Tracking Error (ann. %)","Sharpe Ratio","Beta","Max Drawdown (%)"]
    fig,axes = plt.subplots(2,2,figsize=(14,9))
    for ax,key,title in zip(axes.flat,keys,titles):
        vals = [m[key] for m in all_metrics]
        bars = ax.bar(range(len(NAMES)),vals,color=COLORS[:len(NAMES)],edgecolor="none")
        ax.set_xticks(range(len(NAMES)))
        ax.set_xticklabels(NAMES,rotation=30,ha="right",fontsize=8)
        ax.set_title(title,fontsize=12)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.2f}",
                    ha="center",va="bottom",fontsize=8)
    fig.suptitle("Metrics Progression Across Strategies",fontsize=14,y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(d,"metrics_progression.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/metrics_progression.png")

def plot_weight_boxplots(all_weights, d):
    _setup()
    fig,ax = plt.subplots(figsize=(10,5))
    data = [w*100 for w in all_weights]
    bp = ax.boxplot(data,labels=NAMES,patch_artist=True,showfliers=True)
    for patch,c in zip(bp["boxes"],COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("Weight (%)")
    ax.set_title("Weight Distribution per Strategy",fontsize=13)
    plt.xticks(rotation=25,ha="right",fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(d,"weight_boxplots.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/weight_boxplots.png")

def plot_sector_comparison(all_selected, sector_map, d):
    _setup()
    all_sectors = sorted(set(sector_map.values()))
    data = {s: [] for s in all_sectors}
    for sel in all_selected:
        counts = {}
        for t in sel:
            s = sector_map.get(t,"Unknown")
            counts[s] = counts.get(s,0)+1
        for s in all_sectors:
            data[s].append(counts.get(s,0))
    fig,ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(NAMES)); width=0.7
    bottom = np.zeros(len(NAMES))
    palette = sns.color_palette("tab20",len(all_sectors))
    for i,s in enumerate(all_sectors):
        vals = np.array(data[s],dtype=float)
        ax.bar(x,vals,width,bottom=bottom,label=s,color=palette[i])
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(NAMES,rotation=25,ha="right",fontsize=9)
    ax.set_ylabel("Number of stocks"); ax.set_title("Sector Allocation per Strategy",fontsize=13)
    ax.legend(fontsize=7,ncol=3,loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(d,"sector_comparison.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/sector_comparison.png")

def plot_alpha_beta_sweep(sweep_results, d):
    _setup()
    df = pd.DataFrame(sweep_results)
    pivot = df.pivot_table(index="beta",columns="alpha",values="te")
    fig,ax = plt.subplots(figsize=(9,7))
    sns.heatmap(pivot,annot=True,fmt=".1f",cmap="RdYlGn_r",ax=ax,
                cbar_kws={"label":"Tracking Error (ann. %)"})
    ax.set_title("α-β Sensitivity: Tracking Error",fontsize=13)
    ax.set_xlabel("α (RF-SHAP weight)"); ax.set_ylabel("β (Emb-SHAP weight)")
    fig.tight_layout()
    fig.savefig(os.path.join(d,"alpha_beta_sweep.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/alpha_beta_sweep.png")

def plot_summary_table(all_metrics, d):
    _setup()
    cols = ["tracking_error_pct","info_ratio","total_return_pct",
            "max_drawdown_pct","sharpe","beta"]
    headers = ["TE (%)","IR","Return (%)","MaxDD (%)","Sharpe","Beta"]
    cell_text = []
    for m in all_metrics:
        cell_text.append([f"{m[c]:.2f}" for c in cols])
    fig,ax = plt.subplots(figsize=(14,4))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text,rowLabels=NAMES,colLabels=headers,
                   cellLoc="center",loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1,1.8)
    for i in range(len(headers)):
        tbl[0,i].set_facecolor("#E5E7EB")
    for i in range(len(NAMES)):
        tbl[i+1,-1].set_facecolor("#DCFCE7" if abs(all_metrics[i]["beta"]-1)<0.05 else "#FEE2E2")
    # Highlight best TE
    tes = [m["tracking_error_pct"] for m in all_metrics]
    best_te = tes.index(min(tes))
    tbl[best_te+1,0].set_facecolor("#BBF7D0")
    ax.set_title("Summary: All Strategies Compared",fontsize=14,pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(d,"summary_table.png"),bbox_inches="tight")
    plt.close(fig); print(f"Saved → {d}/summary_table.png")

def save_timeline_results(all_metrics, d):
    os.makedirs(d,exist_ok=True)
    path = os.path.join(d,"timeline_metrics.json")
    with open(path,"w") as f:
        json.dump(all_metrics,f,indent=2)
    print(f"Saved → {path}")
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(d,"timeline_metrics.csv"),index=False)
    print(f"Saved → {d}/timeline_metrics.csv")
