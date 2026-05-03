# eval_wangchanberta.py
# ประเมินคุณภาพ WangchanBERTa Embedding แยกจาก pipeline หลัก
#
# วัด 4 มิติ:
#   1. Embedding Quality  — intra/inter class distance, cosine sim, anisotropy
#   2. Class Separability — silhouette, linear probe (LogReg), KNN accuracy
#   3. Visualization      — t-SNE 2D (by label + by stock)
#   4. Per-Stock Report   — สรุปคะแนนแต่ละหุ้น + บันทึก JSON
#
# Input  : final_5year_dataset.csv  (ต้องรัน prepare_data.py ก่อน)
#          หรือระบุ csv อื่นที่มีคอลัมน์ vec_0..767 + Target_Label + Stock
#
# Output : eval_output/
#   wangchanberta_eval_report.json   ← ผลเชิงตัวเลขทั้งหมด
#   plots/tsne_by_label.png          ← t-SNE จุดสีตาม label
#   plots/tsne_by_stock.png          ← t-SNE จุดสีตามหุ้น
#   plots/cosine_sim_matrix.png      ← heatmap cosine sim ระหว่างหุ้น
#   plots/class_dist_per_stock.png   ← intra/inter distance bar chart
#   plots/probe_summary.png          ← Linear Probe accuracy per stock
#
# Install: pip install scikit-learn matplotlib seaborn pandas numpy tqdm
# (ไม่ต้องโหลด WangchanBERTa อีกครั้ง — ใช้ vec จาก CSV โดยตรง)

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, roc_auc_score, silhouette_score,
    silhouette_samples,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_FILE    = "final_5year_dataset.csv"
OUT_DIR      = Path("eval_output")
PLOT_DIR     = OUT_DIR / "plots"
STOCKS       = ["PTT", "AOT", "DELTA", "ADVANC", "SCB"]
RANDOM_STATE = 42

OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# สีสำหรับ plot
LABEL_COLORS = {0: "#e74c3c", 1: "#2ecc71"}   # 0=Down/Flat, 1=Up
STOCK_PALETTE = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]


# ============================================================
# UTILS
# ============================================================
def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (N×N)"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    X_norm = X / norms
    return X_norm @ X_norm.T


def pairwise_cosine_mean(X_a: np.ndarray, X_b: np.ndarray) -> float:
    """Mean cosine sim ระหว่างสอง group"""
    na = np.linalg.norm(X_a, axis=1, keepdims=True)
    nb = np.linalg.norm(X_b, axis=1, keepdims=True)
    na = np.where(na == 0, 1e-8, na)
    nb = np.where(nb == 0, 1e-8, nb)
    A = X_a / na
    B = X_b / nb
    # Sample เพื่อไม่ให้ช้า
    n = min(300, len(A), len(B))
    idx_a = np.random.choice(len(A), n, replace=False)
    idx_b = np.random.choice(len(B), n, replace=False)
    return float((A[idx_a] @ B[idx_b].T).mean())


def anisotropy(X: np.ndarray, n_sample: int = 500) -> float:
    """
    Anisotropy = ค่าเฉลี่ย cosine sim ระหว่าง random pairs
    ถ้าสูง (>0.8) แปลว่า embeddings กระจุกตัว (ไม่ดี)
    ถ้าต่ำ (≈0)   แปลว่ากระจายในพื้นที่สม่ำเสมอ (ดี)
    """
    idx = np.random.choice(len(X), min(n_sample, len(X)), replace=False)
    sample = X[idx]
    sim = cosine_sim_matrix(sample)
    upper = sim[np.triu_indices(len(sample), k=1)]
    return float(upper.mean())


# ============================================================
# STEP 1: โหลดข้อมูลและแยก Embedding
# ============================================================
def load_embeddings(data_file: str):
    print(f"โหลด {data_file}...")
    df = pd.read_csv(data_file, parse_dates=["Target_Date"])
    df = df.sort_values(["Stock", "Target_Date"]).reset_index(drop=True)

    vec_cols = sorted(
        [c for c in df.columns if c.startswith("vec_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not vec_cols:
        raise ValueError("ไม่พบคอลัมน์ vec_* ใน CSV — รัน prepare_data.py ก่อน")

    X = df[vec_cols].values.astype(np.float32)
    y = df["Target_Label"].values.astype(int)
    stocks_col = df["Stock"].values

    print(f"✅ โหลดสำเร็จ: {len(df):,} แถว | {len(vec_cols)} dim | "
          f"label dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    return df, X, y, stocks_col, vec_cols


# ============================================================
# STEP 2: Embedding Quality (ระดับ Global)
# ============================================================
def eval_embedding_quality(X: np.ndarray, y: np.ndarray, stock_name: str = "ALL"):
    print(f"\n[{stock_name}] Embedding Quality...")
    np.random.seed(RANDOM_STATE)

    X0 = X[y == 0]
    X1 = X[y == 1]

    # L2 distance
    def mean_pairwise_l2(A, B, n=300):
        n = min(n, len(A), len(B))
        ia = np.random.choice(len(A), n, replace=False)
        ib = np.random.choice(len(B), n, replace=False)
        diff = A[ia] - B[ib]
        return float(np.linalg.norm(diff, axis=1).mean())

    def mean_intra_l2(A, n=300):
        n = min(n, len(A))
        ia = np.random.choice(len(A), n, replace=False)
        ib = np.random.choice(len(A), n, replace=False)
        diff = A[ia] - A[ib]
        return float(np.linalg.norm(diff, axis=1).mean())

    intra_down = mean_intra_l2(X0)
    intra_up   = mean_intra_l2(X1)
    inter      = mean_pairwise_l2(X0, X1)
    cos_down   = pairwise_cosine_mean(X0, X0)
    cos_up     = pairwise_cosine_mean(X1, X1)
    cos_inter  = pairwise_cosine_mean(X0, X1)
    ani        = anisotropy(X)

    # Fisher's ratio (inter / intra) — ยิ่งสูงยิ่งดี
    intra_avg  = (intra_down + intra_up) / 2
    fisher     = round(inter / intra_avg, 4) if intra_avg > 0 else 0

    result = {
        "intra_l2_down":  round(intra_down, 4),
        "intra_l2_up":    round(intra_up,   4),
        "inter_l2":       round(inter,       4),
        "fisher_ratio":   fisher,
        "cosine_intra_down": round(cos_down,  4),
        "cosine_intra_up":   round(cos_up,    4),
        "cosine_inter":      round(cos_inter, 4),
        "anisotropy":        round(ani,       4),
        "n_down": int((y == 0).sum()),
        "n_up":   int((y == 1).sum()),
    }

    print(f"  intra L2 Down={intra_down:.3f}  Up={intra_up:.3f}  inter={inter:.3f}  Fisher={fisher:.3f}")
    print(f"  cosine intra_Down={cos_down:.3f}  intra_Up={cos_up:.3f}  inter={cos_inter:.3f}")
    print(f"  anisotropy={ani:.3f}  {'✅ กระจายดี' if ani < 0.5 else '⚠ กระจุกตัว'}")

    return result


# ============================================================
# STEP 3: Linear Probe + KNN (Class Separability)
# ============================================================
def eval_separability(X: np.ndarray, y: np.ndarray, stock_name: str = "ALL"):
    print(f"\n[{stock_name}] Separability (LogReg + KNN)...")

    if len(np.unique(y)) < 2 or len(X) < 20:
        print("  ⚠ ข้อมูลไม่พอ — ข้าม")
        return {"logreg_acc": None, "logreg_auc": None, "knn_acc": None, "silhouette": None}

    # Scale ก่อน (ใช้เฉพาะ eval ไม่กระทบ model จริง)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # PCA 50 dim เพื่อลด noise (ใช้แค่สำหรับ probe)
    pca   = PCA(n_components=min(50, X_s.shape[1], len(X_s) - 1), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_s)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Linear Probe
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    lr_acc_scores = cross_val_score(lr, X_pca, y, cv=cv, scoring="accuracy")
    lr_auc_scores = cross_val_score(lr, X_pca, y, cv=cv, scoring="roc_auc")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn_acc_scores = cross_val_score(knn, X_pca, y, cv=cv, scoring="accuracy")

    # Silhouette (sample เพื่อความเร็ว)
    n_sil = min(1000, len(X_s))
    idx   = np.random.choice(len(X_s), n_sil, replace=False)
    sil   = silhouette_score(X_s[idx], y[idx]) if len(np.unique(y[idx])) > 1 else 0.0

    result = {
        "logreg_acc":  round(float(lr_acc_scores.mean()),  4),
        "logreg_acc_std": round(float(lr_acc_scores.std()), 4),
        "logreg_auc":  round(float(lr_auc_scores.mean()),  4),
        "knn_acc":     round(float(knn_acc_scores.mean()), 4),
        "silhouette":  round(float(sil), 4),
        "pca_components": int(X_pca.shape[1]),
        "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
    }

    print(f"  LogReg Acc={result['logreg_acc']:.1%} AUC={result['logreg_auc']:.3f} "
          f"| KNN Acc={result['knn_acc']:.1%} | Silhouette={result['silhouette']:.4f}")
    return result


# ============================================================
# STEP 4: Visualization — t-SNE
# ============================================================
def plot_tsne(X: np.ndarray, y: np.ndarray, stocks_col: np.ndarray):
    print("\n[VIZ] คำนวณ t-SNE (อาจใช้เวลา 1-3 นาที)...")
    np.random.seed(RANDOM_STATE)

    # ลด dim ด้วย PCA ก่อน t-SNE (เพื่อความเร็ว)
    n_sample = min(2000, len(X))
    idx = np.random.choice(len(X), n_sample, replace=False)
    X_s, y_s, st_s = X[idx], y[idx], stocks_col[idx]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_s)
    pca    = PCA(n_components=50, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X_sc)

    tsne  = TSNE(n_components=2, perplexity=40, learning_rate=200,
                 max_iter=1000, random_state=RANDOM_STATE, init="pca")
    X_2d  = tsne.fit_transform(X_pca)
    print(f"  KL divergence: {tsne.kl_divergence_:.4f} (ต่ำ = ดี)")

    # ─── Plot 1: by Label ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, color, name in [(0, "#e74c3c", "Down/Flat"), (1, "#2ecc71", "Up")]:
        mask = y_s == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=12,
                   alpha=0.55, label=f"{name} (n={mask.sum()})", rasterized=True)
    ax.set_title("t-SNE of WangchanBERTa Embeddings (by Label)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path1 = PLOT_DIR / "tsne_by_label.png"
    plt.savefig(path1, dpi=130)
    plt.close()
    print(f"  บันทึก: {path1}")

    # ─── Plot 2: by Stock ────────────────────────────────────
    unique_stocks = sorted(np.unique(st_s))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, stock in enumerate(unique_stocks):
        mask = st_s == stock
        color = STOCK_PALETTE[i % len(STOCK_PALETTE)]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=12,
                   alpha=0.55, label=f"{stock} (n={mask.sum()})", rasterized=True)
    ax.set_title("t-SNE of WangchanBERTa Embeddings (by Stock)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", markerscale=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path2 = PLOT_DIR / "tsne_by_stock.png"
    plt.savefig(path2, dpi=130)
    plt.close()
    print(f"  บันทึก: {path2}")

    return float(tsne.kl_divergence_)


# ============================================================
# STEP 5: Cosine Similarity Heatmap ระหว่างหุ้น
# ============================================================
def plot_stock_cosine_heatmap(X: np.ndarray, stocks_col: np.ndarray):
    print("\n[VIZ] Cosine similarity heatmap ระหว่างหุ้น...")
    unique_stocks = sorted(np.unique(stocks_col))
    n = len(unique_stocks)
    sim_matrix = np.zeros((n, n))

    for i, s1 in enumerate(unique_stocks):
        for j, s2 in enumerate(unique_stocks):
            X1 = X[stocks_col == s1]
            X2 = X[stocks_col == s2]
            # centroid cosine sim
            c1 = X1.mean(axis=0, keepdims=True)
            c2 = X2.mean(axis=0, keepdims=True)
            norm1 = np.linalg.norm(c1)
            norm2 = np.linalg.norm(c2)
            sim_matrix[i, j] = ((c1 @ c2.T) / (norm1 * norm2 + 1e-8)).item()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        sim_matrix, annot=True, fmt=".3f",
        xticklabels=unique_stocks, yticklabels=unique_stocks,
        cmap="YlOrRd", vmin=0.8, vmax=1.0, ax=ax,
        linewidths=0.5, square=True,
    )
    ax.set_title("Centroid Cosine Similarity (WangchanBERTa)\nระหว่างหุ้น",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / "cosine_sim_matrix.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  บันทึก: {path}")

    # ถ้า similarity ระหว่างหุ้นสูงมาก (>0.98) แปลว่า embedding ไม่แยกความต่างของหุ้ว
    avg_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * n - n)
    print(f"  avg off-diagonal cosine sim = {avg_off_diag:.4f} "
          f"{'⚠ ใกล้เคียงกันมาก' if avg_off_diag > 0.97 else '✅ มีความแตกต่าง'}")
    return round(avg_off_diag, 4)


# ============================================================
# STEP 6: Intra/Inter Distance Bar Chart (per stock)
# ============================================================
def plot_class_distance(per_stock_quality: dict):
    print("\n[VIZ] Intra/Inter L2 distance per stock...")
    stocks_list  = list(per_stock_quality.keys())
    intra_values = [(r["intra_l2_down"] + r["intra_l2_up"]) / 2 for r in per_stock_quality.values()]
    inter_values = [r["inter_l2"] for r in per_stock_quality.values()]

    x     = np.arange(len(stocks_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, intra_values, width, label="Intra-class L2 (avg)", color="#3498db", alpha=0.85)
    b2 = ax.bar(x + width/2, inter_values, width, label="Inter-class L2",        color="#e74c3c", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(stocks_list)
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("WangchanBERTa: Intra vs Inter Class Distance\n(Inter > Intra = ดี, embedding แยก Up/Down ได้)",
                 fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = PLOT_DIR / "class_dist_per_stock.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  บันทึก: {path}")


# ============================================================
# STEP 7: Linear Probe Summary Chart
# ============================================================
def plot_probe_summary(per_stock_sep: dict):
    print("\n[VIZ] Linear Probe summary chart...")
    stocks_list = list(per_stock_sep.keys())
    lr_accs  = [v["logreg_acc"]  or 0.5 for v in per_stock_sep.values()]
    lr_aucs  = [v["logreg_auc"]  or 0.5 for v in per_stock_sep.values()]
    knn_accs = [v["knn_acc"]     or 0.5 for v in per_stock_sep.values()]
    sils     = [v["silhouette"]  or 0.0 for v in per_stock_sep.values()]

    x = np.arange(len(stocks_list))
    w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("WangchanBERTa Embedding — Separability Evaluation",
                 fontsize=13, fontweight="bold")

    # Left: Acc + AUC
    ax = axes[0]
    ax.bar(x - w,   lr_accs,  w, label="LogReg Acc",  color="#2ecc71", alpha=0.85)
    ax.bar(x,       lr_aucs,  w, label="LogReg AUC",  color="#3498db", alpha=0.85)
    ax.bar(x + w,   knn_accs, w, label="KNN Acc",     color="#9b59b6", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Baseline 0.5")
    ax.set_xticks(x); ax.set_xticklabels(stocks_list)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Classification Probe (5-fold CV)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    for i, (a, b, c) in enumerate(zip(lr_accs, lr_aucs, knn_accs)):
        ax.text(i - w,   a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i,       b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w,   c + 0.01, f"{c:.2f}", ha="center", va="bottom", fontsize=7)

    # Right: Silhouette
    ax2 = axes[1]
    colors_sil = ["#2ecc71" if s > 0 else "#e74c3c" for s in sils]
    ax2.bar(x, sils, color=colors_sil, alpha=0.85)
    ax2.axhline(0, color="gray", linestyle="--", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels(stocks_list)
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score (>0 = Up/Down แยกกันดี)")
    ax2.grid(axis="y", alpha=0.3)
    for i, s in enumerate(sils):
        ax2.text(i, s + 0.002, f"{s:.4f}", ha="center", va="bottom", fontsize=8)

    # เพิ่ม interpretation label
    legend_els = [
        mpatches.Patch(color="#2ecc71", label="Positive (ดี)"),
        mpatches.Patch(color="#e74c3c", label="Negative (ไม่ดี)"),
    ]
    ax2.legend(handles=legend_els, fontsize=8)

    plt.tight_layout()
    path = PLOT_DIR / "probe_summary.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  บันทึก: {path}")


# ============================================================
# STEP 8: สร้าง Overall Score Card
# ============================================================
def score_card(quality_all: dict, sep_all: dict) -> dict:
    """
    คำนวณ Overall Score 0-100 จาก 4 metrics:
      - Separability Index  (Fisher ratio)
      - Linear Probe AUC    (ยิ่งสูงยิ่งดี)
      - Silhouette          (ยิ่งสูงยิ่งดี)
      - Anisotropy penalty  (ยิ่งต่ำยิ่งดี)
    """
    # normalize เป็น 0-1 แล้วถ่วงน้ำหนัก
    fisher   = min(quality_all.get("fisher_ratio", 1.0) / 2.0, 1.0)   # cap ที่ 2.0
    auc      = max(sep_all.get("logreg_auc", 0.5) - 0.5, 0) * 2       # 0.5→0, 1.0→1
    sil_raw  = sep_all.get("silhouette", 0.0)
    sil_norm = (sil_raw + 1) / 2                                        # [-1,1]→[0,1]
    ani      = 1.0 - quality_all.get("anisotropy", 0.5)                 # invert

    score = (fisher * 25 + auc * 35 + sil_norm * 25 + ani * 15)
    grade = "A" if score >= 75 else "B" if score >= 60 else "C" if score >= 45 else "D"

    return {
        "overall_score": round(score, 1),
        "grade":  grade,
        "breakdown": {
            "separability_index": round(fisher * 25, 1),
            "linear_probe_auc":   round(auc   * 35, 1),
            "silhouette":         round(sil_norm * 25, 1),
            "anisotropy_penalty": round(ani   * 15, 1),
        },
        "interpretation": {
            "A": "embedding แยก Up/Down ได้ดีมาก — เหมาะกับงาน classification",
            "B": "embedding มีประโยชน์พอสมควร — ควรรวมกับ feature อื่น",
            "C": "embedding แยก class ได้อ่อน — ลอง fine-tune หรือเพิ่มข้อมูล",
            "D": "embedding ไม่ช่วยงานนี้ — พิจารณา model ภาษาอื่น",
        }[grade],
    }


# ============================================================
# MAIN
# ============================================================
def run():
    np.random.seed(RANDOM_STATE)

    print("=" * 60)
    print("WangchanBERTa Embedding Evaluation")
    print("=" * 60)

    df, X, y, stocks_col, vec_cols = load_embeddings(DATA_FILE)

    # ─── Global evaluation ───────────────────────────────────
    quality_all = eval_embedding_quality(X, y, stock_name="ALL")
    sep_all     = eval_separability(X, y, stock_name="ALL")

    # ─── Per-stock evaluation ─────────────────────────────────
    per_stock_quality = {}
    per_stock_sep     = {}

    for stock in STOCKS:
        mask  = stocks_col == stock
        X_st  = X[mask]
        y_st  = y[mask]

        if len(X_st) < 20:
            print(f"\n[{stock}] ข้อมูลน้อยเกินไป ({len(X_st)} แถว) — ข้าม")
            continue

        per_stock_quality[stock] = eval_embedding_quality(X_st, y_st, stock_name=stock)
        per_stock_sep[stock]     = eval_separability(X_st, y_st, stock_name=stock)

    # ─── Visualization ───────────────────────────────────────
    kl_div        = plot_tsne(X, y, stocks_col)
    avg_cos_stock = plot_stock_cosine_heatmap(X, stocks_col)

    if per_stock_quality:
        plot_class_distance(per_stock_quality)
    if per_stock_sep:
        plot_probe_summary(per_stock_sep)

    # ─── Score Card ──────────────────────────────────────────
    card = score_card(quality_all, sep_all)

    # ─── สรุปผล ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("สรุปผลการประเมิน WangchanBERTa Embedding")
    print("=" * 60)
    print(f"\n📊 Overall Score: {card['overall_score']:.1f}/100  (Grade: {card['grade']})")
    print(f"   {card['interpretation']}")

    print(f"\n[GLOBAL — {len(X):,} samples]")
    print(f"  Fisher Ratio  : {quality_all['fisher_ratio']:.4f}  (>1.0 = inter > intra = ดี)")
    print(f"  Anisotropy    : {quality_all['anisotropy']:.4f}  (<0.5 = กระจายดี)")
    print(f"  LogReg AUC    : {sep_all['logreg_auc']:.4f}  (>0.55 = embedding มีประโยชน์)")
    print(f"  KNN Acc       : {sep_all['knn_acc']:.4f}")
    print(f"  Silhouette    : {sep_all['silhouette']:.4f}")

    if per_stock_sep:
        print(f"\n{'หุ้น':8s} {'LogReg AUC':>11s} {'KNN Acc':>9s} {'Silhouette':>11s} {'Fisher':>8s}")
        print("-" * 55)
        for stock in per_stock_sep:
            sep = per_stock_sep[stock]
            q   = per_stock_quality[stock]
            print(f"{stock:8s} {sep['logreg_auc']:>11.4f} {sep['knn_acc']:>9.4f} "
                  f"{sep['silhouette']:>11.4f} {q['fisher_ratio']:>8.4f}")

    print(f"\n  avg stock cosine sim = {avg_cos_stock:.4f}  "
          f"{'⚠ embeddings คล้ายกันมากระหว่างหุ้น' if avg_cos_stock > 0.97 else '✅ มีความแตกต่างระหว่างหุ้น'}")
    print(f"  t-SNE KL divergence  = {kl_div:.4f}  (ต่ำ = projection ดี)")

    # ─── บันทึก JSON ─────────────────────────────────────────
    def to_py(o):
        if isinstance(o, dict):
            return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [to_py(i) for i in o]
        if isinstance(o, (np.floating, np.integer, float)):
            return float(o) if not np.isnan(float(o)) else None
        return o

    report = {
        "model": "airesearch/wangchanberta-base-att-spm-uncased",
        "data_file": DATA_FILE,
        "n_samples": int(len(X)),
        "embed_dim": int(X.shape[1]),
        "score_card": card,
        "global_quality": quality_all,
        "global_separability": sep_all,
        "per_stock_quality": {k: to_py(v) for k, v in per_stock_quality.items()},
        "per_stock_separability": {k: to_py(v) for k, v in per_stock_sep.items()},
        "tsne_kl_divergence": round(kl_div, 4),
        "avg_cross_stock_cosine": avg_cos_stock,
        "plots": {
            "tsne_by_label":    str(PLOT_DIR / "tsne_by_label.png"),
            "tsne_by_stock":    str(PLOT_DIR / "tsne_by_stock.png"),
            "cosine_heatmap":   str(PLOT_DIR / "cosine_sim_matrix.png"),
            "class_distance":   str(PLOT_DIR / "class_dist_per_stock.png"),
            "probe_summary":    str(PLOT_DIR / "probe_summary.png"),
        },
    }

    report_path = OUT_DIR / "wangchanberta_eval_report.json"
    report_path.write_text(
        json.dumps(to_py(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ บันทึกรายงานที่: {report_path}")
    print(f"✅ บันทึกกราฟที่  : {PLOT_DIR}/")
    return report


# ============================================================
# คำอธิบาย Metrics สำหรับ Reference
# ============================================================
METRIC_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║            คู่มือแปลผล WangchanBERTa Evaluation             ║
╠══════════════════════════════════════════════════════════════╣
║  Fisher Ratio  = inter_class_L2 / avg_intra_class_L2         ║
║    > 1.0  → embedding แยก Up/Down ได้ดี                     ║
║    < 1.0  → embedding ไม่ช่วยแยก class                      ║
║                                                              ║
║  Anisotropy  = mean cosine sim ระหว่าง random pairs          ║
║    < 0.3  → embedding กระจายตัวดีในพื้นที่ (ดีมาก)          ║
║    0.3-0.6→ ปานกลาง                                         ║
║    > 0.8  → embedding กระจุกตัว (อาจ degenerate)            ║
║                                                              ║
║  Linear Probe AUC  = ถ้า AUC >> 0.5 → มี linear structure    ║
║    ใน embedding ที่สัมพันธ์กับ label                         ║
║                                                              ║
║  Silhouette  = [-1, 1]                                       ║
║    > 0    → cluster Up / Down แยกจากกัน                     ║
║    ≈ 0    → cluster ทับซ้อนกัน                              ║
║    < 0    → จัด cluster ผิด                                  ║
║                                                              ║
║  t-SNE KL divergence → ต่ำ = 2D projection รักษา structure  ║
║    ดีได้ที่ < 1.0                                           ║
╚══════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(METRIC_GUIDE)
    run()
    print("\nประเมิน WangchanBERTa เสร็จสมบูรณ์!")
