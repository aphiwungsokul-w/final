# train_svm.py
# ทำนายทิศทาง Open Price จากข่าว + ราคา ด้วย SVM (RBF kernel)
#
# Input  : final_5year_dataset.csv
# Output : models/svm_{stock}.pkl
#          models/pca_{stock}_svm.pkl
#          models/scaler_{stock}_svm.pkl
#          models/svm_evaluation_report.json
#          plots/svm_{stock}_cm.png
#          plots/svm_{stock}_roc.png
#          plots/svm_summary.png

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
data_file      = "final_5year_dataset.csv"
model_dir_path = "models"
plot_dir_path  = "plots"
stocks         = ["PTT", "AOT", "DELTA", "ADVANC", "SCB"]
price_cols     = ["Open", "Prev_Close", "High", "Low", "Close", "Volume"]
min_samples    = 80
train_ratio    = 0.70
val_ratio      = 0.15
pca_variance   = 0.95
param_grid     = {
    "C":      [0.01, 0.1, 1.0, 10.0],
    "gamma":  ["scale", "auto", 0.01, 0.001],
    "kernel": ["rbf"],
}
cv_folds       = 5
scoring        = "roc_auc"
class_weight   = "balanced"
probability    = True
random_state   = 42
threshold      = 0.50

model_dir = Path(model_dir_path)
plot_dir  = Path(plot_dir_path)
model_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)


# ============================================================
# STEP 1: โหลดข้อมูล
# ============================================================
def load_data():
    print(f"โหลด {data_file}...")
    df = pd.read_csv(data_file, parse_dates=["Target_Date"])
    df = df.sort_values(["Stock", "Target_Date"]).reset_index(drop=True)

    vec_cols = sorted(
        [c for c in df.columns if c.startswith("vec_")],
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"rows={len(df):,} | stocks={df['Stock'].nunique()} | features={len(vec_cols)+len(price_cols)}")

    print("\nจำนวนแถวต่อหุ้น:")
    for s, n in df.groupby("Stock").size().items():
        flag = "✅" if n >= min_samples else "⚠ น้อย"
        print(f"  {s:8s}: {n:4d}  {flag}")

    return df, vec_cols


# ============================================================
# STEP 2: เตรียม X, y + Scale + PCA แยกรายหุ้น
# ============================================================
def prepare_stock(df, stock, vec_cols):
    g = df[df["Stock"] == stock].sort_values("Target_Date").reset_index(drop=True)
    n = len(g)

    if n < min_samples:
        print(f"{stock}: {n} แถว < {min_samples} → ข้าม")
        return None

    X_vec   = g[vec_cols].values.astype(np.float32)
    X_price = g[price_cols].values.astype(np.float32)
    X_raw   = np.hstack([X_vec, X_price])
    y       = g["Target_Label"].values.astype(int)
    X_raw   = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_raw[:n_train])
    X_val_s   = scaler.transform(X_raw[n_train : n_train + n_val])
    X_test_s  = scaler.transform(X_raw[n_train + n_val :])
    joblib.dump(scaler, model_dir / f"scaler_{stock}_svm.pkl")

    pca       = PCA(n_components=pca_variance, random_state=random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p   = pca.transform(X_val_s)
    X_test_p  = pca.transform(X_test_s)
    joblib.dump(pca, model_dir / f"pca_{stock}_svm.pkl")

    n_comp  = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {X_raw.shape[1]} → {n_comp} components (variance = {var_exp:.1%})")

    return {
        "X_train": X_train_p, "y_train": y[:n_train],
        "X_val":   X_val_p,   "y_val":   y[n_train : n_train + n_val],
        "X_test":  X_test_p,  "y_test":  y[n_train + n_val :],
        "dates_test": g["Target_Date"].values[n_train + n_val :],
        "n_components": n_comp,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


# ============================================================
# STEP 3: GridSearchCV หา C และ gamma ที่ดีที่สุด
# ============================================================
def tune_svm(data):
    from sklearn.model_selection import TimeSeriesSplit

    X_fit = np.vstack([data["X_train"], data["X_val"]])
    y_fit = np.concatenate([data["y_train"], data["y_val"]])

    base_svm = SVC(class_weight=class_weight, probability=probability, random_state=random_state)
    tscv     = TimeSeriesSplit(n_splits=cv_folds)

    gs = GridSearchCV(
        estimator=base_svm, param_grid=param_grid,
        cv=tscv, scoring=scoring,
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_fit, y_fit)

    print(f"  best params: {gs.best_params_}")
    print(f"  best CV AUC: {gs.best_score_:.3f}")

    train_acc = accuracy_score(data["y_train"], gs.best_estimator_.predict(data["X_train"]))
    return gs.best_estimator_, gs.best_params_, gs.best_score_, round(float(train_acc), 4)


# ============================================================
# STEP 4: ประเมินผล
# ============================================================
def evaluate(model, data, stock):
    y_prob = model.predict_proba(data["X_test"])[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    y_true = data["y_test"]

    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    auc    = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    cm     = confusion_matrix(y_true, y_pred).tolist()

    tp = cm[1][1] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  [SVM] Acc={acc:.1%} F1={f1:.3f} AUC={auc:.3f} "
          f"Prec={prec:.1%} Recall={recall:.1%}")

    return {
        "stock": stock, "model_type": "svm",
        "n_test":           len(y_true),
        "n_pca_components": data["n_components"],
        "accuracy":     round(float(acc), 4),
        "f1_score":     round(float(f1),  4),
        "auc_roc":      round(float(auc), 4),
        "precision_up": round(prec,       4),
        "recall_up":    round(recall,     4),
        "confusion_matrix": cm,
        "y_prob": y_prob.tolist(),
        "y_true": y_true.tolist(),
        "pca_variance_ratio": data["pca_variance_ratio"],
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Down/Flat", "Up"], zero_division=0),
    }


# ============================================================
# STEP 5: Plot functions
# ============================================================
def plot_pca_variance(result, pca_variance_ratio):
    """Scree plot แสดง cumulative explained variance ของ PCA"""
    stock     = result["stock"]
    cum_var   = np.cumsum(pca_variance_ratio)
    n_comp    = result["n_pca_components"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker=".", markersize=4,
            color="purple", lw=1.5)
    ax.axhline(pca_variance, color="tomato", linestyle="--", lw=1.5,
               label=f"{pca_variance:.0%} threshold → {n_comp} components")
    ax.axvline(n_comp, color="tomato", linestyle=":", lw=1)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"SVM PCA Scree Plot — {stock}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / f"svm_{stock}_pca.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟ: {path}")


def plot_confusion_matrix(result):
    stock = result["stock"]
    cm    = np.array(result["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    plt.colorbar(im, ax=ax)

    classes    = ["Down/Flat", "Up"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"SVM Confusion Matrix — {stock}", fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=14)

    plt.tight_layout()
    path = plot_dir / f"svm_{stock}_cm.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟ: {path}")


def plot_roc(result):
    stock  = result["stock"]
    y_true = np.array(result["y_true"])
    y_prob = np.array(result["y_prob"])
    auc    = result["auc_roc"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="purple", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"SVM ROC Curve — {stock}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / f"svm_{stock}_roc.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟ: {path}")


def plot_all_summary(all_results):
    stocks = [r["stock"] for r in all_results]
    accs   = [r["accuracy"] for r in all_results]
    aucs   = [r["auc_roc"]  for r in all_results]
    x      = np.arange(len(stocks))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="purple",     alpha=0.75)
    bars2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC",  color="darkorange", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(stocks)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("SVM (RBF) — Summary (All Stocks)", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Baseline 0.5")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = plot_dir / "svm_summary.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟสรุป: {path}")


# ============================================================
# STEP 6: MAIN
# ============================================================
def run():
    np.random.seed(random_state)

    print("=" * 60)
    print("SVM (RBF) — News Embedding + Price → Open Direction")
    print("=" * 60)

    df, vec_cols = load_data()
    all_results  = []

    for stock in stocks:
        print(f"\n{'='*50}\nหุ้น: {stock}\n{'='*50}")

        data = prepare_stock(df, stock, vec_cols)
        if data is None:
            continue

        print(f"  train={data['n_train']} val={data['n_val']} "
              f"test={data['n_test']} up%={data['y_train'].mean():.1%}")

        svm, best_params, best_cv_auc, train_acc = tune_svm(data)
        joblib.dump(svm, model_dir / f"svm_{stock}.pkl")

        result = evaluate(svm, data, stock)
        result["train_acc"]   = train_acc
        result["best_params"] = best_params
        result["best_cv_auc"] = round(float(best_cv_auc), 4)
        result["overfit_gap"] = round(train_acc - result["accuracy"], 4)

        # กราฟ
        plot_pca_variance(result, data["pca_variance_ratio"])
        plot_confusion_matrix(result)
        plot_roc(result)

        all_results.append(result)

    if all_results:
        plot_all_summary(all_results)

    # ── สรุปผล ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("สรุปผล SVM (RBF)")
    print("=" * 60)
    print(f"{'หุ้น':8s} {'Acc':>8s} {'F1':>7s} {'AUC':>7s} {'PCA':>5s} {'N':>5s}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['stock']:8s} {r['accuracy']:>8.1%} {r['f1_score']:>7.3f} "
              f"{r['auc_roc']:>7.3f} {r['n_pca_components']:>5d} {r['n_test']:>5d}")

    if all_results:
        print(f"\n  avg Acc: {np.mean([r['accuracy'] for r in all_results]):.1%}  "
              f"|  avg AUC: {np.mean([r['auc_roc'] for r in all_results]):.3f}")

    print("\nOverfitting check (train_acc vs test_acc):")
    for r in all_results:
        flag = "⚠ overfit" if r["overfit_gap"] > 0.10 else "✓"
        print(f"  {r['stock']:8s} train={r['train_acc']:.1%} "
              f"test={r['accuracy']:.1%} gap={r['overfit_gap']:+.1%} {flag}")

    print("\nBest hyperparameters per stock:")
    for r in all_results:
        print(f"  {r['stock']:8s} {r['best_params']}  CV_AUC={r['best_cv_auc']:.3f}")

    # ── บันทึก JSON ──────────────────────────────────────────
    def to_py(o):
        if isinstance(o, dict):                       return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list):                       return [to_py(i) for i in o]
        if isinstance(o, (np.floating, np.integer)):  return float(o)
        return o

    skip_keys = {"y_prob", "y_true", "pca_variance_ratio"}
    results_clean = [{k: v for k, v in r.items() if k not in skip_keys}
                     for r in all_results]

    report_path = model_dir / "svm_evaluation_report.json"
    report_path.write_text(
        json.dumps(to_py(results_clean), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึกผลที่: {report_path}")
    print(f"บันทึกกราฟทั้งหมดใน: {plot_dir}/")
    return all_results


if __name__ == "__main__":
    run()
    print("SVM เทรนเสร็จสมบูรณ์!")
