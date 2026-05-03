import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# สร้างโฟลเดอร์เก็บโมเดลและกราฟ
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

stocks     = ["PTT", "AOT", "DELTA", "ADVANC", "SCB"]
price_cols = ["Open", "Prev_Close", "High", "Low", "Close", "Volume"]


# ===== โหลดข้อมูล =====

print("โหลด final_5year_dataset.csv...")
df = pd.read_csv("final_5year_dataset.csv", parse_dates=["Target_Date"])
df = df.sort_values(["Stock", "Target_Date"]).reset_index(drop=True)

vec_cols = [c for c in df.columns if c.startswith("vec_")]
vec_cols = sorted(vec_cols, key=lambda x: int(x.split("_")[1]))

print(f"จำนวนแถวทั้งหมด: {len(df):,} | หุ้น: {df['Stock'].nunique()} | features: {len(vec_cols) + len(price_cols)}")
print("\nจำนวนแถวต่อหุ้น:")
for stock, count in df.groupby("Stock").size().items():
    flag = "✅" if count >= 60 else "⚠ น้อย"
    print(f"  {stock}: {count} แถว  {flag}")


# ===== วนลูปเทรนทีละหุ้น =====

np.random.seed(42)

all_results = []

for stock in stocks:
    print(f"\n{'='*50}")
    print(f"หุ้น: {stock}")
    print(f"{'='*50}")

    # กรองข้อมูลเฉพาะหุ้นนั้น
    g = df[df["Stock"] == stock].sort_values("Target_Date").reset_index(drop=True)
    n = len(g)

    if n < 60:
        print(f"  {stock}: มีแค่ {n} แถว น้อยเกินไป → ข้าม")
        continue

    # แยก features และ label
    X_vec   = g[vec_cols].values.astype(np.float32)
    X_price = g[price_cols].values.astype(np.float32)
    y       = g["Target_Label"].values.astype(int)

    # แทน NaN ด้วย 0
    X_vec   = np.nan_to_num(X_vec,   nan=0.0, posinf=0.0, neginf=0.0)
    X_price = np.nan_to_num(X_price, nan=0.0, posinf=0.0, neginf=0.0)

    # แบ่ง train 70% / val 15% / test 15%
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val

    # Scale ข้อมูล embedding (fit เฉพาะ train ป้องกัน data leakage)
    vec_scaler = StandardScaler()
    X_vec[:n_train] = vec_scaler.fit_transform(X_vec[:n_train])
    X_vec[n_train:] = vec_scaler.transform(X_vec[n_train:])

    # ลดมิติด้วย PCA ให้คง variance 90%
    pca = PCA(n_components=0.90, random_state=42)
    X_vec_train_pca = pca.fit_transform(X_vec[:n_train])
    X_vec_rest_pca  = pca.transform(X_vec[n_train:])
    n_comp = pca.n_components_
    print(f"  PCA: {X_vec.shape[1]} → {n_comp} components (var={pca.explained_variance_ratio_.sum():.1%})")

    # บันทึก scaler และ pca ไว้ใช้ตอน inference
    joblib.dump(vec_scaler, f"models/scaler_{stock}_rf.pkl")
    joblib.dump(pca,        f"models/pca_{stock}_rf.pkl")

    # Scale ข้อมูลราคา
    price_scaler = StandardScaler()
    X_price[:n_train] = price_scaler.fit_transform(X_price[:n_train])
    X_price[n_train:] = price_scaler.transform(X_price[n_train:])
    joblib.dump(price_scaler, f"models/price_scaler_{stock}_rf.pkl")

    # รวม vec (ที่ลดมิติแล้ว) + price เข้าด้วยกัน
    X_train = np.hstack([X_vec_train_pca, X_price[:n_train]])
    X_rest  = np.hstack([X_vec_rest_pca,  X_price[n_train:]])
    X_val   = X_rest[:n_val]
    X_test  = X_rest[n_val:]

    y_train = y[:n_train]
    y_val   = y[n_train : n_train + n_val]
    y_test  = y[n_train + n_val:]

    # ชื่อคอลัมน์หลัง PCA (ใช้แสดง feature importance)
    feature_names = [f"pca_{i}" for i in range(n_comp)] + price_cols

    print(f"  features หลัง PCA = {X_train.shape[1]}  (train={n_train} val={n_val} test={n_test})")
    print(f"  สัดส่วน UP ใน train: {y_train.mean():.1%}")

    # สร้างและเทรน Random Forest
    rf = RandomForestClassifier(
        n_estimators     = 500,
        max_depth        = 2,
        min_samples_leaf = 10,
        max_features     = "sqrt",
        max_samples      = 0.80,
        class_weight     = "balanced",
        random_state     = 42,
        n_jobs           = -1
    )
    rf.fit(X_train, y_train)

    # ดู accuracy บน train และ val
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc   = accuracy_score(y_val,   rf.predict(X_val))

    # ทำ Stratified K-Fold CV บน train+val รวมกัน
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    skf  = StratifiedKFold(n_splits=5, shuffle=False)
    cv_scores = cross_val_score(rf, X_tv, y_tv, cv=skf, scoring="roc_auc", n_jobs=-1)

    print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  | train_acc={train_acc:.1%}  val_acc={val_acc:.1%}")

    # บันทึกโมเดล
    joblib.dump(rf, f"models/rf_{stock}.pkl")

    # ประเมินผลบน test set
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    y_true = y_test

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    cm  = confusion_matrix(y_true, y_pred).tolist()

    tp = cm[1][1] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  [RF] Acc={acc:.1%}  F1={f1:.3f}  AUC={auc:.3f}  Prec={prec:.1%}  Recall={recall:.1%}")
    print(classification_report(y_true, y_pred, target_names=["Down/Flat", "Up"], zero_division=0))

    # ===== กราฟ Feature Importance =====
    importances = rf.feature_importances_
    top_idx  = np.argsort(importances)[::-1][:20]
    top_vals = importances[top_idx]
    top_cols = [feature_names[i] for i in top_idx]

    colors = ["darkorange" if c in price_cols else "steelblue" for c in top_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_cols)), top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"RF Feature Importance (Top 20) — {stock}", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    legend_elements = [
        Patch(facecolor="darkorange", label="Price Feature"),
        Patch(facecolor="steelblue",  label="PCA Component")
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"plots/rf_{stock}_importance.png", dpi=120)
    plt.close()

    # ===== กราฟ Confusion Matrix =====
    fig, ax = plt.subplots(figsize=(5, 4))
    im      = ax.imshow(np.array(cm), cmap="Greens")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Down/Flat", "Up"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Down/Flat", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {stock}", fontweight="bold")
    thresh = np.array(cm).max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i][j] > thresh else "black"
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color=color, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/rf_{stock}_cm.png", dpi=120)
    plt.close()

    # ===== กราฟ ROC Curve =====
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="forestgreen", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {stock}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/rf_{stock}_roc.png", dpi=120)
    plt.close()

    print(f"  บันทึกกราฟแล้ว: plots/rf_{stock}_*.png")

    # เก็บผลไว้สรุปทีหลัง
    top_named = [feature_names[i] for i in np.argsort(importances)[::-1][:10].tolist()]
    result = {
        "stock":          stock,
        "n_test":         len(y_true),
        "n_features":     len(importances),
        "accuracy":       round(float(acc),      4),
        "f1_score":       round(float(f1),        4),
        "auc_roc":        round(float(auc),       4),
        "precision_up":   round(prec,             4),
        "recall_up":      round(recall,           4),
        "confusion_matrix": cm,
        "train_acc":      round(float(train_acc), 4),
        "val_acc":        round(float(val_acc),   4),
        "overfit_gap":    round(float(train_acc - acc), 4),
        "cv_auc_mean":    round(float(cv_scores.mean()), 4),
        "cv_auc_std":     round(float(cv_scores.std()),  4),
        "pca_n_comp":     n_comp,
        "top10_features": top_named,
        "y_prob":         y_prob.tolist(),
        "y_true":         y_true.tolist(),
    }
    all_results.append(result)


# ===== กราฟสรุปทุกหุ้น =====

if all_results:
    stock_names = [r["stock"]    for r in all_results]
    accs        = [r["accuracy"] for r in all_results]
    aucs        = [r["auc_roc"]  for r in all_results]
    x     = np.arange(len(stock_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="forestgreen", alpha=0.85)
    bars2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC",  color="darkorange",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(stock_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest Summary (All Stocks)", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Baseline 0.5")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig("plots/rf_summary.png", dpi=120)
    plt.close()
    print("\nบันทึกกราฟสรุปแล้ว: plots/rf_summary.png")


# ===== พิมพ์สรุปผล =====

print("\n" + "=" * 58)
print("สรุปผล Random Forest")
print("=" * 58)
print(f"{'หุ้น':8s} {'Acc':>8s} {'F1':>7s} {'AUC':>7s} {'CV_AUC':>10s} {'PCA':>5s} {'N':>5s}")
print("-" * 58)
for r in all_results:
    print(f"{r['stock']:8s} {r['accuracy']:>8.1%} {r['f1_score']:>7.3f} "
          f"{r['auc_roc']:>7.3f} {r['cv_auc_mean']:>7.3f}±{r['cv_auc_std']:.3f} "
          f"{r['pca_n_comp']:>5d} {r['n_test']:>5d}")

if all_results:
    avg_acc    = np.mean([r["accuracy"]    for r in all_results])
    avg_auc    = np.mean([r["auc_roc"]     for r in all_results])
    avg_cv_auc = np.mean([r["cv_auc_mean"] for r in all_results])
    print(f"\navg Acc: {avg_acc:.1%}  |  avg AUC: {avg_auc:.3f}  |  avg CV_AUC: {avg_cv_auc:.3f}")

print("\nOverfitting check (train_acc vs test_acc):")
for r in all_results:
    flag = "⚠ overfit" if r["overfit_gap"] > 0.15 else "✓"
    print(f"  {r['stock']:8s} train={r['train_acc']:.1%}  val={r['val_acc']:.1%}  "
          f"test={r['accuracy']:.1%}  gap={r['overfit_gap']:+.1%}  {flag}")


# ===== บันทึกผลลัพธ์เป็น JSON =====

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj

# เอา y_prob, y_true ออกก่อนบันทึก (ข้อมูลเยอะเกิน)
results_clean = []
for r in all_results:
    clean = {k: v for k, v in r.items() if k not in ("y_prob", "y_true")}
    results_clean.append(clean)

with open("models/rf_evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(to_serializable(results_clean), f, ensure_ascii=False, indent=2)

print("\nบันทึกโมเดลทั้งหมดใน: models/")
print("บันทึกกราฟทั้งหมดใน:  plots/")
print("\nเทรน Random Forest เสร็จสมบูรณ์!")
