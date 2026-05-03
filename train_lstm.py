import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

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

# หาคอลัมน์ embedding (vec_0, vec_1, ...)
vec_cols = [c for c in df.columns if c.startswith("vec_")]
vec_cols = sorted(vec_cols, key=lambda x: int(x.split("_")[1]))

print(f"จำนวนแถวทั้งหมด: {len(df):,} | หุ้น: {df['Stock'].nunique()} | features: {len(vec_cols) + len(price_cols)}")
print("\nจำนวนแถวต่อหุ้น:")
for stock, count in df.groupby("Stock").size().items():
    flag = "✅" if count >= 60 else "⚠ น้อย"
    print(f"  {stock}: {count} แถว  {flag}")


# ===== ฟังก์ชันทำ Sliding Window =====

def make_sequences(X, y, window):
    # เปลี่ยน X กับ y ให้เป็น sequence ตาม window ที่กำหนด
    # เช่น window=1 คือ ใช้ข้อมูลวันนี้ ทำนายวันพรุ่งนี้
    Xs = []
    ys = []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ===== วนลูปเทรนทีละหุ้น =====

tf.random.set_seed(42)
np.random.seed(42)

all_results = []
all_history = {}
all_overfit = {}

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
    y       = g["Target_Label"].values.astype(np.float32)

    # แทน NaN ด้วย 0
    X_vec   = np.nan_to_num(X_vec,   nan=0.0, posinf=0.0, neginf=0.0)
    X_price = np.nan_to_num(X_price, nan=0.0, posinf=0.0, neginf=0.0)

    # แบ่ง train 70% / val 15% / test 15%
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

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
    joblib.dump(vec_scaler, f"models/scaler_{stock}.pkl")
    joblib.dump(pca,        f"models/pca_{stock}_lstm.pkl")

    # Scale ข้อมูลราคา
    price_scaler = StandardScaler()
    X_price[:n_train] = price_scaler.fit_transform(X_price[:n_train])
    X_price[n_train:] = price_scaler.transform(X_price[n_train:])
    joblib.dump(price_scaler, f"models/price_scaler_{stock}.pkl")

    # รวม vec (ที่ลดมิติแล้ว) + price เข้าด้วยกัน
    X_train_full = np.hstack([X_vec_train_pca, X_price[:n_train]])
    X_rest_full  = np.hstack([X_vec_rest_pca,  X_price[n_train:]])
    X_full       = np.vstack([X_train_full, X_rest_full])
    n_features   = X_full.shape[1]

    # ทำ sliding window (window=1 คือ ใช้วันนี้ทำนายพรุ่งนี้)
    Xs, ys = make_sequences(X_full, y, window=1)

    n_tr_seq = max(n_train - 1, 1)
    n_va_seq = n_val

    X_train = Xs[:n_tr_seq]
    y_train = ys[:n_tr_seq]
    X_val   = Xs[n_tr_seq : n_tr_seq + n_va_seq]
    y_val   = ys[n_tr_seq : n_tr_seq + n_va_seq]
    X_test  = Xs[n_tr_seq + n_va_seq:]
    y_test  = ys[n_tr_seq + n_va_seq:]

    print(f"  train={len(X_train)} val={len(X_val)} test={len(X_test)} features={n_features}")
    print(f"  สัดส่วน UP ใน train: {y_train.mean():.1%}")

    if len(X_train) < 10 or len(X_test) < 5:
        print(f"  {stock}: ข้อมูลน้อยเกินหลัง window → ข้าม")
        continue

    # จัดการ imbalanced class
    classes = np.unique(y_train)
    if len(classes) < 2:
        print("  y_train มีแค่ 1 class → ข้าม")
        continue

    cw           = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.astype(int), cw))

    # สร้างโมเดล LSTM
    tf.keras.backend.clear_session()
    model = keras.Sequential([
        keras.layers.Input(shape=(1, n_features)),
        keras.layers.LSTM(
            32,
            dropout=0.30,
            kernel_regularizer=regularizers.l2(1e-4),
            recurrent_regularizer=regularizers.l2(1e-4)
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )

    model_path = f"models/lstm_{stock}.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    # เทรนโมเดล
    h = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=8,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )

    # หา best epoch จาก val_loss ต่ำสุด
    best_ep = int(np.argmin(h.history["val_loss"]))
    va_key  = "val_accuracy" if "val_accuracy" in h.history else "val_acc"
    train_acc_best = round(float(h.history["accuracy"][best_ep]), 4)
    val_acc_best   = round(float(h.history[va_key][best_ep]), 4)

    print(f"  best epoch: {best_ep + 1}/{len(h.history['loss'])} | train_acc={train_acc_best:.1%} val_acc={val_acc_best:.1%}")

    # กราฟ training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"LSTM Training History — {stock}", fontsize=13, fontweight="bold")

    axes[0].plot(h.history["loss"],     label="Train Loss", color="steelblue")
    axes[0].plot(h.history["val_loss"], label="Val Loss",   color="tomato", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(h.history["accuracy"],  label="Train Acc", color="steelblue")
    axes[1].plot(h.history[va_key],      label="Val Acc",   color="tomato", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/lstm_{stock}_history.png", dpi=120)
    plt.close()

    # โหลด best model แล้วประเมินผลบน test set
    best_model = tf.keras.models.load_model(model_path)

    y_prob = best_model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.50).astype(int)
    y_true = y_test.astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    cm  = confusion_matrix(y_true, y_pred).tolist()

    tp = cm[1][1] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  [LSTM] Acc={acc:.1%}  F1={f1:.3f}  AUC={auc:.3f}  Prec={prec:.1%}  Recall={recall:.1%}")
    print(classification_report(y_true, y_pred, target_names=["Down/Flat", "Up"], zero_division=0))

    # กราฟ Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im      = ax.imshow(np.array(cm), cmap="Blues")
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
    plt.savefig(f"plots/lstm_{stock}_cm.png", dpi=120)
    plt.close()

    # กราฟ ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {stock}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/lstm_{stock}_roc.png", dpi=120)
    plt.close()

    # เก็บผลไว้สรุปทีหลัง
    gap = train_acc_best - val_acc_best
    result = {
        "stock":         stock,
        "n_test":        len(y_true),
        "accuracy":      round(float(acc), 4),
        "f1_score":      round(float(f1),  4),
        "auc_roc":       round(float(auc), 4),
        "precision_up":  round(prec,       4),
        "recall_up":     round(recall,     4),
        "confusion_matrix": cm,
        "best_epoch":    best_ep + 1,
        "total_epochs":  len(h.history["loss"]),
        "train_acc":     train_acc_best,
        "val_acc":       val_acc_best,
        "overfit_gap":   round(gap, 4),
        "pca_n_comp":    n_comp,
        "y_prob":        y_prob.tolist(),
        "y_true":        y_true.tolist(),
    }
    all_results.append(result)

    all_history[stock] = {}
    for key, vals in h.history.items():
        all_history[stock][key] = [float(v) for v in vals]

    all_overfit[stock] = {
        "train_acc": train_acc_best,
        "val_acc":   val_acc_best,
        "gap":       round(gap, 4),
    }


# ===== กราฟสรุปทุกหุ้น =====

if all_results:
    stock_names = [r["stock"]    for r in all_results]
    accs        = [r["accuracy"] for r in all_results]
    aucs        = [r["auc_roc"]  for r in all_results]
    x     = np.arange(len(stock_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC",  color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(stock_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("LSTM Summary (All Stocks)", fontsize=13, fontweight="bold")
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
    plt.savefig("plots/lstm_summary.png", dpi=120)
    plt.close()
    print("\nบันทึกกราฟสรุปแล้ว: plots/lstm_summary.png")


# ===== พิมพ์สรุปผล =====

print("\n" + "=" * 50)
print("สรุปผล LSTM")
print("=" * 50)
print(f"{'หุ้น':8s} {'Acc':>8s} {'F1':>7s} {'AUC':>7s} {'PCA':>5s} {'N':>6s}")
print("-" * 50)
for r in all_results:
    print(f"{r['stock']:8s} {r['accuracy']:>8.1%} {r['f1_score']:>7.3f} {r['auc_roc']:>7.3f} {r['pca_n_comp']:>5d} {r['n_test']:>6d}")

if all_results:
    avg_acc = np.mean([r["accuracy"] for r in all_results])
    avg_auc = np.mean([r["auc_roc"]  for r in all_results])
    print(f"\navg Acc: {avg_acc:.1%}  |  avg AUC: {avg_auc:.3f}")

print("\nOverfitting check:")
for stock, info in all_overfit.items():
    flag = "⚠ overfit" if info["gap"] > 0.10 else "✓"
    print(f"  {stock}: train={info['train_acc']:.1%}  val={info['val_acc']:.1%}  gap={info['gap']:+.1%}  {flag}")


# ===== บันทึกผลลัพธ์เป็น JSON =====

# ฟังก์ชันแปลง numpy type ให้ json.dumps รับได้
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

with open("models/lstm_evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(to_serializable(results_clean), f, ensure_ascii=False, indent=2)

with open("models/lstm_training_history.json", "w", encoding="utf-8") as f:
    json.dump(to_serializable(all_history), f, ensure_ascii=False, indent=2)

meta = {
    "created_at":   datetime.now().isoformat(),
    "model_type":   "lstm",
    "window_size":  1,
    "pca_variance": 0.90,
    "lstm_units":   32,
    "l2_reg":       1e-4,
    "stocks":       stocks,
    "threshold":    0.50,
}
with open("models/model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\nบันทึกโมเดลทั้งหมดใน: models/")
print("บันทึกกราฟทั้งหมดใน:  plots/")
print("\nเทรน LSTM เสร็จสมบูรณ์!")
