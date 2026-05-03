# train_lstm.py
# เทรน LSTM สำหรับทำนายทิศทาง Open Price จากข่าว + ราคา
#
# Input  : final_5year_dataset.csv
# Output : models/lstm_{stock}.keras
#          models/scaler_{stock}.pkl
#          models/lstm_evaluation_report.json
#          models/lstm_training_history.json
#          models/model_meta.json
#          plots/lstm_{stock}_history.png
#          plots/lstm_{stock}_cm.png
#          plots/lstm_{stock}_roc.png

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
data_file         = "final_5year_dataset.csv"
model_dir_path    = "models"
plot_dir_path     = "plots"
stocks            = ["PTT", "AOT", "DELTA", "ADVANC", "SCB"]
price_cols        = ["Open", "Prev_Close", "High", "Low", "Close", "Volume"]
min_samples       = 80
train_ratio       = 0.70
val_ratio         = 0.15
units_1           = 64
units_2           = 32
dropout           = 0.40
recurrent_dropout = 0.00
epochs            = 200
batch_size        = 32
lr                = 0.001
es_patience       = 200
lr_patience       = 200
lr_factor         = 0.5
lr_min            = 1e-6
threshold         = 0.50

MODEL_DIR = Path(model_dir_path)
PLOT_DIR  = Path(plot_dir_path)
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


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
# STEP 2: เตรียม X, y แยกตามหุ้น
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
    y       = g["Target_Label"].values.astype(np.float32)
    X_raw   = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    scaler = StandardScaler()
    X_raw[:n_train] = scaler.fit_transform(X_raw[:n_train])
    X_raw[n_train:] = scaler.transform(X_raw[n_train:])

    joblib.dump(scaler, MODEL_DIR / f"scaler_{stock}.pkl")

    X = X_raw.reshape(n, 1, X_raw.shape[1])

    return {
        "X_train":    X[:n_train],
        "y_train":    y[:n_train],
        "X_val":      X[n_train : n_train + n_val],
        "y_val":      y[n_train : n_train + n_val],
        "X_test":     X[n_train + n_val :],
        "y_test":     y[n_train + n_val :],
        "dates_test": g["Target_Date"].values[n_train + n_val :],
        "n_features": X_raw.shape[1],
        "n_train":    n_train,
        "n_val":      n_val,
        "n_test":     n - n_train - n_val,
    }


# ============================================================
# STEP 3: สร้างโมเดล LSTM
# ============================================================
def build_model(n_features):
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(1, n_features)),
            keras.layers.LSTM(units_1, return_sequences=True,
                              dropout=dropout, recurrent_dropout=recurrent_dropout),
            keras.layers.BatchNormalization(),
            keras.layers.LSTM(units_2, return_sequences=False,
                              dropout=dropout, recurrent_dropout=recurrent_dropout),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="LSTM_model",
    )
    return model


# ============================================================
# STEP 4: เทรน
# ============================================================
def train(model, data, model_path):
    import tensorflow as tf
    from tensorflow import keras

    y_tr    = data["y_train"]
    classes = np.unique(y_tr)
    if len(classes) < 2:
        print("y_train มีแค่ 1 class → ข้าม")
        return None

    cw           = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = dict(zip(classes.astype(int), cw))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=es_patience,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lr_factor,
            patience=lr_patience, min_lr=lr_min, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss",
            save_best_only=True, verbose=0),
    ]

    h = model.fit(
        data["X_train"], y_tr,
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    best_ep = int(np.argmin(h.history["val_loss"]))
    va_key  = "val_accuracy" if "val_accuracy" in h.history else "val_acc"
    return {
        "history":      h.history,
        "best_epoch":   best_ep + 1,
        "total_epochs": len(h.history["loss"]),
        "train_acc":    round(float(h.history["accuracy"][best_ep]), 4),
        "val_acc":      round(float(h.history[va_key][best_ep]), 4),
    }


# ============================================================
# STEP 5: ประเมิน
# ============================================================
def evaluate(model, data, stock):
    y_prob = model.predict(data["X_test"], verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = data["y_test"].astype(int)

    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    auc    = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    cm     = confusion_matrix(y_true, y_pred).tolist()

    tp = cm[1][1] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  [LSTM] Acc={acc:.1%} F1={f1:.3f} AUC={auc:.3f} "
          f"Prec={prec:.1%} Recall={recall:.1%}")

    return {
        "stock": stock, "model_type": "lstm",
        "n_test":       len(y_true),
        "accuracy":     round(float(acc), 4),
        "f1_score":     round(float(f1),  4),
        "auc_roc":      round(float(auc), 4),
        "precision_up": round(prec,       4),
        "recall_up":    round(recall,     4),
        "confusion_matrix": cm,
        "y_prob": y_prob.tolist(),
        "y_true": y_true.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Down/Flat", "Up"], zero_division=0),
    }


# ============================================================
# STEP 6: Plot functions
# ============================================================
def plot_history(history, stock):
    """กราฟ Loss และ Accuracy ระหว่างเทรน"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"LSTM Training History — {stock}", fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(history["loss"],     label="Train Loss",  color="steelblue")
    axes[0].plot(history["val_loss"], label="Val Loss",    color="tomato", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    va_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    axes[1].plot(history["accuracy"], label="Train Acc",  color="steelblue")
    axes[1].plot(history[va_key],     label="Val Acc",    color="tomato", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / f"lstm_{stock}_history.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟ: {path}")


def plot_confusion_matrix(result):
    """Heatmap ของ Confusion Matrix"""
    stock = result["stock"]
    cm    = np.array(result["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    classes = ["Down/Flat", "Up"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"LSTM Confusion Matrix — {stock}", fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}",
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=14)

    plt.tight_layout()
    path = PLOT_DIR / f"lstm_{stock}_cm.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟ: {path}")


def plot_roc(result):
    """ROC Curve"""
    stock  = result["stock"]
    y_true = np.array(result["y_true"])
    y_prob = np.array(result["y_prob"])
    auc    = result["auc_roc"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"LSTM ROC Curve — {stock}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / f"lstm_{stock}_roc.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟ: {path}")


def plot_all_summary(all_results):
    """Bar chart สรุป Acc / AUC ทุกหุ้น"""
    stocks = [r["stock"] for r in all_results]
    accs   = [r["accuracy"] for r in all_results]
    aucs   = [r["auc_roc"]  for r in all_results]
    x      = np.arange(len(stocks))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + width/2, aucs, width, label="AUC-ROC",  color="darkorange", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(stocks)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("LSTM — Summary (All Stocks)", fontsize=13, fontweight="bold")
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
    path = PLOT_DIR / "lstm_summary.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  บันทึกกราฟสรุป: {path}")


# ============================================================
# STEP 7: MAIN
# ============================================================
def run():
    import tensorflow as tf
    tf.random.set_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("LSTM — News Embedding + Price → Open Direction Prediction")
    print(f"TF {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 60)

    df, vec_cols = load_data()
    n_features   = len(vec_cols) + len(price_cols)

    all_results = []
    all_history = {}
    all_overfit = {}

    for stock in stocks:
        print(f"\n{'='*50}\nหุ้น: {stock}\n{'='*50}")

        data = prepare_stock(df, stock, vec_cols)
        if data is None:
            continue

        print(f"  train={data['n_train']} val={data['n_val']} "
              f"test={data['n_test']} up%={data['y_train'].mean():.1%}")

        tf.keras.backend.clear_session()
        model      = build_model(n_features)
        model_path = MODEL_DIR / f"lstm_{stock}.keras"

        hist_info = train(model, data, model_path)
        if hist_info is None:
            continue

        # กราฟ Training History
        plot_history(hist_info["history"], stock)

        best_model = tf.keras.models.load_model(model_path)
        result     = evaluate(best_model, data, stock)

        # กราฟ Confusion Matrix และ ROC
        plot_confusion_matrix(result)
        plot_roc(result)

        gap = hist_info["train_acc"] - hist_info["val_acc"]
        result.update({
            "best_epoch":   hist_info["best_epoch"],
            "total_epochs": hist_info["total_epochs"],
            "train_acc":    hist_info["train_acc"],
            "val_acc":      hist_info["val_acc"],
            "overfit_gap":  round(gap, 4),
        })
        all_results.append(result)
        all_history[stock] = {
            k: [float(v) for v in vals]
            for k, vals in hist_info["history"].items()
        }
        all_overfit[stock] = {
            "train_acc": hist_info["train_acc"],
            "val_acc":   hist_info["val_acc"],
            "gap":       round(gap, 4),
        }

    # กราฟสรุปทุกหุ้น
    if all_results:
        plot_all_summary(all_results)

    # ── สรุปผล ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("สรุปผล LSTM")
    print("=" * 60)
    print(f"{'หุ้น':8s} {'Acc':>8s} {'F1':>7s} {'AUC':>7s} {'N':>6s}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['stock']:8s} {r['accuracy']:>8.1%} {r['f1_score']:>7.3f} "
              f"{r['auc_roc']:>7.3f} {r['n_test']:>6d}")

    if all_results:
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        avg_auc = np.mean([r["auc_roc"]  for r in all_results])
        print(f"\n  avg Acc: {avg_acc:.1%}  |  avg AUC: {avg_auc:.3f}")

    print("\nOverfitting check:")
    for stock, info in all_overfit.items():
        flag = "⚠ overfit" if info["gap"] > 0.10 else "✓"
        print(f"  {stock:8s} train={info['train_acc']:.1%} "
              f"val={info['val_acc']:.1%} gap={info['gap']:+.1%} {flag}")

    # ── บันทึกไฟล์ ───────────────────────────────────────────
    def to_py(o):
        if isinstance(o, dict):                       return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list):                       return [to_py(i) for i in o]
        if isinstance(o, (np.floating, np.integer)):  return float(o)
        return o

    # ลบ y_prob / y_true ออกจาก JSON (เก็บไว้ใช้ plot เท่านั้น)
    results_clean = [{k: v for k, v in r.items() if k not in ("y_prob", "y_true")}
                     for r in all_results]

    (MODEL_DIR / "lstm_evaluation_report.json").write_text(
        json.dumps(to_py(results_clean), ensure_ascii=False, indent=2), encoding="utf-8")
    (MODEL_DIR / "lstm_training_history.json").write_text(
        json.dumps(to_py(all_history), ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "created_at":      datetime.now().isoformat(),
        "model_type":      "lstm",
        "n_features":      n_features,
        "feature_cols":    vec_cols + price_cols,
        "stocks":          stocks,
        "model_dir":       str(MODEL_DIR),
        "threshold":       threshold,
        "news_shift_days": 1,
        "embed_model":     "airesearch/wangchanberta-base-att-spm-uncased",
    }
    (MODEL_DIR / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nบันทึกโมเดลทั้งหมดใน: {MODEL_DIR}/")
    print(f"บันทึกกราฟทั้งหมดใน: {PLOT_DIR}/")
    return all_results


if __name__ == "__main__":
    run()
    print("LSTM เทรนเสร็จสมบูรณ์!")
