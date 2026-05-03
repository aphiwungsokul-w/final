# train_tfidf_models.py
# เทรน Random Forest + SVM + LSTM ด้วย TF-IDF feature
# และเปรียบเทียบกับผล WangchanBERTa embedding
#
# Input  : tfidf_dataset.csv       (จาก prepare_tfidf.py)
# Output : models_tfidf/
#            rf_tfidf_{stock}.pkl
#            svm_tfidf_{stock}.pkl
#            lstm_tfidf_{stock}.keras
#            scaler_tfidf_{stock}.pkl
#            pca_tfidf_{stock}.pkl
#            tfidf_evaluation_report.json
#          plots_tfidf/
#            rf_tfidf_{stock}_importance.png
#            {model}_{stock}_cm.png
#            {model}_{stock}_roc.png
#            comparison_bert_vs_tfidf.png   ← กราฟเปรียบเทียบหลัก

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_FILE      = 'tfidf_dataset.csv'
MODEL_DIR      = Path('models_tfidf')
PLOT_DIR       = Path('plots_tfidf')
BERT_RF_REPORT = Path('models/rf_evaluation_report.json')
BERT_SVM_REPORT= Path('models/svm_evaluation_report.json')
BERT_LSTM_REPORT=Path('models/lstm_evaluation_report.json')

STOCKS         = ['PTT', 'AOT', 'DELTA', 'ADVANC', 'SCB']
PRICE_COLS     = ['Open', 'Prev_Close', 'High', 'Low', 'Close', 'Volume']
MIN_SAMPLES    = 80
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
THRESHOLD      = 0.50
RANDOM_STATE   = 42

# Random Forest
RF_N_EST       = 300
RF_MAX_DEPTH   = 5
RF_MIN_LEAF    = 10

# SVM
SVM_PARAM_GRID = {
    'C':     [0.01, 0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel':['rbf'],
}
SVM_CV_FOLDS   = 5
PCA_VARIANCE   = 0.95

# LSTM
LSTM_UNITS_1   = 64
LSTM_UNITS_2   = 32
LSTM_DROPOUT   = 0.40
LSTM_EPOCHS    = 30
LSTM_BATCH     = 16
LSTM_LR        = 0.001
LSTM_ES_PAT    = 20
LSTM_LR_PAT    = 7

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: โหลดข้อมูล
# ============================================================
def load_data():
    print(f'โหลด {DATA_FILE}...')
    df = pd.read_csv(DATA_FILE, parse_dates=['Target_Date'])
    df = df.sort_values(['Stock', 'Target_Date']).reset_index(drop=True)

    tfidf_cols = sorted(
        [c for c in df.columns if c.startswith('tfidf_')],
        key=lambda x: int(x.split('_')[1]),
    )
    if not tfidf_cols:
        raise ValueError('ไม่พบคอลัมน์ tfidf_* — รัน prepare_tfidf.py ก่อน')

    print(f'rows={len(df):,} | stocks={df["Stock"].nunique()} | '
          f'tfidf_dim={len(tfidf_cols)} | price_cols={len(PRICE_COLS)}')

    print('\nจำนวนแถวต่อหุ้น:')
    for s, n in df.groupby('Stock').size().items():
        flag = '✅' if n >= MIN_SAMPLES else '⚠ น้อย'
        print(f'  {s:8s}: {n:4d}  {flag}')

    return df, tfidf_cols


# ============================================================
# STEP 2: เตรียม X, y แยกรายหุ้น
# ============================================================
def prepare_stock(df, stock, tfidf_cols, scale=True):
    g = df[df['Stock'] == stock].sort_values('Target_Date').reset_index(drop=True)
    n = len(g)

    if n < MIN_SAMPLES:
        print(f'{stock}: {n} แถว < {MIN_SAMPLES} → ข้าม')
        return None

    X_tfidf = g[tfidf_cols].values.astype(np.float32)
    X_price = g[PRICE_COLS].values.astype(np.float32)
    X_raw   = np.hstack([X_tfidf, X_price])
    y       = g['Target_Label'].values.astype(int)
    X_raw   = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_raw[:n_train])
    X_val_s   = scaler.transform(X_raw[n_train : n_train + n_val])
    X_test_s  = scaler.transform(X_raw[n_train + n_val :])
    joblib.dump(scaler, MODEL_DIR / f'scaler_tfidf_{stock}.pkl')

    return {
        'X_train_raw': X_train_s, 'X_val_raw': X_val_s, 'X_test_raw': X_test_s,
        'y_train': y[:n_train],
        'y_val':   y[n_train : n_train + n_val],
        'y_test':  y[n_train + n_val :],
        'dates_test': g['Target_Date'].values[n_train + n_val :],
        'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
        'n_features': X_raw.shape[1],
        'tfidf_cols': tfidf_cols,
    }


# ============================================================
# STEP 3a: Random Forest
# ============================================================
def train_rf(data, stock):
    print(f'\n  [RF] เทรน {stock}...')
    X_fit = np.vstack([data['X_train_raw'], data['X_val_raw']])
    y_fit = np.concatenate([data['y_train'], data['y_val']])

    rf = RandomForestClassifier(
        n_estimators     = RF_N_EST,
        max_depth        = RF_MAX_DEPTH,
        min_samples_leaf = RF_MIN_LEAF,
        max_features     = 'sqrt',
        class_weight     = 'balanced',
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    rf.fit(X_fit, y_fit)
    joblib.dump(rf, MODEL_DIR / f'rf_tfidf_{stock}.pkl')

    y_prob  = rf.predict_proba(data['X_test_raw'])[:, 1]
    result  = _compute_metrics(y_prob, data['y_test'], stock, 'rf_tfidf')
    train_acc = accuracy_score(data['y_train'], rf.predict(data['X_train_raw']))
    result['train_acc']   = round(float(train_acc), 4)
    result['overfit_gap'] = round(train_acc - result['accuracy'], 4)

    # Feature Importance (Top 20 — แยก TF-IDF vs Price)
    imp        = rf.feature_importances_
    all_cols   = data['tfidf_cols'] + PRICE_COLS
    top_idx    = np.argsort(imp)[::-1][:20]
    result['top10_features'] = [all_cols[i] for i in top_idx[:10]]
    result['importances']    = imp.tolist()
    result['all_cols']       = all_cols

    _plot_rf_importance(imp, all_cols, stock)
    _plot_cm(result, stock, 'rf_tfidf', cmap='Greens')
    _plot_roc(result, stock, 'rf_tfidf', color='forestgreen')

    print(f'  Acc={result["accuracy"]:.1%} F1={result["f1_score"]:.3f} '
          f'AUC={result["auc_roc"]:.3f}  gap={result["overfit_gap"]:+.1%}')
    return result


# ============================================================
# STEP 3b: SVM + PCA
# ============================================================
def train_svm(data, stock):
    print(f'\n  [SVM] เทรน {stock} (GridSearchCV)...')

    pca       = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_train_p = pca.fit_transform(data['X_train_raw'])
    X_val_p   = pca.transform(data['X_val_raw'])
    X_test_p  = pca.transform(data['X_test_raw'])
    joblib.dump(pca, MODEL_DIR / f'pca_tfidf_{stock}.pkl')

    n_comp  = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum()
    print(f'  PCA: {data["n_features"]} → {n_comp} components (var={var_exp:.1%})')

    X_fit = np.vstack([X_train_p, X_val_p])
    y_fit = np.concatenate([data['y_train'], data['y_val']])

    tscv = TimeSeriesSplit(n_splits=SVM_CV_FOLDS)
    gs   = GridSearchCV(
        SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE),
        SVM_PARAM_GRID, cv=tscv, scoring='roc_auc',
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_fit, y_fit)
    print(f'  best params: {gs.best_params_}  CV AUC={gs.best_score_:.3f}')

    joblib.dump(gs.best_estimator_, MODEL_DIR / f'svm_tfidf_{stock}.pkl')

    y_prob    = gs.best_estimator_.predict_proba(X_test_p)[:, 1]
    result    = _compute_metrics(y_prob, data['y_test'], stock, 'svm_tfidf')
    train_acc = accuracy_score(data['y_train'],
                               gs.best_estimator_.predict(X_train_p))
    result.update({
        'train_acc':        round(float(train_acc), 4),
        'overfit_gap':      round(train_acc - result['accuracy'], 4),
        'best_params':      gs.best_params_,
        'best_cv_auc':      round(float(gs.best_score_), 4),
        'n_pca_components': int(n_comp),
    })

    _plot_cm(result, stock, 'svm_tfidf', cmap='Purples')
    _plot_roc(result, stock, 'svm_tfidf', color='purple')

    print(f'  Acc={result["accuracy"]:.1%} F1={result["f1_score"]:.3f} '
          f'AUC={result["auc_roc"]:.3f}  gap={result["overfit_gap"]:+.1%}')
    return result


# ============================================================
# STEP 3c: LSTM
# ============================================================
def train_lstm(data, stock):
    print(f'\n  [LSTM] เทรน {stock}...')
    try:
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
        print('  ⚠ ไม่พบ TensorFlow — ข้าม LSTM')
        return None

    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_STATE)

    n_feat    = data['n_features']
    X_train   = data['X_train_raw'].reshape(-1, 1, n_feat)
    X_val     = data['X_val_raw'].reshape(-1, 1, n_feat)
    X_test    = data['X_test_raw'].reshape(-1, 1, n_feat)
    y_train   = data['y_train'].astype(np.float32)
    y_val     = data['y_val'].astype(np.float32)

    classes   = np.unique(y_train)
    if len(classes) < 2:
        print('  ⚠ y_train มีแค่ 1 class — ข้าม')
        return None

    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.astype(int), cw))

    model = keras.Sequential([
        keras.layers.Input(shape=(1, n_feat)),
        keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True, dropout=LSTM_DROPOUT),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(LSTM_UNITS_2, return_sequences=False, dropout=LSTM_DROPOUT),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(LSTM_DROPOUT),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ], name='LSTM_tfidf')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')],
    )

    model_path = MODEL_DIR / f'lstm_tfidf_{stock}.keras'
    callbacks  = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=LSTM_ES_PAT,
            restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=LSTM_LR_PAT,
            min_lr=1e-6, verbose=0),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor='val_loss',
            save_best_only=True, verbose=0),
    ]

    h = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0,
    )

    best_ep   = int(np.argmin(h.history['val_loss']))
    va_key    = 'val_accuracy' if 'val_accuracy' in h.history else 'val_acc'
    best_model = tf.keras.models.load_model(model_path)

    y_prob    = best_model.predict(X_test, verbose=0).flatten()
    result    = _compute_metrics(y_prob, data['y_test'], stock, 'lstm_tfidf')
    result.update({
        'train_acc':    round(float(h.history['accuracy'][best_ep]), 4),
        'val_acc':      round(float(h.history[va_key][best_ep]), 4),
        'best_epoch':   best_ep + 1,
        'total_epochs': len(h.history['loss']),
    })
    result['overfit_gap'] = round(result['train_acc'] - result['accuracy'], 4)

    _plot_lstm_history(h.history, stock)
    _plot_cm(result, stock, 'lstm_tfidf', cmap='Blues')
    _plot_roc(result, stock, 'lstm_tfidf', color='steelblue')

    print(f'  Acc={result["accuracy"]:.1%} F1={result["f1_score"]:.3f} '
          f'AUC={result["auc_roc"]:.3f}  gap={result["overfit_gap"]:+.1%} '
          f'(epoch {result["best_epoch"]})')
    return result


# ============================================================
# SHARED: คำนวณ metrics
# ============================================================
def _compute_metrics(y_prob, y_test, stock, model_type):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    y_true = y_test.astype(int)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    cm   = confusion_matrix(y_true, y_pred).tolist()

    tp = cm[1][1] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'stock': stock, 'model_type': model_type,
        'n_test':       len(y_true),
        'accuracy':     round(float(acc),  4),
        'f1_score':     round(float(f1),   4),
        'auc_roc':      round(float(auc),  4),
        'precision_up': round(prec,        4),
        'recall_up':    round(recall,      4),
        'confusion_matrix': cm,
        'y_prob': y_prob.tolist(),
        'y_true': y_true.tolist(),
        'classification_report': classification_report(
            y_true, y_pred, target_names=['Down/Flat', 'Up'], zero_division=0),
    }


# ============================================================
# PLOT helpers
# ============================================================
def _plot_rf_importance(imp, all_cols, stock, top_n=20):
    tfidf_set = set(c for c in all_cols if c.startswith('tfidf_'))
    top_idx   = np.argsort(imp)[::-1][:top_n]
    top_vals  = imp[top_idx]
    top_cols  = [all_cols[i] for i in top_idx]
    colors    = ['darkorange' if c in PRICE_COLS else 'steelblue' for c in top_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_cols[::-1], fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title(f'RF (TF-IDF) Feature Importance — {stock}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    legend = [mpatches.Patch(facecolor='darkorange', label='Price'),
              mpatches.Patch(facecolor='steelblue',  label='TF-IDF term')]
    ax.legend(handles=legend, loc='lower right')
    plt.tight_layout()
    path = PLOT_DIR / f'rf_tfidf_{stock}_importance.png'
    plt.savefig(path, dpi=120); plt.close()
    print(f'  บันทึก: {path}')


def _plot_cm(result, stock, prefix, cmap='Blues'):
    cm  = np.array(result['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    classes    = ['Down/Flat', 'Up']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{prefix.upper()} Confusion Matrix — {stock}', fontweight='bold')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    path = PLOT_DIR / f'{prefix}_{stock}_cm.png'
    plt.savefig(path, dpi=120); plt.close()


def _plot_roc(result, stock, prefix, color='steelblue'):
    y_true = np.array(result['y_true'])
    y_prob = np.array(result['y_prob'])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC={result["auc_roc"]:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{prefix.upper()} ROC — {stock}', fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / f'{prefix}_{stock}_roc.png'
    plt.savefig(path, dpi=120); plt.close()


def _plot_lstm_history(history, stock):
    va_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'LSTM (TF-IDF) Training History — {stock}', fontweight='bold')
    axes[0].plot(history['loss'],     label='Train', color='steelblue')
    axes[0].plot(history['val_loss'], label='Val',   color='tomato', linestyle='--')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['accuracy'], label='Train', color='steelblue')
    axes[1].plot(history[va_key],     label='Val',   color='tomato', linestyle='--')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / f'lstm_tfidf_{stock}_history.png'
    plt.savefig(path, dpi=120); plt.close()


# ============================================================
# STEP 4: เปรียบเทียบ BERT vs TF-IDF
# ============================================================
def load_bert_results() -> dict:
    """โหลด evaluation report ของ BERT จาก models/ ถ้ามี"""
    bert = {'rf': {}, 'svm': {}, 'lstm': {}}

    for key, path in [('rf', BERT_RF_REPORT),
                      ('svm', BERT_SVM_REPORT),
                      ('lstm', BERT_LSTM_REPORT)]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                for r in data:
                    bert[key][r['stock']] = r
                print(f'  โหลด BERT {key}: {path}')
            except Exception as e:
                print(f'  ⚠ โหลด {path} ไม่ได้: {e}')
        else:
            print(f'  ⚠ ไม่พบ {path} — จะข้าม BERT ใน comparison')

    return bert


def plot_comparison(tfidf_results: dict, bert_results: dict):
    """
    กราฟเปรียบเทียบ TF-IDF vs BERT embedding
    แต่ละ model (RF, SVM, LSTM) แยก subplot
    """
    print('\n[PLOT] สร้างกราฟเปรียบเทียบ BERT vs TF-IDF...')

    model_keys  = ['rf', 'svm', 'lstm']
    model_names = ['Random Forest', 'SVM', 'LSTM']
    metrics     = ['accuracy', 'auc_roc', 'f1_score']
    metric_lbls = ['Accuracy', 'AUC-ROC', 'F1 Score']

    fig, axes = plt.subplots(len(model_keys), len(metrics),
                             figsize=(14, 4 * len(model_keys)))
    fig.suptitle('TF-IDF vs WangchanBERTa Embedding — Model Comparison',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, (mkey, mname) in enumerate(zip(model_keys, model_names)):
        tfidf_rows = tfidf_results.get(mkey, {})
        bert_rows  = bert_results.get(mkey, {})

        stocks_common = sorted(set(tfidf_rows.keys()) & set(bert_rows.keys())) \
                        or sorted(tfidf_rows.keys())

        for col, (metric, mlbl) in enumerate(zip(metrics, metric_lbls)):
            ax = axes[row][col]
            x  = np.arange(len(stocks_common))
            w  = 0.35

            tfidf_vals = [tfidf_rows.get(s, {}).get(metric, 0) for s in stocks_common]
            bert_vals  = [bert_rows.get(s, {}).get(metric, 0)  for s in stocks_common]

            b1 = ax.bar(x - w/2, tfidf_vals, w, label='TF-IDF',
                        color='#e67e22', alpha=0.85)
            b2 = ax.bar(x + w/2, bert_vals,  w, label='BERT',
                        color='#3498db', alpha=0.85)

            ax.axhline(0.5, color='gray', linestyle='--', lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(stocks_common, fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel(mlbl, fontsize=8)
            ax.set_title(f'{mname} — {mlbl}', fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.25)

            # annotate ค่า
            for bar in b1:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=6)
            for bar in b2:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    path = PLOT_DIR / 'comparison_bert_vs_tfidf.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  บันทึก: {path}')


def print_comparison_table(tfidf_results: dict, bert_results: dict):
    """ตารางสรุป diff ระหว่าง TF-IDF และ BERT"""
    print('\n' + '=' * 70)
    print('📊 ตารางเปรียบเทียบ: TF-IDF vs BERT Embedding (AUC-ROC)')
    print('=' * 70)
    print(f'{"":8s} {"":6s} {"TF-IDF":>9s} {"BERT":>9s} {"Diff":>9s} {"Winner":>8s}')
    print('-' * 55)

    for mkey, mname in [('rf', 'RF'), ('svm', 'SVM'), ('lstm', 'LSTM')]:
        tfidf_rows = tfidf_results.get(mkey, {})
        bert_rows  = bert_results.get(mkey, {})
        for stock in STOCKS:
            t_auc = tfidf_rows.get(stock, {}).get('auc_roc', None)
            b_auc = bert_rows.get(stock, {}).get('auc_roc', None)
            if t_auc is None:
                continue
            if b_auc is not None:
                diff   = t_auc - b_auc
                winner = 'TF-IDF ✅' if diff > 0.01 else 'BERT ✅' if diff < -0.01 else 'เสมอ'
                print(f'{stock:8s} {mname:6s} {t_auc:>9.4f} {b_auc:>9.4f} '
                      f'{diff:>+9.4f} {winner:>8s}')
            else:
                print(f'{stock:8s} {mname:6s} {t_auc:>9.4f} {"N/A":>9s}')


# ============================================================
# STEP 5: MAIN
# ============================================================
def run():
    np.random.seed(RANDOM_STATE)

    print('=' * 60)
    print('TF-IDF Models: RF + SVM + LSTM')
    print(f'Input: {DATA_FILE}')
    print('=' * 60)

    df, tfidf_cols = load_data()
    n_tfidf_dim = len(tfidf_cols)

    rf_results   = {}
    svm_results  = {}
    lstm_results = {}

    for stock in STOCKS:
        print(f'\n{"="*50}\nหุ้น: {stock}\n{"="*50}')
        data = prepare_stock(df, stock, tfidf_cols)
        if data is None:
            continue

        print(f'  train={data["n_train"]} val={data["n_val"]} '
              f'test={data["n_test"]} up%={data["y_train"].mean():.1%}')

        rf_results[stock]   = train_rf(data, stock)
        svm_results[stock]  = train_svm(data, stock)
        lstm_results[stock] = train_lstm(data, stock)

    # ─── สรุปตัวเลข ────────────────────────────────────────
    print('\n' + '=' * 60)
    print('สรุปผล TF-IDF Models')
    print('=' * 60)

    for results, name in [(rf_results, 'Random Forest'),
                          (svm_results, 'SVM'),
                          (lstm_results, 'LSTM')]:
        valid = {k: v for k, v in results.items() if v}
        if not valid:
            continue
        print(f'\n[{name}]')
        print(f'  {"หุ้น":8s} {"Acc":>8s} {"F1":>7s} {"AUC":>7s} {"Overfitting":>12s}')
        print('  ' + '-' * 45)
        for stock, r in valid.items():
            gap_flag = '⚠' if abs(r.get('overfit_gap', 0)) > 0.10 else '✓'
            print(f'  {stock:8s} {r["accuracy"]:>8.1%} {r["f1_score"]:>7.3f} '
                  f'{r["auc_roc"]:>7.3f}  gap={r.get("overfit_gap", 0):+.1%} {gap_flag}')
        avg_acc = np.mean([r['accuracy'] for r in valid.values()])
        avg_auc = np.mean([r['auc_roc']  for r in valid.values()])
        print(f'  avg → Acc={avg_acc:.1%}  AUC={avg_auc:.3f}')

    # ─── เปรียบเทียบกับ BERT ────────────────────────────────
    bert_results = load_bert_results()
    tfidf_grouped = {'rf': rf_results, 'svm': svm_results, 'lstm': lstm_results}
    plot_comparison(tfidf_grouped, bert_results)
    print_comparison_table(tfidf_grouped, bert_results)

    # ─── บันทึก JSON ────────────────────────────────────────
    def to_py(o):
        if isinstance(o, dict):                      return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list):                      return [to_py(i) for i in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o

    skip = {'y_prob', 'y_true', 'importances', 'all_cols'}
    all_results = []
    for results in [rf_results, svm_results, lstm_results]:
        for r in results.values():
            if r:
                all_results.append({k: v for k, v in r.items() if k not in skip})

    report = {
        'created_at': datetime.now().isoformat(),
        'tfidf_dim':  n_tfidf_dim,
        'results':    to_py(all_results),
    }
    report_path = MODEL_DIR / 'tfidf_evaluation_report.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n✅ บันทึกโมเดลทั้งหมดใน: {MODEL_DIR}/')
    print(f'✅ บันทึกกราฟทั้งหมดใน:  {PLOT_DIR}/')
    print(f'✅ รายงาน JSON:           {report_path}')
    return all_results


if __name__ == '__main__':
    run()
    print('\nTF-IDF Models เทรนเสร็จสมบูรณ์!')
