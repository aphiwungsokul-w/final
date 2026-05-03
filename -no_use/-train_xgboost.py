# train_xgboost.py
# ทำนายทิศทาง Open Price จากข่าว ด้วย XGBoost
#
# Input  : final_5year_dataset.csv  (จาก prepare_data.py)
# Output : models/xgb_{stock}.pkl
#          models/scaler_{stock}_xgb.pkl
#          models/xgb_evaluation_report.json
#
# Pipeline: StandardScaler → XGBClassifier (ไม่ใช้ PCA)
#           XGBoost มี built-in regularization ครบ (colsample, gamma, reg_alpha/lambda)
#           จึงไม่จำเป็นต้องลด dimension ก่อน
#           ลด colsample_bytree เหลือ 0.05 เพื่อให้แต่ละต้นไม้เห็นแค่ ~40 จาก 768 มิติ
#
# Install: pip install xgboost scikit-learn pandas numpy joblib

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'data_file':   'final_5year_dataset.csv',
    'model_dir':   'models',
    'stocks':      ['PTT', 'AOT', 'DELTA', 'ADVANC', 'SCB'],
    'min_samples': 80,

    # Split (time-ordered — ห้าม shuffle)
    'train_ratio': 0.70,
    'val_ratio':   0.15,
    # test = 0.15

    # XGBoost hyperparameters
    'n_estimators':   500,    # จำนวน boosting rounds สูงสุด
    'learning_rate':  0.05,   # ยิ่งต่ำยิ่ง generalize ดี แต่ต้องการ rounds มากขึ้น
    'max_depth':      3,      # ต้นไม้ตื้น = regularization สูง (แนะนำ 3-5 สำหรับข้อมูลน้อย)
    'min_child_weight': 5,    # ป้องกัน overfit: node ต้องมีตัวอย่างอย่างน้อย 5
    'subsample':      0.8,    # สุ่ม 80% ของ rows ต่อ round (row subsampling)
    'colsample_bytree': 0.05, # สุ่ม ~5% ของ feature ต่อต้นไม้ (~40 จาก 768 มิติ)
                               # ลดจาก 0.8 เพราะไม่มี PCA ลด dim ให้แล้ว
    'gamma':          1.0,    # minimum loss reduction สำหรับ split (regularization)
    'reg_alpha':      0.1,    # L1 regularization บน weights
    'reg_lambda':     1.0,    # L2 regularization บน weights
    'early_stopping_rounds': 20,   # หยุดถ้า val_logloss ไม่ดีขึ้น 20 rounds
    'eval_metric':   'logloss',
    'random_state':   42,
    'threshold':      0.50,
}

MODEL_DIR = Path(CONFIG['model_dir'])
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: โหลดข้อมูล
# ============================================================
def load_data(config):
    log.info(f'โหลด {config["data_file"]}...')
    df = pd.read_csv(config['data_file'], parse_dates=['Target_Date'])
    df = df.sort_values(['Stock', 'Target_Date']).reset_index(drop=True)

    vec_cols = sorted(
        [c for c in df.columns if c.startswith('vec_')],
        key=lambda x: int(x.split('_')[1])
    )
    log.info(f'rows={len(df):,} | stocks={df["Stock"].nunique()} | vec_dims={len(vec_cols)}')

    log.info('\nจำนวนแถวต่อหุ้น:')
    for s, n in df.groupby('Stock').size().items():
        flag = '✅' if n >= config['min_samples'] else '⚠ น้อย'
        log.info(f'  {s:8s}: {n:4d}  {flag}')

    return df, vec_cols


# ============================================================
# STEP 2: เตรียม X, y + Scale แยกรายหุ้น
# ============================================================
def prepare_stock(df, stock, vec_cols, config):
    g = df[df['Stock'] == stock].sort_values('Target_Date').reset_index(drop=True)
    n = len(g)

    if n < config['min_samples']:
        log.warning(f'{stock}: {n} แถว < {config["min_samples"]} → ข้าม')
        return None

    X_raw = g[vec_cols].values.astype(np.float32)
    y     = g['Target_Label'].values.astype(int)
    X_raw = np.nan_to_num(X_raw, nan=0., posinf=0., neginf=0.)

    n_train = int(n * config['train_ratio'])
    n_val   = int(n * config['val_ratio'])
    n_test  = n - n_train - n_val

    # ── StandardScaler (fit บน train เท่านั้น) ──────────────
    # XGBoost ไม่บังคับ scaling แต่ช่วยให้ learning_rate สม่ำเสมอกว่า
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[:n_train])
    X_val   = scaler.transform(X_raw[n_train:n_train + n_val])
    X_test  = scaler.transform(X_raw[n_train + n_val:])

    joblib.dump(scaler, MODEL_DIR / f'scaler_{stock}_xgb.pkl')

    log.info(f'  vec dims = {X_train.shape[1]} (scaled, no PCA)')

    return {
        'X_train': X_train, 'y_train': y[:n_train],
        'X_val':   X_val,   'y_val':   y[n_train:n_train + n_val],
        'X_test':  X_test,  'y_test':  y[n_train + n_val:],
        'dates_test': g['Target_Date'].values[n_train + n_val:],
        'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
    }


# ============================================================
# STEP 3: คำนวณ scale_pos_weight สำหรับ class imbalance
# ============================================================
def get_scale_pos_weight(y_train):
    """XGBoost ใช้ scale_pos_weight แทน class_weight"""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if n_pos == 0:
        return 1.0
    return round(float(n_neg / n_pos), 4)


# ============================================================
# STEP 4: เทรน XGBoost
# ============================================================
def train_xgb(data, config):
    spw = get_scale_pos_weight(data['y_train'])
    log.info(f'  scale_pos_weight = {spw:.3f}')

    xgb = XGBClassifier(
        n_estimators          = config['n_estimators'],
        learning_rate         = config['learning_rate'],
        max_depth             = config['max_depth'],
        min_child_weight      = config['min_child_weight'],
        subsample             = config['subsample'],
        colsample_bytree      = config['colsample_bytree'],
        gamma                 = config['gamma'],
        reg_alpha             = config['reg_alpha'],
        reg_lambda            = config['reg_lambda'],
        scale_pos_weight      = spw,
        early_stopping_rounds = config['early_stopping_rounds'],
        eval_metric           = config['eval_metric'],
        random_state          = config['random_state'],
        use_label_encoder     = False,
        verbosity             = 0,
        n_jobs                = -1,
    )

    xgb.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False,
    )

    best_round  = xgb.best_iteration + 1
    train_pred  = xgb.predict(data['X_train'])
    train_acc   = accuracy_score(data['y_train'], train_pred)

    log.info(f'  best round = {best_round} / {config["n_estimators"]}')

    return xgb, round(float(train_acc), 4), best_round


# ============================================================
# STEP 5: ประเมินผล
# ============================================================
def evaluate(model, data, stock, vec_cols, config):
    thr    = config['threshold']
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    y_pred = (y_prob >= thr).astype(int)
    y_true = data['y_test']

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    cm  = confusion_matrix(y_true, y_pred).tolist()

    tp = cm[1][1] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    log.info(
        f'  [XGB] Acc={acc:.1%} F1={f1:.3f} AUC={auc:.3f} '
        f'Prec={prec:.1%} Recall={recall:.1%}'
    )

    # Feature importances ตาม gain (top 10 vec_i โดยตรง)
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10].tolist()
    top_named   = [vec_cols[i] for i in top_idx]

    return {
        'stock': stock, 'model_type': 'xgboost',
        'n_test': len(y_true),
        'n_features': len(importances),
        'accuracy':     round(float(acc),  4),
        'f1_score':     round(float(f1),   4),
        'auc_roc':      round(float(auc),  4),
        'precision_up': round(prec,        4),
        'recall_up':    round(recall,      4),
        'confusion_matrix': cm,
        'top10_important_features': top_named,
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=['Down/Flat', 'Up'],
            zero_division=0,
        ),
    }


# ============================================================
# STEP 6: MAIN
# ============================================================
def run(config=CONFIG):
    np.random.seed(config['random_state'])

    log.info('=' * 60)
    log.info('XGBoost — News Embedding Open Price Prediction')
    log.info('=' * 60)

    df, vec_cols = load_data(config)
    all_results  = []

    for stock in config['stocks']:
        log.info(f'\n{"="*50}\nหุ้น: {stock}\n{"="*50}')

        data = prepare_stock(df, stock, vec_cols, config)
        if data is None:
            continue

        log.info(
            f'  train={data["n_train"]} val={data["n_val"]} '
            f'test={data["n_test"]} '
            f'up%={data["y_train"].mean():.1%}'
        )

        xgb, train_acc, best_round = train_xgb(data, config)

        model_path = MODEL_DIR / f'xgb_{stock}.pkl'
        joblib.dump(xgb, model_path)

        result = evaluate(xgb, data, stock, vec_cols, config)
        result['train_acc']    = train_acc
        result['best_round']   = best_round
        result['overfit_gap']  = round(train_acc - result['accuracy'], 4)
        all_results.append(result)

    # ── สรุปผล ───────────────────────────────────────────────
    log.info('\n' + '=' * 60)
    log.info('สรุปผล XGBoost')
    log.info('=' * 60)
    log.info(f'{"หุ้น":8s} {"Acc":>8s} {"F1":>7s} {"AUC":>7s} {"N":>5s}')
    log.info('-' * 45)
    for r in all_results:
        log.info(
            f'{r["stock"]:8s} {r["accuracy"]:>8.1%} {r["f1_score"]:>7.3f} '
            f'{r["auc_roc"]:>7.3f} {r["n_test"]:>5d}'
        )

    if all_results:
        avg_acc = np.mean([r['accuracy'] for r in all_results])
        avg_auc = np.mean([r['auc_roc']  for r in all_results])
        log.info(f'\n  avg Acc: {avg_acc:.1%}  |  avg AUC: {avg_auc:.3f}')

    log.info('\nOverfitting check (train_acc vs test_acc):')
    for r in all_results:
        flag = '⚠ overfit' if r['overfit_gap'] > 0.10 else '✓'
        log.info(
            f'  {r["stock"]:8s} train={r["train_acc"]:.1%} '
            f'test={r["accuracy"]:.1%} gap={r["overfit_gap"]:+.1%} {flag}'
        )

    # ── บันทึก JSON ──────────────────────────────────────────
    def to_py(o):
        if isinstance(o, dict):  return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list):  return [to_py(i) for i in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o

    report_path = MODEL_DIR / 'xgb_evaluation_report.json'
    report_path.write_text(
        json.dumps(to_py(all_results), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    log.info(f'\nบันทึกผลที่: {report_path}')
    return all_results


if __name__ == '__main__':
    run(CONFIG)
    log.info('XGBoost เทรนเสร็จสมบูรณ์!')
