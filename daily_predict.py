# daily_predict.py
# รันทุกวันเช้า 08:30 (ก่อนตลาดเปิด 09:30)
# ดึงข่าวจาก Google News RSS → embed → predict → บันทึกลง predictions.json
#
# Render: ถูก trigger จาก cron-job.org (ฟรี) POST /api/refresh
# Local:  python daily_predict.py

import json
import logging
import re
import urllib.parse
from datetime import date, datetime
from pathlib import Path

import feedparser
import joblib
import numpy as np
import torch
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

MODEL_DIR      = Path('models')
PREDICTIONS_FILE = Path('predictions.json')
EMBED_MODEL_NAME = 'airesearch/wangchanberta-base-att-spm-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STOCKS = {
    'PTT':    ('PTT.BK',    'PTT Public Company Limited'),
    'AOT':    ('AOT.BK',    'Airports of Thailand'),
    'DELTA':  ('DELTA.BK',  'Delta Electronics (Thailand)'),
    'ADVANC': ('ADVANC.BK', 'Advanced Info Service'),
    'SCB':    ('SCB.BK',    'SCB X Public Company Limited'),
}

# ── Singleton: โหลดครั้งเดียว ──────────────────────────────
_tokenizer = None
_bert      = None
_ml_models = {}   # {stock: {lstm: model, gru: model}}
_scalers   = {}   # {stock: scaler}
_meta      = None


def _load_assets():
    global _tokenizer, _bert, _ml_models, _scalers, _meta

    if _tokenizer is not None:
        return

    meta_path = MODEL_DIR / 'model_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f'ไม่พบ {meta_path} — กรุณารัน train_compare.py ก่อน'
        )
    with open(meta_path, encoding='utf-8') as f:
        _meta = json.load(f)

    log.info(f'โหลด WangchanBERTa บน {DEVICE}...')
    _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    _bert      = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)
    _bert.eval()

    import tensorflow as tf
    for stock in _meta['stocks']:
        _ml_models[stock] = {}
        for mtype in ['lstm', 'gru']:
            p = MODEL_DIR / f'{mtype}_{stock}.keras'
            if p.exists():
                _ml_models[stock][mtype] = tf.keras.models.load_model(p)
        sp = MODEL_DIR / f'scaler_{stock}.pkl'
        if sp.exists():
            _scalers[stock] = joblib.load(sp)

    log.info('โหลดสำเร็จ')


# ── Text Preprocessing ──────────────────────────────────────
_SW = frozenset(thai_stopwords()) - {
    'ขึ้น', 'ลง', 'บวก', 'ลบ', 'กำไร', 'ขาดทุน', 'สูง', 'ต่ำ',
    'เพิ่ม', 'ลด', 'แข็ง', 'อ่อน', 'พุ่ง', 'ร่วง', 'ฟื้น',
}

def _clean(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return 'ข่าวหุ้น'
    for sep in [' - ', ' | ']:
        if sep in text:
            text = text.rsplit(sep, 1)[0]
    text = text.lower()
    text = re.sub(r'<[^>]+>|https?://\S+', '', text)
    text = re.sub(r'[^\w\sก-๙]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        text = normalize(text)
        tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
        tokens = [t for t in tokens if t not in _SW and len(t) > 1]
        return ' '.join(tokens) or 'ข่าวหุ้น'
    except Exception:
        return text or 'ข่าวหุ้น'


# ── Embed list of texts → mean vector (768,) ────────────────
def _embed(texts: list) -> np.ndarray:
    cleaned = [_clean(t) for t in texts]
    enc = _tokenizer(
        cleaned, return_tensors='pt',
        truncation=True, max_length=128, padding=True,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = _bert(**enc)
    vecs = out.last_hidden_state[:, 0, :].cpu().numpy()  # (N, 768)
    return vecs.mean(axis=0)                               # (768,)


# ── ดึงข่าวจาก Google News RSS ────────────────────────────
def fetch_news(symbol: str, max_items: int = 10) -> list:
    """คืน list ของ dict {headline, source, date, link}"""
    query = urllib.parse.quote(f'หุ้น {symbol}')
    url   = f'https://news.google.com/rss/search?q={query}&hl=th&gl=TH&ceid=TH:th'
    feed  = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        parts    = e.title.rsplit(' - ', 1)
        headline = parts[0]
        source   = parts[1] if len(parts) > 1 else 'Google News'
        try:
            dt = datetime.strptime(e.published, '%a, %d %b %Y %H:%M:%S %Z')
            display_date = dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            display_date = datetime.now().strftime('%Y-%m-%d')
        items.append({
            'headline': headline,
            'source':   source,
            'date':     display_date,
            'link':     e.link,
        })
    return items


# ── ทำนาย 1 หุ้น จาก list ของ news headlines ────────────
def predict_stock(symbol: str, news_list: list) -> dict:
    """คืน dict พร้อม lstm, gru, recommended"""
    if not news_list:
        return {'error': 'ไม่มีข่าว'}

    vec = _embed([n['headline'] for n in news_list])   # (768,)

    if symbol in _scalers:
        vec = _scalers[symbol].transform(vec.reshape(1, -1))[0]

    X   = vec.reshape(1, 1, 768).astype(np.float32)
    thr = _meta.get('threshold', 0.5)

    results = {}
    for mtype, model in _ml_models.get(symbol, {}).items():
        prob  = float(model.predict(X, verbose=0)[0][0])
        label = 'Up' if prob >= thr else 'Down'
        conf  = prob if prob >= thr else 1 - prob
        results[mtype] = {
            'probability_up': round(prob, 4),
            'confidence':     round(conf, 4),
            'signal':         1 if prob >= thr else 0,
            'label':          label,
        }

    # เลือก recommended จาก AUC ใน evaluation_report
    best = _pick_best(symbol, results)

    return {
        'stock':       symbol,
        'news_count':  len(news_list),
        'lstm':        results.get('lstm'),
        'gru':         results.get('gru'),
        'recommended': best,
    }


def _pick_best(symbol: str, results: dict) -> dict:
    try:
        with open(MODEL_DIR / 'evaluation_report.json', encoding='utf-8') as f:
            reports = json.load(f)
        auc = {r['model_type']: r['auc_roc']
               for r in reports if r['stock'] == symbol}
        best_model = max(auc, key=auc.get) if auc else 'lstm'
    except Exception:
        best_model = 'lstm'
    return {'model': best_model, 'result': results.get(best_model, {})}


# ── Main: รันทุกหุ้น บันทึกผล ────────────────────────────
def run_daily_predict() -> dict:
    _load_assets()

    output  = {}
    today   = date.today().isoformat()

    for symbol in STOCKS:
        log.info(f'[{symbol}] ดึงข่าว...')
        news = fetch_news(symbol, max_items=10)
        log.info(f'[{symbol}] ได้ {len(news)} ข่าว → predict...')
        pred = predict_stock(symbol, news) if news else {'error': 'no news'}

        output[symbol] = {
            'updated_at': datetime.now().isoformat(),
            'date':       today,
            'news':       news,
            'prediction': pred,
        }
        log.info(
            f'[{symbol}] done — '
            f'recommended: {pred.get("recommended", {}).get("result", {}).get("label", "?")}'
        )

    # บันทึกลง predictions.json
    PREDICTIONS_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    log.info(f'บันทึกสำเร็จ: {PREDICTIONS_FILE}')
    return output


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    run_daily_predict()
