# main.py — FastAPI web server + ML prediction integration
# รัน: uvicorn main:app --host 0.0.0.0 --port 8000
#
# Endpoints:
#   GET  /                     ← Dashboard
#   GET  /stock/{symbol}       ← Stock detail + chart
#   GET  /news                 ← รวมข่าวทุกหุ้น
#   POST /api/refresh          ← Trigger daily prediction (เรียกจาก cron-job.org)
#   GET  /api/predictions      ← ดู predictions.json ปัจจุบัน (JSON)

import json
import logging
import threading
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import feedparser
import yfinance as yf
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

app         = FastAPI(title='StockNews AI')
templates   = Jinja2Templates(directory='templates')
PREDICTIONS = Path('predictions.json')

THAI_STOCKS = {
    'DELTA':  ('DELTA.BK',  'Delta Electronics (Thailand)'),
    'ADVANC': ('ADVANC.BK', 'Advanced Info Service'),
    'PTT':    ('PTT.BK',    'PTT Public Company Limited'),
    'AOT':    ('AOT.BK',    'Airports of Thailand'),
    'SCB':    ('SCB.BK',    'SCB X Public Company Limited'),
}


# ============================================================
# helpers
# ============================================================
def load_predictions() -> dict:
    """โหลด predictions.json คืน {} ถ้าไม่มีไฟล์"""
    if PREDICTIONS.exists():
        try:
            return json.loads(PREDICTIONS.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def fetch_price(symbol: str) -> dict:
    """ดึงราคาล่าสุดจาก yfinance"""
    yf_ticker = THAI_STOCKS.get(symbol, (None,))[0]
    if not yf_ticker:
        return {'price': '-', 'change': '-', 'is_positive': True, 'history': []}
    try:
        ticker = yf.Ticker(yf_ticker)
        hist   = ticker.history(period='30d')
        if len(hist) < 2:
            raise ValueError('not enough data')
        prev    = hist['Close'].iloc[-2]
        current = hist['Close'].iloc[-1]
        change  = current - prev
        pct     = (change / prev) * 100 if prev else 0
        sign    = '+' if change >= 0 else ''
        # ส่งราคา 30 วันย้อนหลังสำหรับกราฟ
        price_history = [
            {'date': str(d.date()), 'close': round(float(v), 2)}
            for d, v in zip(hist.index, hist['Close'])
        ]
        return {
            'price':       f'{current:.2f}',
            'change':      f'{sign}{change:.2f} ({sign}{pct:.2f}%)',
            'is_positive': change >= 0,
            'history':     price_history,
        }
    except Exception:
        return {'price': '-', 'change': '-', 'is_positive': True, 'history': []}


def fetch_google_news(symbol: str, max_items: int = 5) -> list:
    query = urllib.parse.quote(f'หุ้น {symbol}')
    url   = f'https://news.google.com/rss/search?q={query}&hl=th&gl=TH&ceid=TH:th'
    feed  = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        parts    = e.title.rsplit(' - ', 1)
        headline = parts[0]
        source   = parts[1] if len(parts) > 1 else 'Google News'
        try:
            dt   = datetime.strptime(e.published, '%a, %d %b %Y %H:%M:%S %Z')
            date = dt.strftime('%Y-%m-%d')
            ts   = dt.timestamp()
        except Exception:
            date = datetime.now().strftime('%Y-%m-%d')
            ts   = time.time()
        items.append({
            'headline': headline, 'source': source,
            'date': date, 'link': e.link, 'timestamp': ts,
        })
    return items


# ============================================================
# Background: รัน daily_predict เมื่อ startup ครั้งแรก
# ============================================================
def _run_predict_bg():
    """รันใน thread แยก ไม่บล็อก startup"""
    try:
        from daily_predict import run_daily_predict
        log.info('[startup] เริ่ม daily_predict...')
        run_daily_predict()
        log.info('[startup] daily_predict เสร็จสิ้น')
    except Exception as e:
        log.error(f'[startup] daily_predict error: {e}')


@app.on_event('startup')
async def startup_event():
    # รันเฉพาะเมื่อยังไม่มีข้อมูลวันนี้
    preds = load_predictions()
    today = datetime.now().strftime('%Y-%m-%d')
    has_today = any(
        v.get('date', '') == today
        for v in preds.values()
    )
    if not has_today:
        t = threading.Thread(target=_run_predict_bg, daemon=True)
        t.start()
    else:
        log.info('[startup] ข้อมูลวันนี้มีอยู่แล้ว ข้าม daily_predict')


# ============================================================
# Routes
# ============================================================
@app.get('/', response_class=HTMLResponse)
async def dashboard(request: Request):
    preds  = load_predictions()
    stocks = []

    for symbol, (_, name) in THAI_STOCKS.items():
        price_info = fetch_price(symbol)
        pred_info  = preds.get(symbol, {})
        rec        = pred_info.get('prediction', {}).get('recommended', {})
        rec_result = rec.get('result', {})

        # signal จาก ML (ถ้ามี) หรือใช้ price change แทน
        if rec_result:
            is_up      = rec_result.get('signal', 1) == 1
            confidence = rec_result.get('confidence', 0)
            sentiment  = f"{'ขึ้น' if is_up else 'ลง'} ({confidence:.0%})"
        else:
            is_up     = price_info['is_positive']
            sentiment = 'รอผลจากโมเดล' if is_up else 'รอผลจากโมเดล'

        stocks.append({
            'symbol':      symbol,
            'name':        name,
            'price':       price_info['price'],
            'change':      price_info['change'],
            'is_positive': is_up,
            'sentiment':   sentiment,
            'updated_at':  pred_info.get('updated_at', '-'),
            'history_json': json.dumps([h['close'] for h in price_info.get('history', [])]),
        })

    return templates.TemplateResponse(
        'index.html', {'request': request, 'stocks': stocks}
    )


@app.get('/stock/{symbol}', response_class=HTMLResponse)
async def stock_detail(request: Request, symbol: str):
    symbol = symbol.upper()
    if symbol not in THAI_STOCKS:
        return HTMLResponse('ไม่พบข้อมูลหุ้นนี้', status_code=404)

    name       = THAI_STOCKS[symbol][1]
    price_info = fetch_price(symbol)
    preds      = load_predictions()
    pred_info  = preds.get(symbol, {})
    prediction = pred_info.get('prediction', {})

    # ข่าวล่าสุดจาก predictions หรือดึงใหม่
    news_items = pred_info.get('news') or fetch_google_news(symbol, 5)

    # เตรียม chart data (ราคาจริง 30 วัน)
    price_history = price_info.get('history', [])

    # เตรียม ML result สำหรับ template
    lstm_res = prediction.get('lstm') or {}
    rf_res  = prediction.get('rf')  or {}
    rec      = prediction.get('recommended') or {}

    return templates.TemplateResponse('stock.html', {
        'request':      request,
        'symbol':       symbol,
        'name':         name,
        'stock_info':   price_info,
        'news_items':   news_items,
        'price_history_json': json.dumps(price_history),
        'lstm': {
            'signal':     lstm_res.get('signal', None),
            'label':      lstm_res.get('label', '-'),
            'prob':       f"{lstm_res.get('probability_up', 0):.1%}",
            'confidence': f"{lstm_res.get('confidence', 0):.1%}",
        },
        'rf': {
            'signal':     rf_res.get('signal', None),
            'label':      rf_res.get('label', '-'),
            'prob':       f"{rf_res.get('probability_up', 0):.1%}",
            'confidence': f"{rf_res.get('confidence', 0):.1%}",
        },
        'recommended_model': rec.get('model', '-').upper(),
        'recommended_label': rec.get('result', {}).get('label', '-'),
        'recommended_conf':  f"{rec.get('result', {}).get('confidence', 0):.1%}",
        'updated_at': pred_info.get('updated_at', 'ยังไม่มีข้อมูล'),
    })


@app.get('/news', response_class=HTMLResponse)
async def all_news(request: Request):
    preds    = load_predictions()
    all_news = []
    for symbol in THAI_STOCKS:
        news = preds.get(symbol, {}).get('news') or fetch_google_news(symbol, 3)
        for n in news:
            n['symbol'] = symbol
        all_news.extend(news)
    all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    return templates.TemplateResponse(
        'news.html', {'request': request, 'news_items': all_news}
    )


# ============================================================
# API: Refresh endpoint (trigger จาก cron-job.org)
# ============================================================
@app.post('/api/refresh')
async def api_refresh(background_tasks: BackgroundTasks):
    """
    Trigger daily prediction ใหม่
    เรียกจาก cron-job.org ทุกวันเช้า 08:30
    หรือ manual: curl -X POST https://your-app.onrender.com/api/refresh
    """
    def _run():
        try:
            from daily_predict import run_daily_predict
            run_daily_predict()
        except Exception as e:
            log.error(f'/api/refresh error: {e}')

    background_tasks.add_task(_run)
    return {'status': 'started', 'message': 'กำลังรัน daily predict ใน background'}


@app.get('/api/predictions')
async def api_predictions():
    """ดู predictions.json ปัจจุบัน"""
    return JSONResponse(load_predictions())


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=False)
