# prepare_tfidf.py
# Pipeline: 5year_news.csv + price_stock_5yr.csv → tfidf_dataset.csv
#
# เหมือน prepare_data.py แต่ใช้ TF-IDF แทน WangchanBERTa
#
# ข้อดีเทียบกับ BERT embedding:
#   - เร็วกว่ามาก (วินาที vs 20+ นาที)
#   - ไม่ต้องการ GPU
#   - มิติน้อยกว่า (1,000–3,000 vs 768) → overfit น้อยกว่าเมื่อข้อมูลน้อย
#   - ตีความ feature ได้ (รู้ว่าคำไหนสำคัญ)
#
# T+1 rule เหมือนเดิม: ข่าววัน T → ส่งผลต่อ Open T+1
#
# Output:
#   tfidf_dataset.csv   ← (Stock, Target_Date, Target_Label, tfidf_0..N, price_cols)
#   tfidf_vectorizer.pkl ← TF-IDF model สำหรับ inference
#
# Install: pip install scikit-learn pythainlp pandas numpy

import re
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================
NEWS_FILE    = '5year_news.csv'
PRICE_FILE   = 'price_stock_5yr.csv'
OUTPUT_FILE  = 'tfidf_dataset.csv'
TFIDF_MODEL  = 'tfidf_vectorizer.pkl'

TFIDF_DIM    = 400       # จำนวน feature ที่เก็บ (แนะนำ 500–3000)
MIN_DF       = 2           # คำต้องปรากฏ >= 2 ข่าวจึงนับ (กำจัด typo)
MAX_DF       = 0.85        # คำที่ปรากฏใน > 85% ของข่าวถือว่า common เกินไป
NGRAM_RANGE  = (1, 2)      # unigram + bigram (เช่น "กำไร ลด" เป็น feature เดียว)
NEWS_SHIFT   = 1           # T+1 rule


# ============================================================
# STEP 1: Text Preprocessing (เหมือน prepare_data.py)
# ============================================================
_STOPWORDS = frozenset(thai_stopwords()) - {
    'ขึ้น', 'ลง', 'บวก', 'ลบ', 'กำไร', 'ขาดทุน', 'สูง', 'ต่ำ',
    'เพิ่ม', 'ลด', 'แข็ง', 'อ่อน', 'พุ่ง', 'ร่วง', 'ฟื้น', 'ซบ',
}

def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    for sep in [' - ', ' | ']:
        if sep in text:
            text = text.rsplit(sep, 1)[0]
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\sก-๙]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        text = normalize(text)
        tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
        return ' '.join(tokens)
    except Exception:
        return text


# ============================================================
# STEP 2: สร้าง Target Label จากราคา
# ============================================================
def build_price_labels(price_file: str) -> pd.DataFrame:
    log.info(f'โหลดราคา {price_file}...')
    df = pd.read_csv(price_file, parse_dates=['Date'])

    if 'Ticker' in df.columns:
        df = df.rename(columns={'Ticker': 'Stock'})
        df['Stock'] = df['Stock'].str.replace('.BK', '', regex=False)

    df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
    df['Prev_Close']   = df.groupby('Stock')['Close'].shift(1)
    df['Target_Label'] = (df['Open'] > df['Prev_Close']).astype(int)
    df = df.dropna(subset=['Prev_Close'])

    log.info(f'ราคา: {len(df):,} rows | label: {df["Target_Label"].value_counts().to_dict()}')
    return df[['Stock', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Prev_Close', 'Target_Label']]


# ============================================================
# STEP 3: Fit TF-IDF และแปลง
# ============================================================
def fit_tfidf(train_texts: list, all_texts: list):
    """
    Fit บน train set เท่านั้น (ป้องกัน data leakage)
    แล้ว transform ทุกข้อความ
    """
    log.info(f'Fit TF-IDF: max_features={TFIDF_DIM}, ngram={NGRAM_RANGE}, '
             f'min_df={MIN_DF}, max_df={MAX_DF}')

    vectorizer = TfidfVectorizer(
        max_features = TFIDF_DIM,
        ngram_range  = NGRAM_RANGE,
        min_df       = MIN_DF,
        max_df       = MAX_DF,
        sublinear_tf = True,    # ใช้ log(1+tf) แทน tf ดิบ → ลด effect คำที่ซ้ำมาก
        analyzer     = 'word',
    )

    vectorizer.fit(train_texts)
    log.info(f'Vocabulary size: {len(vectorizer.vocabulary_):,} คำ')

    # Top 20 feature ที่มี IDF สูงสุด (คำที่ distinctive ที่สุด)
    idf_scores  = vectorizer.idf_
    feature_names = vectorizer.get_feature_names_out()
    top_idx     = np.argsort(idf_scores)[::-1][:20]
    log.info('Top 20 distinctive terms: ' + ', '.join(feature_names[top_idx]))

    # Transform ทั้งหมด
    X_sparse = vectorizer.transform(all_texts)
    return vectorizer, X_sparse


# ============================================================
# STEP 4: MAIN PIPELINE
# ============================================================
def run_pipeline():
    log.info('=' * 60)
    log.info('TF-IDF Data Preparation Pipeline')
    log.info('=' * 60)

    # ── โหลดข่าว ─────────────────────────────────────────────
    log.info(f'โหลดข่าว {NEWS_FILE}...')
    df_news = pd.read_csv(NEWS_FILE)
    df_news.drop_duplicates(subset=['Title', 'Stock'], keep='first', inplace=True)

    log.info('Preprocessing...')
    df_news['Processed'] = df_news['Title'].apply(preprocess)
    df_news['News_Date'] = pd.to_datetime(df_news['Date']).dt.date
    df_news['Target_Date'] = (
        pd.to_datetime(df_news['News_Date']) + pd.Timedelta(days=NEWS_SHIFT)
    ).dt.date

    log.info(f'ข่าวทั้งหมด: {len(df_news):,} | Stock: {df_news["Stock"].value_counts().to_dict()}')

    # ── โหลดราคา + merge เพื่อรู้ train boundary ────────────
    df_price = build_price_labels(PRICE_FILE)
    df_price['Date'] = df_price['Date'].dt.date
    df_news['Target_Date_dt'] = pd.to_datetime(df_news['Target_Date'])

    # ── กำหนด train boundary (70% แรกของทุกหุ้น รวมกัน) ──
    # ใช้ global date boundary เพื่อ fit TF-IDF บน train only
    all_dates = sorted(df_news['Target_Date'].unique())
    n_train_dates = int(len(all_dates) * 0.70)
    train_cutoff  = all_dates[n_train_dates]

    train_mask = df_news['Target_Date'] <= train_cutoff
    train_texts = df_news.loc[train_mask, 'Processed'].tolist()
    all_texts   = df_news['Processed'].tolist()

    log.info(f'Train cutoff: {train_cutoff} | train={train_mask.sum():,} / all={len(df_news):,}')

    # ── Fit & Transform TF-IDF ────────────────────────────────
    vectorizer, X_sparse = fit_tfidf(train_texts, all_texts)

    # บันทึก vectorizer สำหรับ inference
    with open(TFIDF_MODEL, 'wb') as f:
        pickle.dump(vectorizer, f)
    log.info(f'บันทึก vectorizer: {TFIDF_MODEL}')

    df_news['tfidf_vec'] = list(X_sparse.toarray())

    # ── รวมข่าวรายวัน (mean pooling ของ TF-IDF) ─────────────
    log.info('รวมข่าวรายวัน (mean pooling)...')
    news_daily = df_news.groupby(['Stock', 'Target_Date'])['tfidf_vec'].apply(
        lambda vs: np.mean(np.stack(vs.tolist()), axis=0)
    ).reset_index()

    # ── Merge กับราคา ────────────────────────────────────────
    log.info('Merge ข่าว + ราคา...')
    news_daily['Target_Date'] = pd.to_datetime(news_daily['Target_Date']).dt.date
    merged = pd.merge(
        news_daily, df_price,
        left_on=['Stock', 'Target_Date'],
        right_on=['Stock', 'Date'],
        how='inner',
    ).sort_values(['Stock', 'Target_Date']).reset_index(drop=True)

    log.info(f'\nผล Merge:')
    for stock, n in merged.groupby('Stock').size().items():
        log.info(f'  {stock:8s}: {n:4d} แถว')
    log.info(f'  รวม: {len(merged):4d} แถว | label: {merged["Target_Label"].value_counts().to_dict()}')

    # ── แยก vector → คอลัมน์ ─────────────────────────────────
    vec_df = pd.DataFrame(
        merged['tfidf_vec'].tolist(),
        index=merged.index,
        columns=[f'tfidf_{i}' for i in range(TFIDF_DIM)],
    )
    out = pd.concat([
        merged[['Stock', 'Target_Date', 'Open', 'Prev_Close',
                'High', 'Low', 'Close', 'Volume', 'Target_Label']],
        vec_df,
    ], axis=1)

    out.to_csv(OUTPUT_FILE, index=False)
    log.info(f'\n✅ บันทึกสำเร็จ: {OUTPUT_FILE} ({len(out):,} rows, {TFIDF_DIM + 8} cols)')

    # ── แสดง Top terms ต่อ label ─────────────────────────────
    _show_top_terms_by_label(df_news, df_price, vectorizer)

    return out


def _show_top_terms_by_label(df_news, df_price, vectorizer):
    """แสดง Top TF-IDF terms ของข่าวที่ label=0 vs label=1"""
    df_price['Date'] = pd.to_datetime(df_price['Date']) if df_price['Date'].dtype != 'object' else df_price['Date']

    merged_news = pd.merge(
        df_news[['Stock', 'Target_Date', 'Processed']],
        df_price[['Stock', 'Date', 'Target_Label']],
        left_on=['Stock', 'Target_Date'],
        right_on=['Stock', 'Date'],
        how='inner',
    )

    if merged_news.empty:
        return

    log.info('\nTop 10 terms ที่เกี่ยวข้องกับ label (global):')
    feature_names = vectorizer.get_feature_names_out()

    for label, name in [(0, 'Down/Flat'), (1, 'Up')]:
        texts = merged_news.loc[merged_news['Target_Label'] == label, 'Processed'].tolist()
        if not texts:
            continue
        X = vectorizer.transform(texts)
        mean_tfidf = X.mean(axis=0).A1
        top_idx = np.argsort(mean_tfidf)[::-1][:10]
        top_terms = [f"{feature_names[i]}({mean_tfidf[i]:.3f})" for i in top_idx]
        log.info(f'  {name:10s}: {", ".join(top_terms)}')


if __name__ == '__main__':
    df = run_pipeline()
    log.info('Pipeline TF-IDF เสร็จสมบูรณ์!')
