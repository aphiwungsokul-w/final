
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize


# Text Preprocessing
_STOPWORDS = frozenset(thai_stopwords()) - {
    'ขึ้น', 'ลง', 'บวก', 'ลบ', 'กำไร', 'ขาดทุน', 'สูง', 'ต่ำ', 'ซื้อ' , 'อัพ' , 'Up' , 'บาย' , 'Buy' ,
    'เพิ่ม', 'ลด', 'แข็ง', 'อ่อน', 'พุ่ง', 'ร่วง', 'ฟื้น', 'ซบ', 'ขาย', 'ดาวน์' , 'Down' , 'เซลล์' , 'Sell' , 'เท'
}

def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    # ตัด source tag ท้าย (เช่น " - Thunhoon")
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



# WangchanBERTa Embedding

def load_model(model_name: str, device: torch.device):
    print(f'โหลดโมเดล {model_name} บน {device}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print('โหลดสำเร็จ')
    return tokenizer, model


def embed_batch(texts: list, tokenizer, model, device, max_length=128) -> np.ndarray:
    safe = [t if t.strip() else 'ข่าวหุ้น' for t in texts]
    enc  = tokenizer(
        safe, return_tensors='pt', truncation=True,
        max_length=max_length, padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    # CLS token = index 0
    return out.last_hidden_state[:, 0, :].cpu().numpy()


def embed_all(texts: list, tokenizer, model, device,
              batch_size=32, max_length=128, cache_path=None) -> np.ndarray:
    
    # โหลด cache ถ้ามี
    cache = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f'โหลด cache: {len(cache):,} entries')

    # แยกที่ต้องคำนวณใหม่
    need_idx  = [i for i, t in enumerate(texts) if t not in cache]
    need_text = [texts[i] for i in need_idx]
    print(f'ต้องคำนวณ: {len(need_text):,} / {len(texts):,} ข้อความ')

    # Batch inference
    for i in tqdm(range(0, len(need_text), batch_size), desc='Embedding', ncols=70):
        batch = need_text[i:i + batch_size]
        vecs  = embed_batch(batch, tokenizer, model, device, max_length)
        for text, vec in zip(batch, vecs):
            cache[text] = vec

    # บันทึก cache
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f'บันทึก cache: {cache_path}')

    return np.array([cache[t] for t in texts])



# สร้าง Target Label จากราคา

def build_price_labels(price_file: str) -> pd.DataFrame:
    """
    อ่าน price_stock_3yr.csv สร้าง Target_Label
    Label = 1 ถ้า Open > Prev_Close (ราคาเปิดสูงกว่าปิดเมื่อวาน)
    """
    print(f'โหลดข้อมูลราคา {price_file}...')
    df = pd.read_csv(price_file, parse_dates=['Date'])
    
    # แก้ไขชื่อคอลัมน์ Ticker -> Stock และเอา .BK ออก
    if 'Ticker' in df.columns:
        df = df.rename(columns={'Ticker': 'Stock'})
        df['Stock'] = df['Stock'].str.replace('.BK', '', regex=False)
    
    df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)

    df['Prev_Close']   = df.groupby('Stock')['Close'].shift(1)
    df['Target_Label'] = (df['Open'] > df['Prev_Close']).astype(int)
    df = df.dropna(subset=['Prev_Close'])

    print(f'ราคา: {len(df):,} rows | label dist: {df["Target_Label"].value_counts().to_dict()}')
    return df[['Stock', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Prev_Close', 'Target_Label']]

# main

def run_pipeline(
    news_file='5year_news.csv',
    price_file='price_stock_5yr.csv',
    output_file='final_5year_dataset.csv',
    cache_file='embed_cache.pkl',       # cache embedding ไว้ใน disk

    max_length=128,
    batch_size=32,                      # ยิ่งมาก GPU ยิ่งเร็ว

    # T+1 rule: ข่าววัน T ส่งผลต่อ Open วัน T+1
    # ถ้าอยากให้ข่าววัน T ส่งผลต่อ Open วัน T เอง ให้เปลี่ยนเป็น False
    # (ระวัง leakage ถ้าข่าวออกหลัง market open)
    news_shift_days=1,
    
):
    print('=' * 60)
    print('Data Preparation Pipeline')
    print('=' * 60)

    # ── โหลดและ preprocess ข่าว ──────────────────────────────
    print(f'โหลดข่าว {news_file}...')
    df_news = pd.read_csv(news_file)
    df_news.drop_duplicates(subset=['Title', 'Stock'], keep='first', inplace=True)

    print('Preprocessing ข้อความ...')
    df_news['Processed_Title'] = df_news['Title'].apply(preprocess)
    df_news['News_Date']       = pd.to_datetime(df_news['Date']).dt.date

    # T+1 rule: ข่าววัน T → ส่งผลต่อ Open T+1
    df_news['Target_Date'] = pd.to_datetime(df_news['News_Date']) \
        + pd.Timedelta(days=news_shift_days)
    df_news['Target_Date'] = df_news['Target_Date'].dt.date

    print(f'ข่าวทั้งหมด: {len(df_news):,} | Stock dist: {df_news["Stock"].value_counts().to_dict()}')

    # ── Embedding ────────────────────────────────────────────
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanbart-large-finance")
    model = AutoModel.from_pretrained("airesearch/wangchanbart-large-finance")

    texts = df_news['Processed_Title'].tolist()
    vecs  = embed_all(
        texts, tokenizer, model, device,
        batch_size=batch_size,
        max_length=max_length,
        cache_path=cache_file,
    )
    df_news['Vector'] = list(vecs)

    # ── รวมข่าวรายวัน (mean pooling) ────────────────────────
    print('รวมข่าวรายวัน (mean pooling)...')
    news_daily = df_news.groupby(['Stock', 'Target_Date'])['Vector'].apply(
        lambda vs: np.mean(np.stack(vs.tolist()), axis=0)
    ).reset_index()

    # ── โหลดราคาและสร้าง label ──────────────────────────────
    df_price = build_price_labels(price_file)
    df_price['Date'] = df_price['Date'].dt.date

    # ── Merge ────────────────────────────────────────────────
    print('Merge ข่าว + ราคา...')
    news_daily['Target_Date'] = pd.to_datetime(news_daily['Target_Date']).dt.date
    merged = pd.merge(
        news_daily, df_price,
        left_on=['Stock', 'Target_Date'],
        right_on=['Stock', 'Date'],
        how='inner',
    )
    merged = merged.sort_values(['Stock', 'Target_Date']).reset_index(drop=True)

    print(f'\nผล Merge:')
    for stock, n in merged.groupby('Stock').size().items():
        print(f'  {stock:8s}: {n:4d} แถว')
    print(f'  รวม     : {len(merged):4d} แถว')
    print(f'  label dist: {merged["Target_Label"].value_counts().to_dict()}')

    # ── แยก Vector → คอลัมน์ย่อย ────────────────────────────
    vec_df = pd.DataFrame(
        merged['Vector'].tolist(), index=merged.index,
        columns=[f'vec_{i}' for i in range(768)],
    )
    out = pd.concat([
        merged[['Stock', 'Target_Date', 'Open', 'Prev_Close',
                 'High', 'Low', 'Close', 'Volume', 'Target_Label']],
        vec_df,
    ], axis=1)

    out.to_csv(output_file, index=False)
    print(f'\nบันทึกสำเร็จ: {output_file} ({len(out):,} rows)')
    return out


if __name__ == '__main__':
    df = run_pipeline()
    print('เสร็จสมบูรณ์!')
