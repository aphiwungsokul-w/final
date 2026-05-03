import pandas as pd
import numpy as np
import yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModel
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
import re

# 1. ฟังก์ชัน Text Preprocessing (จากข้อมูลของคุณ)
def preprocess_stock_news(text):
    if not isinstance(text, str): return ""
    if ' - ' in text: text = text.rsplit(' - ', 1)[0] 
    if ' | ' in text: text = text.rsplit(' | ', 1)[0] 
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\sก-๙]', '', text)
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'\s+', ' ', text).strip()
    text = normalize(text)
    tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    standard_stopwords = frozenset(thai_stopwords())
    important_financial_words = {'ขึ้น', 'ลง', 'บวก', 'ลบ', 'กำไร', 'ขาดทุน', 'สูง', 'ต่ำ'}
    custom_stopwords = standard_stopwords - important_financial_words
    cleaned_tokens = [word for word in tokens if word not in custom_stopwords and len(word) > 1]
    return ' '.join(cleaned_tokens)

# 2. ตั้งค่า WangchanBERTa
print("กำลังโหลดโมเดล WangchanBERTa...")
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_text_embedding(text):
    if not isinstance(text, str) or text.strip() == "": return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()

# 3. โหลดและจัดการไฟล์ 3year_news.csv
print("กำลังโหลดและประมวลผลข้อมูลข่าว 3 ปี (กระบวนการนี้จะใช้เวลานาน กรุณารอ)...")
df_news = pd.read_csv('3year_news.csv')

# ลบข่าวที่ซ้ำกัน
df_news.drop_duplicates(subset=['Title', 'Stock'], keep='first', inplace=True)

# Clean Text และ Embedding
df_news['Processed_Title'] = df_news['Title'].apply(preprocess_stock_news)
df_news['Target_Date'] = pd.to_datetime(df_news['Date']).dt.date
df_news['Vector'] = df_news['Processed_Title'].apply(get_text_embedding)

# หาค่าเฉลี่ยของข่าวในวันเดียวกัน
news_grouped = df_news.groupby(['Stock', 'Target_Date'])['Vector'].apply(
    lambda vectors: np.mean(vectors.tolist(), axis=0)
).reset_index()

# 4. ดึงราคาหุ้นย้อนหลัง 3 ปี (3y)
print("กำลังดึงข้อมูลราคาหุ้นย้อนหลัง 3 ปี และจับคู่ Label...")
tickers_map = {'PTT': 'PTT.BK', 'AOT': 'AOT.BK', 'DELTA': 'DELTA.BK', 'ADVANC': 'ADVANC.BK', 'SCB': 'SCB.BK'}
all_stock_data = []

for stock_name, ticker in tickers_map.items():
    stock_df = yf.Ticker(ticker).history(period="3y", interval="1d")
    if not stock_df.empty:
        stock_df.index = stock_df.index.tz_localize(None)
        stock_df['Prev_Close'] = stock_df['Close'].shift(1)
        stock_df['Target_Label'] = (stock_df['Open'] > stock_df['Prev_Close']).astype(int)
        stock_df['Target_Date'] = stock_df.index.date
        stock_df['Stock'] = stock_name
        stock_df = stock_df[['Stock', 'Target_Date', 'Open', 'Prev_Close', 'Target_Label']].dropna()
        all_stock_data.append(stock_df)

df_stock_final = pd.concat(all_stock_data, ignore_index=True)

# 5. รวมข้อมูลและบันทึก
news_grouped['Target_Date'] = pd.to_datetime(news_grouped['Target_Date']).dt.date
df_stock_final['Target_Date'] = pd.to_datetime(df_stock_final['Target_Date']).dt.date

final_dataset = pd.merge(news_grouped, df_stock_final, on=['Stock', 'Target_Date'], how='inner')

vector_df = pd.DataFrame(final_dataset['Vector'].tolist(), index=final_dataset.index)
vector_df.columns = [f'vec_{i}' for i in range(vector_df.shape[1])]
final_dataset = pd.concat([final_dataset.drop(columns=['Vector']), vector_df], axis=1)

final_dataset.to_csv('final_3year_dataset.csv', index=False)
print(f"เสร็จสมบูรณ์! ข้อมูล 3 ปีพร้อมเทรน บันทึกไว้ที่ 'final_3year_dataset.csv' มีข้อมูลทั้งสิ้น {len(final_dataset)} รายการ")