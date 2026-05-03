import yfinance as yf
import pandas as pd

# 1. กำหนดรายชื่อหุ้นและช่วงวันที่
tickers = ['PTT.BK', 'AOT.BK', 'DELTA.BK', 'ADVANC.BK', 'SCB.BK']
start_date = "2021-04-01"
end_date = "2026-04-02"

# 2. ดึงข้อมูลจาก Yahoo Finance
print(f"กำลังดึงข้อมูลตั้งแต่ {start_date} ถึง {end_date}...")
df = yf.download(tickers, start=start_date, end=end_date)

# 3. เลือกเฉพาะข้อมูลที่เราต้องการ
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 4. แปลงตารางให้อยู่ในรูปแบบที่มีคอลัมน์ Ticker (Long Format)
# กำหนดชื่อให้ระดับของคอลัมน์ (Index level) เพื่อเตรียมแปลงข้อมูล
df.columns.names = ['Attributes', 'Ticker']

# ใช้ .stack() ดึงชื่อหุ้นลงมาเป็นแถว และใช้ .reset_index() เพื่อให้ Date กลับมาเป็นคอลัมน์ปกติ
df_long = df.stack(level='Ticker').reset_index()

# จัดเรียงคอลัมน์ใหม่ให้สวยงาม โดยเอา Date และ Ticker ขึ้นก่อน
df_long = df_long[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

# 5. แสดงผลลัพธ์
print("\n--- ตัวอย่างข้อมูล 10 บรรทัดแรก ---")
print(df_long.head(10))

print("\n--- ตัวอย่างข้อมูล 10 บรรทัดสุดท้าย ---")
print(df_long.tail(10))

csv_filename = "price_stock_5yr.csv"
df_long.to_csv(csv_filename, index=False)