import pandas as pd
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize

# 1. กำหนด Stopwords และคำเฉพาะที่ต้องการเก็บไว้ (ลอจิกเดียวกับโปรเจกต์คุณ)
_STOPWORDS = frozenset(thai_stopwords()) - {
    'ขึ้น', 'ลง', 'บวก', 'ลบ', 'กำไร', 'ขาดทุน', 'สูง', 'ต่ำ', 'ซื้อ' , 'อัพ' , 'Up' , 'บาย' , 'Buy' ,
    'เพิ่ม', 'ลด', 'แข็ง', 'อ่อน', 'พุ่ง', 'ร่วง', 'ฟื้น', 'ซบ', 'ขาย', 'ดาวน์' , 'Down' , 'เซลล์' , 'Sell' , 'เท'
}

# 2. ฟังก์ชันทำความสะอาดและตัดคำ
def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    # ตัด source tag ท้าย
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

# 3. ฟังก์ชันหลักสำหรับอ่านไฟล์ เทียบข้อความ และบันทึก
def test_tokenization(input_file='5year_news.csv', output_file='news_tokenized.csv'):
    print(f"กำลังอ่านไฟล์: {input_file}")
    try:
        df = pd.read_csv(input_file)
        
        print(f"เจอข้อมูลทั้งหมด {len(df):,} แถว")
        print("กำลังตัดคำและทำความสะอาดข้อความ (อาจใช้เวลาสักครู่)...")
        
        # สร้างคอลัมน์ใหม่เพื่อดูผลลัพธ์
        df['Processed_Title'] = df['Title'].apply(preprocess)
        
        # จัดเรียงคอลัมน์ให้ดูเปรียบเทียบง่ายๆ
        df_preview = df[['Date', 'Stock', 'Title', 'Processed_Title']]
        
        # บันทึกเป็น CSV (ใช้ utf-8-sig เพื่อให้เปิดใน Excel ภาษาไทยได้ไม่เพี้ยน)
        df_preview.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ บันทึกผลลัพธ์สำเร็จ: {output_file}")
        
        # ปริ้นตัวอย่างให้ดูใน Console 3 แถว
        print("\n--- ตัวอย่างผลการตัดคำ (3 แถวแรก) ---")
        for i in range(min(3, len(df_preview))):
            print(f"[{i+1}] ต้นฉบับ  : {df_preview.iloc[i]['Title']}")
            print(f"    ตัดคำแล้ว: {df_preview.iloc[i]['Processed_Title']}\n")
            
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ '{input_file}' กรุณาตรวจสอบว่ามีไฟล์นี้ในโฟลเดอร์เดียวกับโค้ดหรือไม่")

if __name__ == '__main__':
    test_tokenization()