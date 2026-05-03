import feedparser
import pandas as pd
from datetime import datetime
import time
import urllib.parse
import os

# --- ตั้งค่า ---
TARGET_STOCKS = ['PTT', 'AOT', 'DELTA', 'ADVANC', 'SCB']
CSV_FILE = 'news_data.csv'
INTERVAL_MINUTES = 15 # ตั้งเวลาดึงข้อมูลใหม่ทุกๆ 15 นาที

# เก็บ link ข่าวที่ดึงมาแล้วเพื่อกันซ้ำ
seen_links = set()

# โหลด link เก่าจากไฟล์ CSV (ถ้ามี) มาใส่ในเซ็ต เพื่อให้รันโปรแกรมใหม่แล้วข่าวไม่ซ้ำเดิม
if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE)
    if 'Link' in df_existing.columns:
        seen_links.update(df_existing['Link'].tolist())

def fetch_realtime_news():
    new_articles = []
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] กำลังดึงข่าวอัปเดต...")

    for stock in TARGET_STOCKS:
        # ใช้ when:1d สำหรับดึงข่าวย้อนหลัง 24 ชั่วโมง
        # (หากต้องการดึงแค่วันนี้วันเดียวเป๊ะๆ สามารถใช้ when:12h หรือระบุวันที่ได้)
        query = f'หุ้น {stock} when:1d'
        encoded_query = urllib.parse.quote(query)
        
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=th&gl=TH&ceid=TH:th"
        
        try:
            feed = feedparser.parse(rss_url)
            
            if hasattr(feed, 'status') and feed.status == 429:
                print(" โดนบล็อคจาก Google พัก 60 วิ...")
                time.sleep(60)
                continue
            
            if feed.entries:
                for entry in feed.entries:
                    link = entry.link
                    
                    # เช็คว่าข่าวนี้เคยดึงมาหรือยัง
                    if link not in seen_links:
                        seen_links.add(link)
                        
                        # ดึงเวลาที่ตีพิมพ์ (เพิ่ม ชัวโมง:นาที:วินาที เข้าไปเพื่อใช้ทำนายราคาเปิดได้แม่นยำขึ้น)
                        try:
                            dt_struct = entry.published_parsed
                            pub_date = datetime(*dt_struct[:6]).strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pub_date = entry.published
                            
                        new_articles.append({
                            'Stock': stock,
                            'Source': entry.source.title if 'source' in entry else 'Google News',
                            'Date': pub_date,
                            'Title': entry.title,
                            'Link': link
                        })
        except Exception as e:
            print(f" Error สำหรับ {stock}: {e}")

        # พัก 2 วินาทีระหว่างเปลี่ยนหุ้น เพื่อไม่ให้ Google มองว่าเป็น Bot สแปม
        time.sleep(2) 

    # --- บันทึกข้อมูลลง CSV ---
    if new_articles:
        df_new = pd.DataFrame(new_articles)
        
        # ถ้ามีไฟล์อยู่แล้วให้ append (ต่อท้าย) โหมด 'a'
        if os.path.exists(CSV_FILE):
            df_new.to_csv(CSV_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            # ถ้ายังไม่มีไฟล์ สร้างใหม่พร้อมใส่ Header
            df_new.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
            
        print(f"เจอข่าวใหม่ {len(new_articles)} รายการ บันทึกลง {CSV_FILE} แล้ว")
        for article in new_articles:
            print(f"  -> [{article['Stock']}] {article['Title']}")
    else:
        print("ยังไม่มีข่าวใหม่ในรอบนี้")

def start_polling():
    print(f"เริ่มต้นระบบดึงข่าวแบบ Real-time (ทำงานทุกๆ {INTERVAL_MINUTES} นาที)")
    print("กด Ctrl+C เพื่อหยุดการทำงาน")
    
    try:
        while True:
            fetch_realtime_news()
            print(f"รอ {INTERVAL_MINUTES} นาที เพื่อตรวจสอบข่าวรอบถัดไป...")
            time.sleep(INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        print("\nหยุดการทำงานเรียบร้อยแล้ว")

if __name__ == "__main__":
    start_polling()