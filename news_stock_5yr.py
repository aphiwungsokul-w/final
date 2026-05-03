import feedparser
import pandas as pd
from datetime import datetime
import urllib.parse
import time
import calendar


def fetch_stock_news():
    # รายชื่อหุ้นที่ต้องการดึงข่าว
    stocks = ["PTT", "AOT", "DELTA", "ADVANC", "SCB"]

    # กำหนดช่วงเวลาย้อนหลัง 5 ปี
    # แบ่งช่วงเวลาเป็นปี เพื่อเลี่ยงข้อจำกัดการดึงข้อมูลสูงสุด 100 รายการของ RSS
    date_ranges = []
    quarters = [
        ("-01-01", "-03-31"),
        ("-04-01", "-06-30"),
        ("-07-01", "-09-30"),
        ("-10-01", "-12-31"),
    ]
    all_news = []

    print("เริ่มกระบวนการดึงข่าว...")

    for stock in stocks:
        print(f"\nกำลังดึงข้อมูลข่าวของหุ้น: {stock}")

        # เพิ่มคำว่า หุ้น ข้างหน้าเพื่อให้ Google News ค้นหาได้ตรงกับข่าวที่เกี่ยวข้องกับหุ้นนั้นๆ มากขึ้น
        query = f'หุ้น "{stock}"'
        encoded_query = urllib.parse.quote(query)

        for year in range(2021, 2027):
            for month in range(1, 13):
                start = f"{year}-{month:02d}-01"
                last_day = calendar.monthrange(year, month)[1]
                end = f"{year}-{month:02d}-{last_day:02d}"

                if start < "2021-04-01" or start >= "2026-04-01":
                    continue
                date_ranges.append((start, end))
                print(f" -> ช่วงเวลา: {start} ถึง {end}")

                # โครงสร้าง URL ของ Google News RSS ที่รองรับการค้นหาแบบกำหนดช่วงเวลา
                # hl=th & gl=TH เพื่อเน้นผลลัพธ์ในประเทศไทย
                url = f"https://news.google.com/rss/search?q={encoded_query}+after:{start}+before:{end}&hl=th&gl=TH&ceid=TH:th"

                # ดึงข้อมูลจาก RSS
                feed = feedparser.parse(url)

                # วนลูปเก็บข้อมูลข่าวแต่ละหัวข้อ
                for entry in feed.entries:
                    # จัดรูปแบบวันที่ให้เป็นมาตรฐาน
                    try:
                        # แปลงจาก string เป็น datetime object
                        published_date = datetime.strptime(
                            entry.published, "%a, %d %b %Y %H:%M:%S %Z"
                        )
                        formatted_date = published_date.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_date = entry.published

                    all_news.append(
                        {
                            "Stock": stock,
                            "Date": formatted_date,
                            "Title": entry.title,
                            "Source": (
                                entry.source.title if "source" in entry else "N/A"
                            ),
                            "Link": entry.link,
                        }
                    )

                # หน่วงเวลา 1 วินาทีเพื่อป้องกันไม่ให้ Google บล็อก IP (Rate Limiting)
                time.sleep(1)

    # สร้าง DataFrame ด้วย Pandas
    df = pd.DataFrame(all_news)

    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if df.empty:
        print("\nไม่พบข้อมูลข่าวในช่วงเวลาที่กำหนด")
    else:
        # ลบข้อมูลที่ซ้ำซ้อนกัน (เผื่อมีข่าวที่ทับซ้อนในช่วงรอยต่อ)
        df = df.drop_duplicates(subset=["Stock", "Title"])

        # เรียงลำดับข้อมูลตามชื่อหุ้นและวันที่
        df = df.sort_values(by=["Stock", "Date"], ascending=[True, False])

        # บันทึกเป็นไฟล์ CSV (ใช้ utf-8-sig เพื่อให้เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน)
        filename = "5year_news_data.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"\n✅ ดึงข้อมูลสำเร็จ! ได้ข่าวทั้งหมด {len(df)} รายการ")
        print(f"✅ บันทึกข้อมูลลงไฟล์ '{filename}' เรียบร้อยแล้ว")


if __name__ == "__main__":
    fetch_stock_news()
