# fetch_data.py
import yfinance as yf
import pandas as pd

def download_data(symbol, filename):
    print(f"正在下載 {symbol} 的資料...")
    
    # 下載最近 1 個月的日線資料
    # auto_adjust=True 會幫我們處理除權息的問題，這對回測很重要
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True)
    
    # 檢查有沒有下載到
    if df.empty:
        print("❌ 下載失敗，可能是代號錯誤或網路問題")
        return

    # 把索引 (Date) 變成一個實體欄位，方便存成 CSV
    df.reset_index(inplace=True)
    
    # 存成 CSV
    df.to_csv(filename, index=False)
    print(f"✅ 成功！資料已儲存為 {filename}")
    print(df.head()) # 印出前幾筆看看

if __name__ == "__main__":
    # 你可以在這裡改股票代號
    download_data("2330.TW", "stock_data.csv")