#ifndef PRINT_RESULT_H
#define PRINT_RESULT_H

#include "DataStructures.h"

// 2. 顯示資料預覽的函式 (只印出前 5 筆和後 5 筆，避免洗版)
inline void printDataPreview(const std::vector<CandleStick>& history) {
    if (history.empty()) {
        std::cout << "無資料可顯示。" << std::endl;
        return;
    }

    std::cout << "\n📊 資料預覽 (共 " << history.size() << " 筆):" << std::endl;
    std::cout << "===================================" << std::endl;

    // 顯示前 3 筆
    int limit = 3;
    for (size_t i = 0; i < history.size() && i < limit; ++i) {
        std::cout << "日期: " << history[i].date 
                  << " | 收盤: " << history[i].close << std::endl;
    }

    if (history.size() > limit * 2) {
        std::cout << "... (略過中間資料) ..." << std::endl;
    }

    // 顯示後 3 筆 (如果資料夠多的話)
    if (history.size() > limit) {
        for (size_t i = (history.size() > limit ? history.size() - limit : 0); i < history.size(); ++i) {
            std::cout << "日期: " << history[i].date 
                      << " | 收盤: " << history[i].close << std::endl;
        }
    }
    std::cout << "===================================\n" << std::endl;
}


inline void printTechnicalIndicator(const std::vector<CandleStick>& history, 
                             const std::vector<double>& indicator_data, 
                             std::string indicator_name, 
                             int count = 5) {
    
    std::cout << "🔍 " << indicator_name << " 驗證 (最後 " << count << " 筆):" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    int data_len = history.size();
    int ind_len = indicator_data.size();

    // 防呆機制：如果算出來的指標是空的，就不要印
    if (ind_len == 0) {
        std::cout << "❌ 指標資料為空！" << std::endl;
        return;
    }

    // 設定小數點顯示 2 位
    std::cout << std::fixed << std::setprecision(2);

    for (int i = 0; i < count; i++) {
        // 從尾巴往回推 (0 代表最後一筆, 1 代表倒數第二筆...)
        int h_idx = data_len - 1 - i;
        int s_idx = ind_len - 1 - i;

        // 確保沒有超出陣列範圍 (尤其是指標剛開始計算的那幾天)
        if (h_idx >= 0 && s_idx >= 0) {
            std::cout << "📅 日期: " << history[h_idx].date
                      << " | 收盤價: " << std::setw(6) << history[h_idx].close 
                      << " | " << indicator_name << ": " << std::setw(6) << indicator_data[s_idx] 
                      << std::endl;
        }
    }
    std::cout << "------------------------------------------------------" << std::endl;
}


// 專門用來印 RSI 的函式，處理了 index 對齊問題
inline void printRSI(const std::vector<CandleStick>& history, 
              const std::vector<double>& rsi_data, 
              int period, 
              int count = 5) {
    
    std::cout << "🌊 RSI(" << period << ") 驗證 (最後 " << count << " 筆):" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // 確保不會印超過範圍
    int rsi_len = rsi_data.size();
    if (rsi_len == 0) return;

    std::cout << std::fixed << std::setprecision(2);

    // 我們從 RSI 的最後一筆往回印
    for (int i = 0; i < count; i++) {
        int rsi_idx = rsi_len - 1 - i;
        
        // 關鍵：RSI 的 index 加上 period 才是 history 的 index
        int h_idx = rsi_idx + period; 

        if (rsi_idx >= 0 && h_idx < history.size()) {
             // 根據 RSI 數值給一點視覺提示
            std::string signal = "";
            if (rsi_data[rsi_idx] > 70) signal = " 🔥 (超買)";
            else if (rsi_data[rsi_idx] < 30) signal = " ❄️ (超賣)";

            std::cout << "📅 日期: " << history[h_idx].date
                      << " | 收盤: " << std::setw(6) << history[h_idx].close 
                      << " | RSI: " << std::setw(6) << rsi_data[rsi_idx] 
                      << signal
                      << std::endl;
        }
    }
    std::cout << "------------------------------------------------------" << std::endl;
}


// 專門印 MACD 的函式
// 技巧：我們以「最短」的柱狀圖 (Histogram) 為基準，向後對齊
inline void printMACD(const std::vector<CandleStick>& history, 
               const MACDResult& macd, 
               int count = 5) {
    
    std::cout << "📊 MACD(12, 26, 9) 驗證 (最後 " << count << " 筆):" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    // 防呆：如果柱狀圖沒算出來，代表資料太少，整個 MACD 都沒意義
    if (macd.histogram.empty()) {
        std::cout << "❌ 資料不足，無法計算 MACD" << std::endl;
        return;
    }

    std::cout << std::fixed << std::setprecision(2);

    // 我們以 histogram 的長度為基準 (它是最短的)
    int len = macd.histogram.size();

    for (int i = 0; i < count; i++) {
        // 從最後一筆往回推
        int idx = len - 1 - i;
        
        // 對齊原始資料：
        // 雖然我們不知道具體的 offset 是多少，但我們知道
        // history 的最後一筆 = histogram 的最後一筆
        int h_idx = history.size() - 1 - i;

        // 對齊 DIF 和 DEA：
        // 同理，大家都靠右對齊
        int dif_idx = macd.dif.size() - 1 - i;
        int dea_idx = macd.dea.size() - 1 - i;

        if (idx >= 0 && h_idx >= 0) {
            double bar = macd.histogram[idx];
            
            // 視覺化：正數用綠色，負數用紅色 (有些終端機可能不支援顏色，用文字符號代替)
            std::string bar_visual = (bar >= 0) ? "🟩 " : "🟥 ";
            
            std::cout << "📅 " << history[h_idx].date 
                      << " | DIF: " << std::setw(6) << macd.dif[dif_idx]
                      << " | DEA: " << std::setw(6) << macd.dea[dea_idx]
                      << " | BAR: " << bar_visual << std::setw(6) << bar
                      << std::endl;
        }
    }
    std::cout << "----------------------------------------------------------------------" << std::endl;
}


// 輔助函式：把 SignalType (Enum) 轉成字串
inline std::string SignalTypeToString(SignalType type) {
    if (type == SignalType::BUY) return "BUY ";
    if (type == SignalType::SELL) return "SELL";
    return "NONE";
}

// 主函式：印出漂亮的表格
inline void PrintTradeSignals(const std::vector<TradeSignal>& signals) {
    std::cout << "========================================================================================\n";
    std::cout << "                                  TRADE SIGNALS REPORT                                  \n";
    std::cout << "========================================================================================\n";
    
    // 1. 印出表頭 (Header)
    std::cout << std::left 
              << std::setw(12) << "Date" 
              << std::setw(8)  << "Type" 
              << std::setw(10) << "Price" 
              << std::setw(20) << "Reason" 
              << std::setw(20) << "DIF" 
              << std::setw(10) << "DEA" 
              << std::setw(10) << "Hist" 
              << std::endl;

    std::cout << "----------------------------------------------------------------------------------------\n";

    if (signals.empty()) {
        std::cout << "No signals found." << std::endl;
        return;
    }

    // 2. 設定小數點格式
    std::cout << std::fixed << std::setprecision(2); // 價格顯示 2 位小數

    // 3. 逐行印出資料 (使用 rbegin 和 rend 從後面往回印)
    for (auto it = signals.rbegin(); it != signals.rend(); ++it) {
        std::cout << std::left 
                  << std::setw(12) << it->date 
                  << std::setw(8)  << SignalTypeToString(it->type) 
                  << std::setw(10) << it->price 
                  << std::setw(20) << it->reason;
        
        // MACD 數值切換精確度 (4 位小數)
        std::cout << std::setprecision(1) 
                  << std::setw(10) << it->macd_dif 
                  << std::setw(10) << it->macd_dea 
                  << std::setw(10) << it->macd_hist 
                  << std::setprecision(2) // 切換回價格用的 2 位小數
                  << std::endl;
    }
    std::cout << "========================================================================================\n";
    std::cout << "Total Signals: " << signals.size() << std::endl;
}


// 輔助函式：根據數值正負回傳帶顏色的字串
inline std::string ColorText(double value, int precision = 2, std::string unit = "") {
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision);
    
    if (value > 0) {
        ss << GREEN << "+" << value << unit << RESET;
    } else if (value < 0) {
        ss << RED << value << unit << RESET; // 負數自帶負號
    } else {
        ss << value << unit;
    }
    return ss.str();
}


inline void PrintBacktestResult(const BacktestResult& result) {
    std::cout << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "           BACKTEST PERFORMANCE           " << std::endl;
    std::cout << "==========================================" << std::endl;

    // 1. 基礎統計
    std::cout << std::left << std::setw(25) << "Total Trades:" 
              << result.TotalTrades << std::endl;

    std::cout << std::left << std::setw(25) << "Win Rate:" 
              << std::fixed << std::setprecision(2) << result.WinRate << "%" << std::endl;
    
    std::cout << std::left << std::setw(25) << "Stock on Hold:" 
              << result.remaining_stock << std::endl;

    // 2. 財務表現 (使用 ColorText 簡化)
    std::cout << std::left << std::setw(25) << "Total P/L:" 
              << ColorText(result.TotalProfit) << std::endl;

    // 加入 ROI (投資報酬率)
    std::cout << std::left << std::setw(25) << "ROI:" 
              << ColorText(result.ROI, 2, "%") << std::endl;

    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;
}

inline void PrintKdj(const KdjResult& result, const std::vector<std::string>& dates, int limit = 20) {
    // 1. 基本檢查
    size_t total_size = result.kValues.size();
    if (dates.size() != total_size) {
        std::cerr << "錯誤：日期數量與 KDJ 數據數量不符！" << std::endl;
        return;
    }

    // 2. 計算起始點 (start_index)
    size_t start_index = 0;
    if (limit > 0 && limit < total_size) {
        start_index = total_size - limit;
    }

    std::cout << "=========================================================" << std::endl;
    // 顯示標題，如果有限制，可以提示一下
    std::cout << "KDJ Table (Last " << (limit == 0 ? total_size : limit) << " Days)" << std::endl;
    std::cout << std::left << std::setw(15) << "Date" 
              << std::right << std::setw(10) << "K" 
              << std::setw(10) << "D" 
              << std::setw(10) << "J" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    std::cout << std::fixed << std::setprecision(2);

    // 3. 迴圈從 start_index 開始，而不是從 0 開始
    for (size_t i = start_index; i < total_size; ++i) {
        // 根據 J 值加入簡單的顏色標示 (Mac/Linux Only, Windows 可拿掉)
        std::string color = "\033[0m"; // Reset
        if (result.jValues[i] < 0) color = "\033[32m";      // 綠色 (超賣/潛在買點)
        else if (result.jValues[i] > 100) color = "\033[31m"; // 紅色 (超買/潛在賣點)

        std::cout << std::left << std::setw(15) << dates[i] 
                  << std::right << std::setw(10) << result.kValues[i] 
                  << std::setw(10) << result.dValues[i] 
                  << color << std::setw(10) << result.jValues[i] << "\033[0m" << std::endl;
    }
    std::cout << "=========================================================" << std::endl;
}


// 1. 印出優化過程的表頭
inline void PrintOptimizationHeader() {
    std::cout << "🚀 開始尋找 KDJ 最佳參數 (暴力搜尋中...)" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left  << std::setw(10) << "狀態" 
              << std::right << std::setw(6)  << "RSV" 
              << std::setw(6)  << "K" 
              << std::setw(6)  << "D" 
              << std::setw(15) << "淨利 (Profit)" 
              << std::setw(12) << "交易次數"
              << std::setw(12) << "勝率 (%)" 
              << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
}

// 2. 印出每一筆新的最佳紀錄
inline void PrintNewBestRecord(int rsv, int k, int d, const BacktestResult& result) {
    std::cout << std::left  << std::setw(10) << "🏆 新冠軍" 
              << std::right << std::setw(6)  << rsv 
              << std::setw(6)  << k 
              << std::setw(6)  << d 
              << std::fixed << std::setprecision(2) << std::setw(15) << result.TotalProfit
              << std::setw(12) << result.TotalTrades 
              << std::setw(11) << result.WinRate << "%"
              << result.remaining_stock
              << std::endl;
}


#endif