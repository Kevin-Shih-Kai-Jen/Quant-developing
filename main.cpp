#include <iostream>
#include <vector>
#include <cstdlib> // 需要這個來使用 std::system
#include "MarketDataFetcher.h"

// --- 輔助函式區域 ---

// 1. 執行 Python 下載器的函式
bool runPythonDownloader() {
    std::cout << "🐍 正在呼叫 Python 下載資料..." << std::endl;
    
    // 呼叫系統指令 (Windows 用 "python", Mac/Linux 通常用 "python3")
    // system() 會回傳指令執行的狀態碼，0 代表成功
    int result = std::system("python3 fetch_data.py");
    
    if (result != 0) {
        std::cerr << "❌ Python 腳本執行失敗！請檢查環境或檔案是否存在。" << std::endl;
        return false;
    }
    std::cout << "✅ Python 下載完成。" << std::endl;
    return true;
}

// 2. 顯示資料預覽的函式 (只印出前 5 筆和後 5 筆，避免洗版)
void printDataPreview(const std::vector<CandleStick>& history) {
    if (history.empty()) {
        std::cout << "無資料可顯示。" << std::endl;
        return;
    }

    std::cout << "\n📊 資料預覽 (共 " << history.size() << " 筆):" << std::endl;
    std::cout << "===================================" << std::endl;

    // 顯示前 3 筆
    int limit = 3;
    for (size_t i = 0; i < history.size() && i < limit; ++i) {
        std::cout << "日期: " << history[i].timestamp 
                  << " | 收盤: " << history[i].close << std::endl;
    }

    if (history.size() > limit * 2) {
        std::cout << "... (略過中間資料) ..." << std::endl;
    }

    // 顯示後 3 筆 (如果資料夠多的話)
    if (history.size() > limit) {
        for (size_t i = (history.size() > limit ? history.size() - limit : 0); i < history.size(); ++i) {
            std::cout << "日期: " << history[i].timestamp 
                      << " | 收盤: " << history[i].close << std::endl;
        }
    }
    std::cout << "===================================\n" << std::endl;
}

// --- 主程式 ---

int main() {
    // 第一步：呼叫 Python 更新資料
    // 如果 Python 失敗，主程式直接結束，不繼續執行
    if (!runPythonDownloader()) {
        return 1; 
    }

    // 第二步：初始化 C++ 讀取引擎
    MarketDataFetcher fetcher;
    std::string csvFile = "stock_data.csv";
    
    std::cout << "🚀 C++ 引擎啟動，讀取: " << csvFile << std::endl;
    std::vector<CandleStick> history = fetcher.loadHistoryFromCSV(csvFile);

    // 第三步：顯示結果
    printDataPreview(history);

    std::cout << "🎉 程式執行完畢。" << std::endl;
    return 0;
}