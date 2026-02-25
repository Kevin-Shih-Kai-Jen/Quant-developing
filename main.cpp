#include <vector>
#include <cstdlib>    // 需要這個來使用 std::system
#include <algorithm>  // Required for std::transform
#include <cctype>     // toupper(string)

#include "MarketDataFetcher.h"
#include "TechnicalIndicators.h"
#include "printResult.h"
#include "strategy.h"
#include "BacktestEngine.h"


// --- 輔助函式區域 ---

// 1. 執行 Python 下載器的函式
bool runPythonDownloader() {
    std::cout << "🐍 正在呼叫 Python 下載資料..." << std::endl;
    
    // 呼叫系統指令 (Windows 用 "python", Mac/Linux 通常用 "python3")
    // system() 會回傳指令執行的狀態碼，0 代表成功
    int result = std::system("/Users/coolguy/Desktop/quant/venv/bin/python fetch_data.py");
    
    if (result != 0) {
        std::cerr << "❌ Python 腳本執行失敗！請檢查環境或檔案是否存在。" << std::endl;
        return false;
    }
    std::cout << "✅ Python 下載完成。" << std::endl;
    return true;
}


// --- 主程式 ---
// ... (保留之前的 include 和 runPythonDownloader)

int main() {
    // 1. 下載並讀取資料
    if (!runPythonDownloader()) return 1;

    MarketDataFetcher fetcher;
    std::vector<CandleStick> history = fetcher.loadHistoryFromCSV("stock_data.csv");
    
    MixedStrategy strategy_1;
    std::vector<TradeSignal> test = strategy_1.Analyze(history);
    BacktestResult back_test = BacktestEngine::Run(test, 100000, history.back().close, requirePrint::YES);
    PrintBacktestResult(back_test);
}