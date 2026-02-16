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


// 2. 修改函式輸入，改收 PriceType
std::vector<double> getSpecificDataSets(const std::vector<CandleStick>& history, PriceType type) {
    std::vector<double> dataVector;

    // 3. 使用 switch-case 取代 if-else，這是 C++ 的標準寫法
    switch (type) {
        case PriceType::Open:
            for (const auto& k : history) dataVector.push_back(k.open);
            break;
        case PriceType::High:
            for (const auto& k : history) dataVector.push_back(k.high);
            break;
        case PriceType::Low:
            for (const auto& k : history) dataVector.push_back(k.low);
            break;
        case PriceType::Close:
            for (const auto& k : history) dataVector.push_back(k.close);
            break;
        case PriceType::Volume:
            for (const auto& k : history) dataVector.push_back(k.volume);
            break;
    }

    return dataVector;
}


// --- 主程式 ---
// ... (保留之前的 include 和 runPythonDownloader)

int main() {
    // 1. 下載並讀取資料
    // if (!runPythonDownloader()) return 1;

    MarketDataFetcher fetcher;
    std::vector<CandleStick> history = fetcher.loadHistoryFromCSV("stock_data.csv");
    
    // 2. 計算 5日 SMA (使用 Close 收盤價)
    std::cout << "📈 正在計算技術指標..." << std::endl;
    std::vector<double> closePrices = getSpecificDataSets(history, PriceType::Close);
    std::vector<double> sma5 = TechnicalIndicators::CalculateSMA(5, closePrices);
    std::vector<double> sma20 = TechnicalIndicators::CalculateSMA(20, closePrices);

    // 3. 使用新函式列印結果
    printTechnicalIndicator(history, sma5, "5日均線 (SMA5)", 5);
    printTechnicalIndicator(history, sma20, "20日均線 (SMA20)", 5);

    // 4. 計算 5日 RSI (使用 Close 收盤價)
    std::vector<double> rsi5 = TechnicalIndicators::CalculateRSI(5, closePrices);
    printRSI(history, rsi5, 5, 10);


   // MACD
   MACDResult macd = TechnicalIndicators::CalculateMACD(closePrices);
   printMACD(history, macd);


   // Trading Strategy
    MacdStrategy strategy(12, 26, 9);
    
    // 3. 執行分析
    std::vector<TradeSignal> trade_signal = strategy.Analyze(history);
    
    // 4. ★ 呼叫剛剛寫好的 Print 函式 ★
    PrintTradeSignals(trade_signal);

    BacktestResult back_test_result = BacktestEngine::Run(trade_signal, 100000, history[0].close);
    PrintBacktestResult(back_test_result);
}