#ifndef DATASTRUCTURE_H
#define DATASTRUCTURE_H

#include<string>
#include<iostream>
#include<iomanip>
#include<sstream>
#include<vector>
#include<limits>
#include <set>


enum class SignalType {
    HOLD = 0,
    BUY = 1,
    SELL = -1
};


enum class PriceType {
    Open,
    High,
    Low,
    Close,
    Volume
};


struct TradeSignal {
    std::string date;    // 訊號發生的日期
    double price;        // 訊號發生時的價格 (通常是收盤價)
    SignalType type;     // 買入/賣出/持有
    std::string reason;  // 策略說明 (例如 "MACD_Golden_Cross")
    double rsi_val;
    double macd_dif;     // 快線
    double macd_dea;     // 慢線
    double macd_hist;    // histogram （柱狀圖）
    double kVal, dVal, jVal;
};


// K線
struct CandleStick{
    std::string date;
    double open, low, high, close;
    unsigned long long int volume;
};


// 回傳股價的 stirng
struct Stock {
    std::string symbol;
    double price;
    double open, high, low;
    unsigned long long int volume;

    // 這就是你想要的：讓物件自己會講話
    std::string toString() const {
        std::stringstream ss;
        // 設定小數點後兩位
        ss << "[" << symbol << "] 目前股價: " << std::fixed << std::setprecision(2) << price;
        ss << "| 開盤: " << std::fixed << std::setprecision(2) << open;
        ss << "| 最高: " << std::fixed << std::setprecision(2) << high;
        ss << "| 最低: " << std::fixed << std::setprecision(2) << low;
        ss << "| 成交量: " << volume;
        return ss.str();
    }
};


struct MACDResult {
    std::vector<double> dif;       // 快線
    std::vector<double> dea;       // 慢線 (訊號線)
    std::vector<double> histogram; // 柱狀圖
};


struct KdjResult{
    std::vector<double> kValues;
    std::vector<double> dValues;
    std::vector<double> jValues;
};


struct bestKDJn{
    int RSV_N = 1;
    int K_N = 1;
    int D_N = 1;
};

struct BacktestResult{
    double TotalProfit;
    int TotalTrades;
    double WinRate;
    int remaining_stock;
    double ROI;
};


struct BBandResult{
    std::vector<double> middleBand, upperBand, lowerBand;
    std::vector<double> standardDeviation;
};


//============================================ 輔助函式 ============================================//
// 2. 修改函式輸入，改收 PriceType
inline std::vector<double> getSpecificDataSets(const std::vector<CandleStick>& history, PriceType type) {
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


// 只拿 dates
inline std::vector<std::string> getDateData(const std::vector<CandleStick>& history) {
    std::vector<std::string> dateVector;
    for (const auto& k : history) dateVector.push_back(k.date);
    return dateVector;
}


// 不然 RUN 一直跑出訊息太多了
enum requirePrint{
    YES,
    NO
};


struct Crosses{
    std::set<int> golden_cross, death_cross;
};

#endif