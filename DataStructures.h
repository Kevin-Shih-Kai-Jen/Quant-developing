#ifndef DATASTRUCTURE_H
#define DATASTRUCTURE_H

#include<string>
#include<iostream>
#include<iomanip>
#include<sstream>
#include<vector>

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


struct BacktestResult{
    double TotalProfit;
    int TotalTrades;
    double WinRate;
};

#endif