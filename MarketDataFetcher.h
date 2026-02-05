#ifndef MARKET_DATA_FETCHER_H
#define MARKET_DATA_FETCHER_H

#include <string>
#include <sstream> // stringstream
#include <iomanip> // setprecision
#include <vector>

using Price = double;


// K線
struct CandleStick{
    std::string timestamp;
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

class MarketDataFetcher {
public:
    // 1. 建構子 (Constructor) - 負責 init
    MarketDataFetcher();

    // 2. 解構子 (Destructor) - 負責 cleanup
    ~MarketDataFetcher();

    std::vector<CandleStick> loadHistoryFromCSV(const std::string& filename);
};

#endif // MARKET_DATA_FETCHER_H