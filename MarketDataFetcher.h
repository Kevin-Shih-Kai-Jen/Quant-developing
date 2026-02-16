#ifndef MARKET_DATA_FETCHER_H
#define MARKET_DATA_FETCHER_H

#include <vector>

#include "DataStructures.h"




class MarketDataFetcher {
public:
    // 1. 建構子 (Constructor) - 負責 init
    MarketDataFetcher();

    // 2. 解構子 (Destructor) - 負責 cleanup
    ~MarketDataFetcher();

    std::vector<CandleStick> loadHistoryFromCSV(const std::string& filename);
};

#endif // MARKET_DATA_FETCHER_H