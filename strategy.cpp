#include "strategy.h"


std::vector<TradeSignal> MacdStrategy::Analyze(const std::vector<CandleStick>& data){
    std::vector<double> dataVector;
    for (const auto&k : data) dataVector.push_back(k.close);

    MACDResult macd = TechnicalIndicators::CalculateMACD(dataVector, FastPeriod, SlowPeriod, SignalPeriod);
    int TOTAL_OFFSET = (SlowPeriod - 1) + (SignalPeriod - 1); // minus one since the start of the day has data
    std::vector<TradeSignal> signal;

    for (int i = 1; i < macd.histogram.size(); i ++){
        double prev_hist = macd.histogram[i - 1];
        double curr_hist = macd.histogram[i];
        
        // Buy in MACD_Golden_Cross
        if (prev_hist < 0 and curr_hist > 0){
            TradeSignal item;
            item.date = data[i + TOTAL_OFFSET].date;
            item.price = data[i + TOTAL_OFFSET].close;
            item.type = SignalType::BUY;
            item.reason = "MACD_Golden_Cross";

            // 1. DIF (快線) 比較長，它比 DEA 多出了一個 "SignalPeriod - 1" 的偏移量
            item.macd_dif = macd.dif[i+ (SignalPeriod - 1)];

            // 2. DEA (慢線) 和 Histogram 是完全同步的，所以索引直接對應
            item.macd_dea = macd.dea[i];
            
            item.macd_hist = macd.histogram[i];

            signal.push_back(item);
        }

        // Sell in MACD_Golden_Cross
        if (prev_hist > 0 and curr_hist < 0){
            TradeSignal item;
            item.date = data[i + TOTAL_OFFSET].date;
            item.price = data[i + TOTAL_OFFSET].close;
            item.type = SignalType::SELL;
            item.reason = "MACD_Death_Cross";

            // 1. DIF (快線) 比較長，它比 DEA 多出了一個 "SignalPeriod - 1" 的偏移量
            item.macd_dif = macd.dif[i + (SignalPeriod - 1)];

            // 2. DEA (慢線) 和 Histogram 是完全同步的，所以索引直接對應
            item.macd_dea = macd.dea[i];
            
            item.macd_hist = macd.histogram[i];

            signal.push_back(item);
        }
    }
    return signal;
};


