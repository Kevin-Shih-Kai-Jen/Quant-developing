#include "strategy.h"
#include "printResult.h"
#include "BacktestEngine.h"


std::vector<TradeSignal> MacdStrategy::Analyze(const std::vector<CandleStick>& data){
    std::vector<double> dataVector;
    for (const auto&k : data) dataVector.push_back(k.volume);

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


std::vector<TradeSignal> KDJstrategy::Analyze(const std::vector<CandleStick>& data){
    std::vector<TradeSignal> dataVector;
    std::vector<double> closePrices = getSpecificDataSets(data, PriceType::Close);
    std::vector<std::string> dates = getDateData(data);

    KdjResult KDJ = TechnicalIndicators::CalculateKDJ(closePrices, rsv_n, k_n, d_n);

    for (int i = 1; i < data.size(); i++){
        //Buy
        if ((KDJ.kValues[i] <= 20 || KDJ.dValues[i] <= 20) && KDJ.kValues[i - 1] < KDJ.dValues[i - 1] && KDJ.kValues[i] >= KDJ.dValues[i]){
            TradeSignal signal;
            signal.date = data[i].date;
            signal.price = data[i].close;
            signal.type = SignalType::BUY;
            signal.reason = "KDJ's K or D <= 20 and Golden Cross";
            signal.kVal = KDJ.kValues[i];
            signal.dVal = KDJ.dValues[i];
            
            dataVector.push_back(signal);
        }

        // Sell
        if ((KDJ.kValues[i] >= 80|| KDJ.dValues[i] >= 80) && KDJ.kValues[i - 1] > KDJ.dValues[i - 1] && KDJ.kValues[i] <= KDJ.dValues[i]){
            TradeSignal signal;
            signal.date = data[i].date;
            signal.price = data[i].close;
            signal.type = SignalType::SELL;
            signal.reason = "KDJ's K or D >= 80 and Death Cross";
            signal.kVal = KDJ.kValues[i];
            signal.dVal = KDJ.dValues[i];

            dataVector.push_back(signal);
        }
    }
    return dataVector;
}


bestKDJn KDJstrategy::FindBestParameters(const std::vector<CandleStick>& data) {
    BacktestResult bestResult;
    // 初始值設為負無限大
    bestResult.TotalProfit = -std::numeric_limits<double>::infinity(); 
    
    bestKDJn bestParam;

    // 👇 呼叫函式印表頭，一行搞定！
    PrintOptimizationHeader();

    for (int i = 2; i <= 240; i++) {
        for (int j = 2; j <= 10; j++) {
            for (int k = 2; k <= 10; k++) {
                
                KDJstrategy kdj_strategy(i, j, k);
                std::vector<TradeSignal> kdj_trade_signal = kdj_strategy.Analyze(data);
                BacktestResult current_result = BacktestEngine::Run(kdj_trade_signal, 100000, data[0].close, requirePrint::NO);

                if (current_result.TotalProfit > bestResult.TotalProfit) {
                    bestResult = current_result;
                    bestParam.RSV_N = i;
                    bestParam.K_N = j;
                    bestParam.D_N = k;
                }
            }
        }
    }

    // 印出最終詳細結果
    PrintBacktestResult(bestResult);

    std::cout << "最佳參數組合: RSV=" << bestParam.RSV_N 
                              << ", K=" << bestParam.K_N 
                              << ", D=" << bestParam.D_N << std::endl;

    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    std::cout << "✅ 搜尋結束！" << std::endl;
    


    return bestParam;    
}
