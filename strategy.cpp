#include <cmath>

#include "strategy.h"
#include "printResult.h"
#include "BacktestEngine.h"


std::vector<TradeSignal> MacdStrategy::Analyze(const std::vector<CandleStick>& data){
    std::vector<double> dataVector;
    for (const auto&k : data) dataVector.push_back(k.volume);

    MACDResult macd = TechnicalIndicators::CalculateMACD(FastPeriod, SlowPeriod, SignalPeriod, dataVector, 0, data.size() - 1);
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

    KdjResult KDJ = TechnicalIndicators::CalculateKDJ(closePrices, rsv_n, k_n, d_n, 0, data.size() - 1);

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


std::vector<TradeSignal> MixedStrategy::Analyze(const std::vector<CandleStick>& data) {
    std::vector<TradeSignal> trade_signal;
    if (data.size() < 243) return trade_signal; // 確保有足夠年線資料

    // 取得資料
    std::vector<double> closePrices = getSpecificDataSets(data, PriceType::Close);
    std::vector<double> volumes = getSpecificDataSets(data, PriceType::Volume);

    // 計算指標
    auto bband_20 = TechnicalIndicators::CalculateBBand(closePrices, 20, 0, data.size() - 1);
    auto sma5 = TechnicalIndicators::CalculateSMA(5, closePrices, 0, data.size() - 1);
    auto sma22 = TechnicalIndicators::CalculateSMA(22, closePrices, 0, data.size() - 1);
    auto sma243 = TechnicalIndicators::CalculateSMA(243, closePrices, 0, data.size() - 1);
    auto macd_price = TechnicalIndicators::CalculateMACD(10, 22, 5, closePrices, 0, data.size() - 1);
    auto macd_vol = TechnicalIndicators::CalculateMACD(10, 22, 5, volumes, 0, data.size() - 1);
    auto kdj = TechnicalIndicators::CalculateKDJ(closePrices, 6, 2, 2, 0, data.size() - 1);

    auto cross_ma = CrossLine(closePrices, sma22, 0, closePrices.size());
    auto cross_macd_prices = CrossLine(macd_price.dif, macd_price.dea, 0, macd_price.dif.size());

    std::deque<int> dq; // 紀錄均線穿越的時間點

    // 起點：至少要有一段歷史資料（例如 120 天）來算 Percentile
    for (int i = 120; i < data.size(); i++) {
        // 更新滑動視窗 (10天內)
        while (!dq.empty() && dq.front() < i - 10) {
            dq.pop_front();
        }
        if (cross_ma.golden_cross.count(i) || cross_ma.death_cross.count(i)) {
            dq.push_back(i);
        }

        // =========================== 策略 1: 趨勢突破 =========================== //
        // 修正：往回看 120 天
        double p80 = TechnicalIndicators::PercentVal(bband_20.standardDeviation, i - 120, i, 0.8);
        
        // 價格在年線上 (sma243)
        if (closePrices[i] > sma243[i] && bband_20.standardDeviation[i] > p80) {
            // MACD 快線剛好穿過慢線 (黃金交叉)
            if (cross_macd_prices.golden_cross.count(i) && macd_vol.histogram[i] > 0) {
                trade_signal.push_back({data[i].date, data[i].close, SignalType::BUY, "Trend Breakout (P80 BBW)"});
                continue;
            }
        }

        // =========================== 策略 2: 盤整噴發前兆 =========================== //
        double p20 = TechnicalIndicators::PercentVal(bband_20.standardDeviation, i - 120, i, 0.2);
        
        // 波動率極低 (P20) 且 近期頻繁穿梭均線 (盤整)
        if (bband_20.standardDeviation[i] < p20 && dq.size() >= 3) {
            // 在低波動盤整時，如果 MACD 出現向上的動能
            if (macd_price.histogram[i] > 0 && macd_price.histogram[i-1] < macd_price.histogram[i] && macd_vol.histogram[i] >= 0) {
                trade_signal.push_back({data[i].date, data[i].close, SignalType::BUY, "MACD good"});
            }

            // 盤整間的套利 --> 價格平均位於低檔（KD 在低檔）
            if (kdj.kValues[i - 1] < kdj.dValues[i - 1] && kdj.kValues[i] > kdj.dValues[i] && kdj.kValues[i] < 20){
                trade_signal.push_back({data[i].date, data[i].close, SignalType::BUY, "Consolidation, kdj is low, kdj golden cross"});
            }
        }


        //========================================= 賣 =========================================//
        // 跌破 5 日線
        if (closePrices[i] <= sma5[i]){
            trade_signal.push_back({data[i].date, data[i].close, SignalType::SELL, "Drop below MA5"});
        }

        // 盤整時的低點
        if (bband_20.standardDeviation[i] < p20 && dq.size() >= 3){
            if (macd_price.histogram[i] < 0 && macd_price.histogram[i-1] > macd_price.histogram[i] && macd_vol.histogram[i] < 0) {
                trade_signal.push_back({data[i].date, data[i].close, SignalType::SELL, "MACD bad"});
            }
        }

        // 盤整間的套利 --> 價格平均位於低檔（KD 在低檔）
        if (kdj.kValues[i - 1] > kdj.dValues[i - 1] && kdj.kValues[i] < kdj.dValues[i] && kdj.kValues[i] > 80){
            trade_signal.push_back({data[i].date, data[i].close, SignalType::SELL, "Consolidation, kdj is high, kdj death cross"});
        }

    }

    return trade_signal;
}