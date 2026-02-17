#ifndef STRATEGY_H
#define STRATEGY_H


#include "DataStructures.h"
#include "TechnicalIndicators.h"


class Strategy{
public:
    Strategy() = default;
    virtual ~Strategy() = default;
    
    virtual std::vector<TradeSignal> Analyze(const std::vector<CandleStick>& data) = 0;
};


class MacdStrategy: public Strategy{
private:
    int FastPeriod, SlowPeriod, SignalPeriod;

public:
    MacdStrategy(int fast, int slow, int signal): FastPeriod(fast), SlowPeriod(slow), SignalPeriod(signal){};
    ~MacdStrategy() = default;
    std::vector<TradeSignal> Analyze(const std::vector<CandleStick>& data) override;
};


class KDJstrategy: public Strategy{
private:
    int rsv_n, k_n, d_n;

public:
    KDJstrategy(int RSV_N, int K_N, int D_N): rsv_n(RSV_N), k_n(K_N), d_n(D_N){};
    ~KDJstrategy() = default;
    std::vector<TradeSignal> Analyze(const std::vector<CandleStick>& data) override;
    static bestKDJn FindBestParameters(const std::vector<CandleStick>& data);
};


#endif