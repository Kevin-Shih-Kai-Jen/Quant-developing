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

#endif