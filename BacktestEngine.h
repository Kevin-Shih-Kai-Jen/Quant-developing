#ifndef BACK_TEST_ENGINE_h
#define BACK_TEST_ENGINE_h

#include "DataStructures.h"

class BacktestEngine{
public:
    BacktestEngine() = default;
    ~BacktestEngine() = default;
    
    static BacktestResult Run(const std::vector<TradeSignal>& data, double initial_money, double curr_stock_price, requirePrint require_print = YES);
};

# endif