#include<deque>

#include "BacktestEngine.h"


BacktestResult BacktestEngine::Run(const std::vector<TradeSignal>& signals, double initial_money, double curr_stock_price, requirePrint require_print) {
    BacktestResult result;
    // 初始化結果，避免垃圾值
    result.TotalProfit = 0.0;
    result.TotalTrades = 0;
    result.WinRate = 0.0;
    result.remaining_stock = 0.0;

    // 用來算 ROI
    double TotalExpenditure = 0.0;
    double TotalEarned = 0.0;

    double cash = initial_money;
    std::deque<double> stock_inventory; // 庫存成本隊列
    int win_trades = 0; // 賺錢的次數

    if (require_print == requirePrint::YES){
        std::cout << "========== 開始回測 ==========" << std::endl;
    }

    for (const auto& sig : signals) {
        switch (sig.type) {
            case SignalType::BUY:{
                // 檢查資金是否足夠
                if (cash < sig.price) {
                    std::cout << "[警告] 資金不足無法買入 | 日期: " << sig.date 
                              << " | 現金: " << cash 
                              << " | 股價: " << sig.price << std::endl;
                    break; 
                }    

                // 執行買入
                stock_inventory.push_back(sig.price); // 紀錄成本
                cash -= sig.price;
                result.remaining_stock += 1;
                
                // 注意：買入時我們先不增加 TotalTrades，等賣出才算完成一筆交易
                if (require_print == requirePrint::YES){
                    std::cout << "[買入] " << sig.date << " @ " << sig.price << std::endl;
                }
                
                break;
            }
            case SignalType::SELL: {
                // 檢查是否有庫存可賣
                if (stock_inventory.empty()) {
                    if (require_print == requirePrint::YES){
                        std::cout << "[忽略] 賣出訊號忽略 (無庫存) | 日期: " << sig.date << std::endl;
                    }
                    break;
                }    

                // 執行賣出 (先進先出 FIFO)
                double buy_cost = stock_inventory.front();
                stock_inventory.pop_front();
                
                cash += sig.price;
                double profit = sig.price - buy_cost;
                result.remaining_stock -= 1;
                TotalExpenditure += buy_cost;
                TotalEarned += profit;
            

                // 判斷勝負
                if (profit > 0) {
                    win_trades++;
                }

                // 完成一次完整交易 (一買一賣)，次數 +1
                result.TotalTrades++;

                if (require_print == requirePrint::YES){
                    std::cout << "[賣出] " << sig.date << " @ " << sig.price 
                            << " | 成本: " << buy_cost 
                            << " | 損益: " << profit << std::endl;
                }
                break;
            }
            case SignalType::HOLD:
                // 什麼都不做
                break;
        }
    }

    if (require_print == requirePrint::YES){
        std::cout << "========== 回測結束 ==========" << std::endl;
    }

    // 計算總資產 (現金 + 剩餘股票市值) - 初始本金
    double remaining_stock_value = stock_inventory.size() * curr_stock_price;
    result.TotalProfit = (cash + remaining_stock_value) - initial_money;

    // 計算勝率 (防止除以 0 當機)
    if (result.TotalTrades > 0) {
        result.WinRate = (double)win_trades / result.TotalTrades * 100.0; // 乘 100 變百分比
    } else {
        result.WinRate = 0.0;
    }

    if (TotalExpenditure != 0){
        result.ROI = TotalEarned / TotalExpenditure * 100;
    }

    return result;
}
