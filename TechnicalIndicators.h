#ifndef TECHNICAL_INDICATORS_H
#define TECHNICAL_INDICATORS_H

#include <vector>
#include <numeric> // 這一行一定要加，因為 std::accumulate 在這裡面
#include <ranges>


class TechnicalIndicators {
public:
    // 加上 static，讓它變成靜態工具函式
    template<typename T>
    static std::vector<double> CalculateSMA(int period, const std::vector<T>& numbers) {
        double window_sum;
        std::vector<double> SMA;

        // 1. 檢查資料長度
        if (numbers.size() < period) {
            return SMA;
        }
        
        // 2. 初始視窗總和 (First Window)
        window_sum = std::accumulate(numbers.begin(), numbers.begin() + period, 0.0);
        SMA.push_back(window_sum / period);
        
        // 3. 滑動視窗 (Sliding Window)
        for (size_t i = period; i < numbers.size(); i++) {
            // 修正了這裡的 typo，並改用 [] 存取運算子
            window_sum = window_sum - numbers[i - period] + numbers[i];

            SMA.push_back(window_sum / period);
        }
        
        return SMA;
    }



    //======================================= RSI =======================================//
    template <typename T>
    static std::vector<double> CalculateRSI(int period, const std::vector<T>& numbers){
        std::vector<double> RSIVector;
        
        // period + 1 ==> Momentum: N gaps, N + 1 numbers
        if (numbers.size() < period + 1){
            return RSIVector;
        }

        //================= initial RSI =========================//
        double avg_gain = 0.0;
        double avg_loss = 0.0;
        double RSI = 0.0;

        // Get the initial RSI
        for (int i = 1; i < period + 1; i++){
            double diff = numbers[i] - numbers[i - 1];
            double curr_gain = (diff > 0) ?  diff : 0.0;
            double curr_loss = (diff < 0) ? -diff : 0.0;

            avg_gain += curr_gain;
            avg_loss += curr_loss;
        }
        avg_gain = avg_gain / period;
        avg_loss = avg_loss / period;

        // Avoid 0 division issue
        if (avg_gain + avg_loss == 0){
            RSI = 0;
        }else{
            RSI = 100 * avg_gain / (avg_gain + avg_loss);
        }

        RSIVector.push_back(RSI);

        //================= changing RSI =========================//
        for (int i = period + 1; i < numbers.size(); i++){
            double diff = numbers[i] - numbers[i - 1];
            double curr_gain = (diff > 0) ?  diff : 0.0;
            double curr_loss = (diff < 0) ? -diff : 0.0;

            avg_gain = (avg_gain * (period - 1) + curr_gain) / period;
            avg_loss = (avg_loss * (period - 1) + curr_loss) / period;
            
            // Avoid 0 division issue
            if (avg_gain + avg_loss == 0){
                RSI = 0;
            }else{
                RSI = 100 * avg_gain / (avg_gain + avg_loss);
            }

            RSIVector.push_back(RSI);
        }
        
        return RSIVector;
    }


    //============================== EMA ==============================//
    // 計算 EMA (指數移動平均)
    // Input: 週期 (period), 收盤價 (prices)
    static std::vector<double> CalculateEMA(int period, const std::vector<double>& prices) {
        std::vector<double> ema_values;
        
        // 1. 檢查資料長度 (至少要能算出第一個 SMA 當種子)
        if (prices.size() < period) return ema_values;

        // 2. 計算權重因子 k (multiplier)
        double k = 2.0 / (1 + period);
        
        // 3. 計算初始值 (用前 N 天的 SMA 當作第一個 EMA)
        // TODO: 寫一個小迴圈算 sum，然後 push_back(sum / period)
        double sum = 0.0;
        for (int i = 0; i < period; i++){
            sum += prices[i];
        }
        sum /= period;
        ema_values.push_back(sum);

    
        // 4. 遞迴計算後續的 EMA
        // TODO: 套用公式：(Price * k) + (PrevEMA * (1 - k))
        for (int i = period; i < prices.size(); i ++){
            double sma = prices[i] * k + ema_values[ema_values.size() - 1] * (1 - k);
            ema_values.push_back(sma);
        }
        
        return ema_values;
    }


    // 計算 MACD
    // 標準參數: short_p=12, long_p=26, signal_p=9
    static MACDResult CalculateMACD(const std::vector<double>& prices, int short_p = 12, int long_p = 26, int signal_p = 9) {
        MACDResult result;

        // 1. 計算兩條 EMA
        // ema_short (12) 長度會比較長
        // ema_long (26) 長度會比較短 <--- 這是我們的限制因素
        std::vector<double> ema_short = CalculateEMA(short_p, prices);
        std::vector<double> ema_long = CalculateEMA(long_p, prices);

        // 基本檢查
        if (ema_long.empty()) return result;

        // 2. 計算 DIF (快線) = EMA_short - EMA_long
        // 關鍵：對齊問題
        // ema_short 的第 0 格是對應第 12 天
        // ema_long  的第 0 格是對應第 26 天
        // 所以 ema_short 必須「跳過」前 (26 - 12) = 14 格，才能跟 ema_long 的第 0 格對齊
        int offset = long_p - short_p;

        for (size_t i = 0; i < ema_long.size(); ++i) {
            // ema_short[i + offset] 對應 ema_long[i]
            double val = ema_short[i + offset] - ema_long[i];
            result.dif.push_back(val);
        }

        // 3. 計算 DEA (慢線)
        // DEA 其實就是「DIF 的 EMA」
        // 我們直接把剛剛算出來的 dif 丟進去 CalculateEMA 就好了！
        result.dea = CalculateEMA(signal_p, result.dif);

        // 4. 計算柱狀圖 (Histogram) = DIF - DEA
        // 這裡又會有一次對齊問題：
        // DIF 有很多筆，但 DEA (因為又是 EMA) 會前幾筆算不出來
        // 所以柱狀圖的長度會被 DEA 限制住
        int dea_offset = result.dif.size() - result.dea.size();

        for (size_t i = 0; i < result.dea.size(); ++i) {
            // 用對齊後的 DIF 減去 DEA
            double val = result.dif[i + dea_offset] - result.dea[i];
            result.histogram.push_back(val);
        }

        return result;
    }
};

#endif