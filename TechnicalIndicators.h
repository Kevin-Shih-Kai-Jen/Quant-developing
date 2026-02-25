#ifndef TECHNICAL_INDICATORS_H
#define TECHNICAL_INDICATORS_H

#include <numeric> // 這一行一定要加，因為 std::accumulate 在這裡面
#include <limits>
#include <cmath>
#include <ranges>
#include <deque>
#include <algorithm>

#include "DataStructures.h"

class TechnicalIndicators {
public:
    // 加上 static，讓它變成靜態工具函式
template<typename T>
static std::vector<double> CalculateSMA(int period, const std::vector<T>& data, int start_index, int end_index) {
    // 1. 基本防呆機制
    // 若週期無效、資料為空、起始點小於 0，或起始大於結束，直接回傳空陣列
    if (period <= 0 || data.empty() || start_index < 0 || end_index < start_index || start_index >= data.size()) {
        return {}; 
    }

    // 確保 end_index 不會超出 data 的實際邊界
    end_index = std::min(end_index, static_cast<int>(data.size() - 1));

    // 2. 找出真正能開始計算 SMA 的起點
    // 要算 SMA，當下 index 必須至少要有 period 個資料 (即 index >= period - 1)
    int actual_calc_start = std::max(start_index, period - 1);

    // 3. 初始化結果陣列 (對齊資料的關鍵)
    // 建立大小為 end_index + 1 的陣列，預設全部填滿 NaN
    // 這樣 SMA[i] 的 index 會跟 data[i] 完美對齊
    std::vector<double> SMA(end_index + 1, std::numeric_limits<double>::quiet_NaN());

    // 如果指定的區間完全無法算出任何 SMA (例如資料不夠長)，直接回傳滿是 NaN 的陣列
    if (actual_calc_start > end_index) {
        return SMA;
    }

    // 4. 計算第一個有效視窗的總和
    double window_sum = 0.0;
    // 往前推算 period 個元素，累加到 actual_calc_start
    for (int i = actual_calc_start - period + 1; i <= actual_calc_start; ++i) {
        window_sum += static_cast<double>(data[i]); // 轉型 double 避免 T 為整數時溢位或捨去
    }
    
    // 紀錄第一筆 SMA 數值
    SMA[actual_calc_start] = window_sum / period;

    // 5. 滑動視窗 (Sliding Window) 處理剩下的範圍
    for (int i = actual_calc_start + 1; i <= end_index; ++i) {
        // 減去滑出視窗的舊資料 (i - period)，加上剛進視窗的新資料 (i)
        window_sum = window_sum - static_cast<double>(data[i - period]) + static_cast<double>(data[i]);
        SMA[i] = window_sum / period;
    }

    return SMA;
}


    //======================================= RSI =======================================//
    template <typename T>
    static std::vector<double> CalculateRSI(int period, const std::vector<T>& numbers, int start_index, int end_index) {
        // 1. 防呆機制：檢查週期是否有效、資料是否夠長
        if (period <= 0 || numbers.empty() || start_index < 0 || end_index < start_index || start_index + period >= numbers.size()) {
            return {};
        }

        // 確保 end_index 不會超出邊界
        end_index = std::min(end_index, static_cast<int>(numbers.size() - 1));

        // 2. 對齊資料：配置好大小，預設填滿 NaN
        std::vector<double> RSIVector(end_index + 1, std::numeric_limits<double>::quiet_NaN());

        // 第一個可以算出 RSI 的索引位置 (需要 period 根 K 線的價差，所以是 start + period)
        int first_rsi_index = start_index + period;
        if (first_rsi_index > end_index) {
            return RSIVector; // 區間太短，算不出第一根 RSI
        }

        //================= Initial RSI (簡單平均) =========================//
        double sum_gain = 0.0;
        double sum_loss = 0.0;

        // 收集從 start_index 到 first_rsi_index 的 period 個價差
        for (int i = start_index + 1; i <= first_rsi_index; i++) {
            // 轉型 double 避免無號數相減溢位
            double diff = static_cast<double>(numbers[i]) - static_cast<double>(numbers[i - 1]); 
            if (diff > 0) {
                sum_gain += diff;
            } else {
                sum_loss -= diff; // 損失取正數
            }
        }

        double avg_gain = sum_gain / period;
        double avg_loss = sum_loss / period;

        // 避免 0 除問題 (如果完全沒漲跌，RSI 設為 50 代表中立)
        if (avg_gain + avg_loss == 0) {
            RSIVector[first_rsi_index] = 50.0;
        } else {
            RSIVector[first_rsi_index] = 100.0 * avg_gain / (avg_gain + avg_loss);
        }

        //================= Changing RSI (平滑移動平均) =========================//
        // 注意：這裡從 first_rsi_index + 1 開始，接續剛剛算完的地方
        for (int i = first_rsi_index + 1; i <= end_index; i++) {
            double diff = static_cast<double>(numbers[i]) - static_cast<double>(numbers[i - 1]);
            double curr_gain = (diff > 0) ? diff : 0.0;
            double curr_loss = (diff < 0) ? -diff : 0.0;

            // Wilder's Smoothing
            avg_gain = (avg_gain * (period - 1) + curr_gain) / period;
            avg_loss = (avg_loss * (period - 1) + curr_loss) / period;
            
            if (avg_gain + avg_loss == 0) {
                RSIVector[i] = 50.0; 
            } else {
                RSIVector[i] = 100.0 * avg_gain / (avg_gain + avg_loss);
            }
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


    //============================== MACD ==============================//
    static MACDResult CalculateMACD(int short_p, int long_p, int signal_p, const std::vector<double>& data, int start_index, int end_index) {
        MACDResult result;
        if (data.empty() || start_index < 0 || end_index < start_index || long_p <= short_p) return result;
        end_index = std::min(end_index, static_cast<int>(data.size() - 1));

        // 1. 預先分配並填滿 NaN
        result.dif.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());
        result.dea.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());
        result.histogram.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());

        // DIF 要到 long_p 才有足夠資料算出第一筆
        int actual_dif_start = std::max(start_index, long_p - 1);
        if (actual_dif_start > end_index) return result;

        // 2. 計算兩條對齊好的 EMA
        std::vector<double> ema_short = CalculateEMA(short_p, data);
        std::vector<double> ema_long = CalculateEMA(long_p, data);

        // 3. 計算 DIF
        for (int i = actual_dif_start; i <= end_index; ++i) {
            result.dif[i] = ema_short[i] - ema_long[i];
        }

        // 4. 計算 DEA (DIF 的 EMA) 與 Histogram
        // 因為 DIF 前面有 NaN，我們不能直接呼叫 CalculateEMA，必須手動算第一筆 DEA
        int actual_dea_start = actual_dif_start + signal_p - 1;
        if (actual_dea_start <= end_index) {
            double sum = 0.0;
            for (int i = actual_dea_start - signal_p + 1; i <= actual_dea_start; ++i) {
                sum += result.dif[i];
            }
            double current_dea = sum / signal_p;
            result.dea[actual_dea_start] = current_dea;
            result.histogram[actual_dea_start] = result.dif[actual_dea_start] - current_dea;

            double k = 2.0 / (signal_p + 1.0);
            for (int i = actual_dea_start + 1; i <= end_index; ++i) {
                current_dea = (result.dif[i] * k) + (current_dea * (1.0 - k));
                result.dea[i] = current_dea;
                result.histogram[i] = result.dif[i] - current_dea;
            }
        }
        return result;
    }


    //========================= KDJ =========================//
    static KdjResult CalculateKDJ(const std::vector<double>& data, int RSV_N, int K_N, int D_N, int start_index, int end_index) {
        KdjResult result;
        if (data.empty() || start_index < 0 || end_index < start_index || RSV_N <= 0) return result;
        end_index = std::min(end_index, static_cast<int>(data.size() - 1));

        result.kValues.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());
        result.dValues.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());
        result.jValues.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());

        int actual_start = std::max(start_index, RSV_N - 1);
        if (actual_start > end_index) return result;

        std::deque<int> max_dq, min_dq;
        
        // 預熱 Sliding Window (填補 actual_start 前的資料)
        for (int i = actual_start - RSV_N + 1; i < actual_start; ++i) {
            while (!max_dq.empty() && data[i] >= data[max_dq.back()]) max_dq.pop_back();
            max_dq.push_back(i);
            while (!min_dq.empty() && data[i] <= data[min_dq.back()]) min_dq.pop_back();
            min_dq.push_back(i);
        }

        double prev_k = 50.0;
        double prev_d = 50.0;

        for (int i = actual_start; i <= end_index; ++i) {
            // 更新 Deque
            while (!max_dq.empty() && data[i] >= data[max_dq.back()]) max_dq.pop_back();
            max_dq.push_back(i);
            while (!min_dq.empty() && data[i] <= data[min_dq.back()]) min_dq.pop_back();
            min_dq.push_back(i);

            // 移除過期索引
            if (max_dq.front() <= i - RSV_N) max_dq.pop_front();
            if (min_dq.front() <= i - RSV_N) min_dq.pop_front();

            double max_val = data[max_dq.front()];
            double min_val = data[min_dq.front()];
            double RSV = 50.0;
            
            if (max_val - min_val != 0) {
                RSV = (data[i] - min_val) / (max_val - min_val) * 100.0;
            }
            
            double k_val = ((K_N - 1) / (double)K_N) * prev_k + (1.0 / K_N) * RSV;
            double d_val = ((D_N - 1) / (double)D_N) * prev_d + (1.0 / D_N) * k_val;
            double j_val = 3.0 * k_val - 2.0 * d_val; // 簡化公式: K + 2(K - D) = 3K - 2D

            result.kValues[i] = k_val;
            result.dValues[i] = d_val;
            result.jValues[i] = j_val;

            prev_k = k_val;
            prev_d = d_val;
        }
        return result;
    }


    //========================= BBand =========================//
    static BBandResult CalculateBBand(const std::vector<double>& data, int period, int start_index, int end_index) {
        BBandResult result;
        if (period <= 0 || data.empty() || start_index < 0 || end_index < start_index) return result;
        end_index = std::min(end_index, static_cast<int>(data.size() - 1));

        result.upperBand.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());
        result.middleBand.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());
        result.lowerBand.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());
        result.standardDeviation.assign(end_index + 1, std::numeric_limits<double>::quiet_NaN());

        int actual_start = std::max(start_index, period - 1);
        if (actual_start > end_index) return result;

        double window_sum = 0.0;
        for (int i = actual_start - period + 1; i <= actual_start; ++i) {
            window_sum += data[i];
        }

        for (int i = actual_start; i <= end_index; ++i) {
            if (i > actual_start) {
                window_sum = window_sum - data[i - period] + data[i];
            }
            double middleBand = window_sum / period;

            double sigmaSum = 0.0;
            for (int j = 0; j < period; ++j) {
                double x = data[i - period + 1 + j];
                sigmaSum += std::pow(x - middleBand, 2);
            }
            double standardDeviation = std::sqrt(sigmaSum / period);

            result.middleBand[i] = middleBand;
            result.upperBand[i] = middleBand + 2.0 * standardDeviation;
            result.lowerBand[i] = middleBand - 2.0 * standardDeviation;
            result.standardDeviation[i] = standardDeviation;
        }
        return result;
    }


    //========================= ATR =========================//
    // 轉化為回傳陣列，使其能與其他指標完美對齊
    static std::vector<double> CalculateATR(const std::vector<CandleStick>& data, int period, int start_index, int end_index) {
        if (period <= 0 || data.empty() || start_index < 0 || end_index < start_index) return {};
        end_index = std::min(end_index, static_cast<int>(data.size() - 1));

        std::vector<double> ATRVector(end_index + 1, std::numeric_limits<double>::quiet_NaN());

        // ATR 需要先有 period 根 K 線的 TR (而第一筆 TR 需要前一天的 Close，所以索引至少為 1)
        // 因此至少要到 index = period 才能算出第一筆完整的 ATR
        int actual_start = std::max(start_index, period);
        if (actual_start > end_index) return ATRVector;

        auto getTR = [&data](int i) {
            return std::max({
                std::abs(data[i].high - data[i].low), 
                std::abs(data[i].high - data[i - 1].close),
                std::abs(data[i].low - data[i - 1].close)
            });
        };

        double current_atr = 0.0;
        // 算出第一個 ATR (簡單平均)
        for (int i = actual_start - period + 1; i <= actual_start; ++i) {
            current_atr += getTR(i);
        }
        current_atr /= period;
        ATRVector[actual_start] = current_atr;

        // 後續平滑
        for (int i = actual_start + 1; i <= end_index; ++i) {
            current_atr = (current_atr * (period - 1) + getTR(i)) / period;
            ATRVector[i] = current_atr;
        }

        return ATRVector;
    }


    static double PercentVal(const std::vector<double>& data, int start_index, int end_index, double percent = 0.8){
        if (start_index < 0 || end_index <= start_index || end_index >= data.size()){
            return -1.0;
        }

        // Faster than normal iteration and more readable
        std::vector<double> data_copy(data.begin() + start_index, data.begin() + end_index + 1);

        // 乘以 (size - 1) 保證 target_index 絕對不會大於最大合法索引
        int target_index = std::round((data_copy.size() - 1) * percent);

        std::nth_element(
            data_copy.begin(),
            data_copy.begin() + target_index,
            data_copy.end()
        );

        return data_copy[target_index];
    }
};

    
static Crosses CrossLine(const std::vector<double>& value, const std::vector<double>& MA, int start_index, int end_index){
    Crosses res;
    if (value.empty() || MA.empty() || start_index < 0 || end_index <= start_index || end_index > value.size() || end_index > MA.size()){
        return res;
    }   

    for (int i = start_index + 1; i < end_index; i++){
        if (value[i - 1] < MA[i - 1] && value[i] >= MA[i]){
            res.golden_cross.insert(i);
        }

        if (value[i - 1] >= MA[i - 1] && value[i] < MA[i]){
            res.death_cross.insert(i);
        }
    }

    return res;
}


static std::vector<double> CalculateVWAP(const std::vector<CandleStick>& data) {
    // Approximate only due to the data constraint of yfinance
    std::vector<double> vwap;
    vwap.reserve(data.size());

    double cumulative_tp_vol = 0.0;
    long long cumulative_vol = 0;

    for (const auto& candle : data) {
        double typical_price = (candle.high + candle.low + candle.close) / 3.0;
        
        cumulative_tp_vol += typical_price * candle.volume;
        cumulative_vol += candle.volume;

        if (cumulative_vol == 0) {
            vwap.push_back(typical_price); // 防止除以 0
        } else {
            vwap.push_back(cumulative_tp_vol / cumulative_vol);
        }
    }
    return vwap;
}


#endif