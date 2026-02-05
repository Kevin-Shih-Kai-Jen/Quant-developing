// MarketDataFetcher.cpp
#include "MarketDataFetcher.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

MarketDataFetcher::MarketDataFetcher() {
}

MarketDataFetcher::~MarketDataFetcher() {
}

std::vector<CandleStick> MarketDataFetcher::loadHistoryFromCSV(const std::string& filename) {
    std::vector<CandleStick> history;
    std::ifstream file(filename);

    if (!file.is_open()){
        std::cerr << "❌ 無法開啟檔案: " << filename << " (請確認有沒有先執行 Python 腳本)" << std::endl;
        return history;
    }

    // 讀取標題
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        CandleStick kline;
        std::string temp;

        try {
            // 1. 日期 (字串)
            if (!std::getline(ss, kline.timestamp, ',')) continue; // 如果連日期都沒有，這行直接跳過

            // 2. 價格 (Double) - 遇到空值會丟出例外，被 catch 接住
            std::getline(ss, temp, ','); kline.open = std::stod(temp);
            std::getline(ss, temp, ','); kline.high = std::stod(temp);
            std::getline(ss, temp, ','); kline.low = std::stod(temp);
            std::getline(ss, temp, ','); kline.close = std::stod(temp);

            // 3. 成交量 (UInt64)
            std::getline(ss, temp, ','); kline.volume = std::stoull(temp);
            
            // 全部成功才存進去
            history.push_back(kline);

        } catch (const std::exception& e) {
            // 這裡就是安全氣囊 💥
            // 如果轉換失敗 (例如資料是 "null" 或空字串)，程式不會當機，只會印出這行
            std::cerr << "跳過壞掉的資料行: " << line << std::endl;
        }
    }

    file.close();
    std::cout << "✅ 成功讀取 " << history.size() << " 筆 K 棒資料！" << std::endl;
    return history;
}
