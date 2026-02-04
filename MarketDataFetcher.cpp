#include "MarketDataFetcher.h"
#include <curl/curl.h>
#include <iostream>
#include <nlohmann/json.hpp> // 🔥 新武器：JSON 解析庫

// 為了方便，我們設定一個別名
using json = nlohmann::json;


// --- Callback 函數 (保持不變) ---
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append((char*)contents, totalSize);
    return totalSize;
}


// 1. 建構子
MarketDataFetcher::MarketDataFetcher() {
    // TODO: 初始化 curl，並把結果存到 this->curl
    // 記得要先做 curl_global_init(CURL_GLOBAL_ALL); 雖然這通常在 main 做一次就好，
    // 但為了封裝完整性，或是簡單起見，我們這裡專注於 easy_init
    
    // return Nullptr if error happens
    void* temp = curl_easy_init(); 
    
    if (temp){
        this->curl = temp;
    }
    else{
        std::cout << "Error initiailzing curl" << std::endl;
    }
}


// 2. 解構子
MarketDataFetcher::~MarketDataFetcher() {
    // TODO: 檢查 curl 是否存在，如果存在就 cleanup
    if (this->curl)
    {
        // requires CURL* type, but it is void* type here
        curl_easy_cleanup(static_cast<CURL*>(this->curl));
    }
    
}


static double extractOpeningPrice(const json& result, const json& meta){
    // 通常資料會放在 regularMarketOpen
    if (meta.contains("regularMarketOpen")){
        return meta["regularMarketOpen"].get<Price>();
    }

    // 掃描 indicator 裡面的 open  -->  找到第一個不為 Null 的資訊
    // Yahoo 很常有資料缺失的問題
    if (result.contains("indicators") && 
        result["indicators"].contains("quote") && 
        !result["indicators"]["quote"].empty() &&
        result["indicators"]["quote"][0].contains("open")){
            
            auto OpenArray = result["indicators"]["quote"][0]["open"];

            for (const auto& price: OpenArray){
                if (!price.is_null()){
                    return price.get<Price>();
                }
            }       
        }

    return -1.0; 
}


// 3. getPrice (先留個空殼或簡單實作)
Stock MarketDataFetcher::getPrice(const std::string& stockSymbol) {
    Stock ErrorReturn = Stock{stockSymbol, -1.0, -1.0, -1.0, -1.0, 0};

    // Safety check
    if (!this->curl) return ErrorReturn; // 1.0 ==> error message

    // Turn void* back to CURL*
    CURL* newCurl = static_cast<CURL*>(this->curl);

    std::string response_string;

    // 組合 URL
    std::string url = "https://query1.finance.yahoo.com/v8/finance/chart/" + stockSymbol + "?interval=1m";
    std::string userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36";

    // Set curl option
    curl_easy_setopt(newCurl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(newCurl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(newCurl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(newCurl, CURLOPT_USERAGENT, userAgent.c_str()); // set agent to access json or will be blokced

    // 3. 執行請求
    CURLcode res = curl_easy_perform(newCurl);

    if (res != CURLE_OK) {
        std::cerr << "CURL Error: " << curl_easy_strerror(res) << std::endl;
        return ErrorReturn; // -1.0 ==> error message
    }

    // 4. 解析 JSON
    try {
        auto data = json::parse(response_string);
        auto meta = data["chart"]["result"][0]["meta"];
        
        // TODO_2: 像剝洋蔥一樣取出價格
        // 路徑: chart -> result -> [0] -> meta -> regularMarketPrice
        // 提示: Price price = ...;
        Price price = meta.value("regularMarketPrice", -1.0);
        Price open = extractOpeningPrice(data["chart"]["result"][0], meta); // Open 的第一筆資料（Open 包含整天每分鐘的價格）
        Price high = meta.value("regularMarketDayHigh", -1.0);
        Price low = meta.value("regularMarketDayLow", -1.0);
        unsigned long long int volume = meta.value("regularMarketVolume", 0);

        return Stock{stockSymbol, price, open, high, low, volume};
         
    } catch (const std::exception& e) {
        std::cerr << "JSON Error: " << e.what() << std::endl;
    }
    
    return ErrorReturn;// -1.0 ==> error message 
}




