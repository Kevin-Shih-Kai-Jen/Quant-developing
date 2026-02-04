#include <iostream>
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 簡單的 WriteCallback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    CURL* curl = curl_easy_init();
    std::string readBuffer;

    // 設定網址 (Yahoo Finance API)
    curl_easy_setopt(curl, CURLOPT_URL, "https://query1.finance.yahoo.com/v8/finance/chart/^TWII?interval=1m");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    // 記得要偽裝 User-Agent
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    try {
        // 1. 解析 JSON
        auto data = json::parse(readBuffer);

        // 2. 🔥 關鍵：用 dump(4) 印出漂亮的結構 (4 格的 indentation)
        std::cout << data.dump(4) << std::endl;

    } catch (json::exception& e) {
        std::cerr << "JSON Error: " << e.what() << std::endl;
    }

    return 0;
}

// Compile message
// g++ -std=c++17 -I/opt/homebrew/include -I/usr/local/include JsonProbe.cpp -o probe -lcurl
//  ./probe > log.txt