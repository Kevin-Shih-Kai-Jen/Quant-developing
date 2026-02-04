# include <iostream>
# include "MarketDataFetcher.h"

int main(){
    MarketDataFetcher dataFetcher;
    Stock stockPriceStr = dataFetcher.getPrice("2330.TW");
    std::cout << std::endl << "===================================" << std::endl;
    std::cout << stockPriceStr.toString() << std::endl;
    std::cout << "===================================" << std::endl << std::endl;
}