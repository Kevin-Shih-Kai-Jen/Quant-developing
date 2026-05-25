import logging
import os
from typing import Optional

import pandas as pd
from fredapi import Fred

from nexus_quant_os.plugins.base_plugin import BasePlugin, PluginOutput

logger = logging.getLogger("nexus_quant_os.plugins.macro.fed_policy")


class FedPolicyPlugin(BasePlugin):
    """聯邦基金利率 (DFF) 外掛。
    
    反映美國央行貨幣政策寬鬆/緊縮狀態。
    資料源：FRED
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__("FedPolicyPlugin")
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("必須提供 FRED_API_KEY 以初始化 FedPolicyPlugin")
        self.client = Fred(api_key=self.api_key)

    def fetch(self, start: str, end: str) -> PluginOutput:
        logger.info(f"[{self.name}] 正在獲取 DFF 聯邦基金利率數據: {start} -> {end}")
        
        try:
            # DFF: Federal Funds Effective Rate
            series = self.client.get_series('DFF')
            
            series.index = pd.to_datetime(series.index)
            series = series.sort_index().dropna()
            
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            
            # 多抓一點歷史資料以確保 PiT 偏移後仍有資料
            series_filtered = series[
                (series.index >= start_ts - pd.Timedelta(days=10)) & 
                (series.index <= end_ts)
            ].copy()
            
            df = pd.DataFrame({
                'timestamp': series_filtered.index,
                'fed_funds_rate': series_filtered.values
            })
            
            # 實作 PiT 安全偏移 (1天延遲發布)
            df['timestamp'] = df['timestamp'] + pd.Timedelta(days=1)
            df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].reset_index(drop=True)
            
            output = PluginOutput(
                data=df,
                timestamp=pd.Timestamp.now(),
                lookahead_safe=True,  # 已經過手動平移 1 天
                data_granularity="daily"
            )
            
            if self.validate_pit(output):
                logger.info(f"[{self.name}] 成功獲取 {len(df)} 筆安全資料")
                return output
            else:
                raise RuntimeError(f"[{self.name}] 獲取資料未通過 PIT 驗證")
                
        except Exception as e:
            logger.error(f"[{self.name}] 獲取資料失敗: {e}")
            raise
