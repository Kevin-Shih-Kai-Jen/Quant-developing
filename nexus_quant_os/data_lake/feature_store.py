"""
nexus_quant_os/data_lake/feature_store.py — 雙時態特徵庫 (Bitemporal Feature Store)

負責處理 Knowledge_Date 與 Effective_Date 的時間對齊，徹底消滅未來函數。
"""

import logging
import polars as pl
from typing import Optional
from pydantic import BaseModel
from nexus_quant_os.core.models import BitemporalFinancialData

logger = logging.getLogger("FeatureStore")

class FeatureStore:
    def __init__(self, data_lake_dir: str = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/data_lake/parquet"):
        self.data_lake_dir = data_lake_dir
        # Mock database for financials
        self._financial_db: list[BitemporalFinancialData] = []
        
    def add_financial_record(self, record: BitemporalFinancialData):
        self._financial_db.append(record)
        
    def get_latest_financials(self, perm_id: str, current_backtest_date: str) -> Optional[BitemporalFinancialData]:
        """
        【升級二：雙時態共生矩陣】
        嚴格過濾：Knowledge_Date 必須 <= current_backtest_date
        這樣就算 2018-Q1 的財報在 2018-08-01 被重編 (Silent Restatement)，
        如果在 2018-06-01 回測，只會拿到舊的錯誤數據！這才是真實的！
        """
        # Filter by perm_id
        records = [r for r in self._financial_db if r.perm_id == perm_id]
        
        # Filter by Knowledge_Date <= current_backtest_date
        valid_records = [r for r in records if r.knowledge_date <= current_backtest_date]
        
        if not valid_records:
            return None
            
        # 找出最新知曉的財報 (Sort by Knowledge_Date desc, then Effective_Date desc)
        valid_records.sort(key=lambda x: (x.knowledge_date, x.effective_date), reverse=True)
        
        return valid_records[0]

# 全局單例 Feature Store
feature_store = FeatureStore()
