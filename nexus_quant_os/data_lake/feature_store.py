"""
nexus_quant_os/data_lake/feature_store.py — 雙時態特徵庫 (Bitemporal Feature Store)

負責處理 Knowledge_Date 與 Effective_Date 的時間對齊，徹底消滅未來函數。
"""

import logging
import os
import json
import polars as pl
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from nexus_quant_os.core.models import BitemporalFinancialData

logger = logging.getLogger("FeatureStore")

DEFAULT_DATA_LAKE_DIR = str(Path(__file__).parent / "parquet")

class FeatureStore:
    def __init__(self, data_lake_dir: str = DEFAULT_DATA_LAKE_DIR):
        self.data_lake_dir = data_lake_dir
        self.db_path = Path(self.data_lake_dir) / "financial_db.json"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._financial_db: list[BitemporalFinancialData] = self._load_db()
        
    def _load_db(self) -> list[BitemporalFinancialData]:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                return [BitemporalFinancialData(**item) for item in data]
            except Exception as e:
                logger.error(f"Failed to load financial DB from JSON: {e}")
        return []

    def _save_db(self):
        try:
            with open(self.db_path, "w") as f:
                json.dump([item.model_dump() for item in self._financial_db], f)
        except Exception as e:
            logger.error(f"Failed to save financial DB to JSON: {e}")

    def add_financial_record(self, record: BitemporalFinancialData):
        self._financial_db.append(record)
        self._save_db()
        
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
