from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("nexus_quant_os.plugins")

@dataclass
class PluginOutput:
    """外掛標準化輸出格式 (JSON IPC 對應資料結構)。"""
    data: pd.DataFrame
    timestamp: pd.Timestamp
    lookahead_safe: bool
    data_granularity: str  # "daily", "monthly", "tick" 等


class BasePlugin(ABC):
    """Nexus Quant OS 外掛基類。
    
    所有外部資料源（總經、情緒、另類數據）都必須繼承此類，
    並實作 fetch() 與 validate_pit() 方法，確保符合單向數據流與無前瞻偏誤設計。
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fetch(self, start: str, end: str) -> PluginOutput:
        """獲取指定時間範圍內的資料。
        
        必須回傳標準化的 PluginOutput，其中 data DataFrame 必須包含 'timestamp' 欄位。
        """
        pass

    def validate_pit(self, output: PluginOutput) -> bool:
        """驗證資料是否符合 PiT (Point-in-Time) 原則。
        
        1. 確認 lookahead_safe 為 True
        2. 確認 timestamp 沒有來自未來的時間
        """
        if not output.lookahead_safe:
            logger.error(f"[{self.name}] PIT 驗證失敗：lookahead_safe 標記為 False")
            return False
            
        now = pd.Timestamp.now(tz=output.timestamp.tz)
        if output.timestamp > now:
            logger.error(f"[{self.name}] PIT 驗證失敗：資料時間戳 ({output.timestamp}) 來自未來")
            return False
            
        return True
