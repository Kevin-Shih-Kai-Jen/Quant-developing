"""
nexus_quant_os/core/event_bus.py — 核心事件總線 (The Abyss Message Bus)

所有的模組間通訊 (如 Signal -> Portfolio -> Execution) 強制透過此 Pub/Sub 模式。
完全解耦，確保改 A 不壞 B。
"""

import logging
from typing import Callable, Dict, List, Any
from pydantic import BaseModel

logger = logging.getLogger("EventBus")

class EventBus:
    _subscribers: Dict[str, List[Callable[[BaseModel], None]]] = {}

    @classmethod
    def subscribe(cls, event_type: str, callback: Callable[[BaseModel], None]):
        """註冊監聽器"""
        if event_type not in cls._subscribers:
            cls._subscribers[event_type] = []
        cls._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed {callback.__name__} to {event_type}")

    @classmethod
    def publish(cls, event_type: str, event: BaseModel):
        """發佈事件 (Fire and Forget)"""
        if event_type not in cls._subscribers:
            logger.debug(f"No subscribers for event: {event_type}")
            return
            
        for callback in cls._subscribers[event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in subscriber {callback.__name__} for {event_type}: {e}", exc_info=True)

    @classmethod
    def clear(cls):
        """測試用：清空所有訂閱"""
        cls._subscribers.clear()
