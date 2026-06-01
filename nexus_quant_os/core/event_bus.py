"""
nexus_quant_os/core/event_bus.py — 核心事件總線 (The Abyss Message Bus)

所有的模組間通訊 (如 Signal -> Portfolio -> Execution) 強制透過此 Pub/Sub 模式。
完全解耦，確保改 A 不壞 B。
"""

import logging
from typing import Callable, Dict, List, Any, Tuple
from pydantic import BaseModel
import threading
import queue

logger = logging.getLogger("EventBus")

class EventBus:
    _subscribers: Dict[str, List[Tuple[int, Callable[[BaseModel], None]]]] = {}
    _event_queue = queue.Queue()
    _is_dispatching = False
    _lock = threading.RLock()
    MAX_DEPTH = 1000

    @classmethod
    def subscribe(cls, event_type: str, callback: Callable[[BaseModel], None], priority: int = 10):
        """註冊監聽器，支援優先級 (數字越小優先級越高)"""
        with cls._lock:
            if event_type not in cls._subscribers:
                cls._subscribers[event_type] = []
            cls._subscribers[event_type].append((priority, callback))
            # Sort by priority
            cls._subscribers[event_type].sort(key=lambda x: x[0])
            logger.debug(f"Subscribed {callback.__name__} to {event_type} with priority {priority}")

    @classmethod
    def unsubscribe(cls, event_type: str, callback: Callable[[BaseModel], None]):
        """解除監聽器 (防止記憶體洩漏)"""
        with cls._lock:
            if event_type in cls._subscribers:
                cls._subscribers[event_type] = [(p, c) for p, c in cls._subscribers[event_type] if c != callback]
                logger.debug(f"Unsubscribed {callback.__name__} from {event_type}")

    @classmethod
    def publish(cls, event_type: str, event: BaseModel):
        """發佈事件 (使用 Queue 避免遞迴過深)"""
        with cls._lock:
            if cls._event_queue.qsize() >= cls.MAX_DEPTH:
                logger.error(f"EventBus MAX_DEPTH ({cls.MAX_DEPTH}) exceeded! Dropping event: {event_type}")
                return
            
            cls._event_queue.put((event_type, event))
            
            if cls._is_dispatching:
                return # 讓正在分發的執行緒去處理
            cls._is_dispatching = True
            
        try:
            while True:
                with cls._lock:
                    if cls._event_queue.empty():
                        break
                    e_type, e_data = cls._event_queue.get_nowait()
                
                with cls._lock:
                    subs = list(cls._subscribers.get(e_type, [])) # copy to avoid modification during iteration
                
                for _, callback in subs:
                    try:
                        callback(e_data)
                    except Exception as e:
                        logger.error(f"Error in subscriber {callback.__name__} for {e_type}: {e}", exc_info=True)
        finally:
            with cls._lock:
                cls._is_dispatching = False

    @classmethod
    def clear(cls):
        """測試用：清空所有訂閱與佇列"""
        with cls._lock:
            cls._subscribers.clear()
            while not cls._event_queue.empty():
                try:
                    cls._event_queue.get_nowait()
                except queue.Empty:
                    break
