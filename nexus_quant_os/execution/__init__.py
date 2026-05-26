"""
Trade execution & broker interface.
====================================

Public API
----------
- ``BrokerBase``       — Abstract broker adapter interface.
- ``SimulatedBroker``  — Paper-trading broker backed by SQLite.
- ``TradeLogger``      — Structured JSON trade & performance logger.

Data classes (from ``broker_base``)::

    AccountSnapshot, Position, OrderIntent, OrderResult
"""

from nexus_quant_os.execution.broker_base import (
    AccountSnapshot,
    BrokerBase,
    OrderIntent,
    OrderResult,
    Position,
)
from nexus_quant_os.execution.broker_router import SimulatedBroker
from nexus_quant_os.execution.trade_logger import TradeLogger

__all__ = [
    # ABC
    "BrokerBase",
    # Concrete broker
    "SimulatedBroker",
    # Logger
    "TradeLogger",
    # Data classes
    "AccountSnapshot",
    "Position",
    "OrderIntent",
    "OrderResult",
]
