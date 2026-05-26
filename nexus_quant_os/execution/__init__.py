"""
Trade execution & broker interface.
====================================

Public API
----------
- ``BrokerBase``       — Abstract broker adapter interface.
- ``SimulatedBroker``  — Paper-trading broker backed by SQLite.
- ``FutuBroker``       — Moomoo/Futu paper-trading broker (SIMULATE only).
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

# Lazy import: FutuBroker requires moomoo-api which may not be installed
try:
    from nexus_quant_os.execution.futu_broker import FutuBroker
except ImportError:
    FutuBroker = None  # type: ignore[assignment,misc]

__all__ = [
    # ABC
    "BrokerBase",
    # Concrete brokers
    "SimulatedBroker",
    "FutuBroker",
    # Logger
    "TradeLogger",
    # Data classes
    "AccountSnapshot",
    "Position",
    "OrderIntent",
    "OrderResult",
]
