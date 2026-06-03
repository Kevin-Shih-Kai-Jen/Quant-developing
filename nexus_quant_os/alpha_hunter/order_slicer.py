"""
alpha_hunter/order_slicer.py — 智慧切單機與執行引擎 (Smart Order Slicer)

防禦 Edge Case #30: 單筆委託上限 499 張（499,000 股）
防禦 Edge Case #15: 一股 vs 一張（1000股）的度量衡陷阱

功能升級：
    1. 無狀態架構 (Stateless Recovery)
    2. 滑價保護 (Slippage Threshold)
    3. 雙軌速率限制 (Dual-Track Rate Limit)
    4. 流動性阻塞等待與 Timeout 熔斷 (Liquidity Wait & Timeout)
"""

from __future__ import annotations

import logging
import math
import time
import uuid
import dataclasses
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── 交易所限制常數 ──────────────────────────────────────
TW_MAX_LOTS_PER_ORDER: int = 499
TW_LOT_SIZE: int = 1000

# 美股同樣套用 499,000 股防護上限（防止極端市價單衝擊市場）
US_MAX_SHARES_PER_ORDER: int = 499000

try:
    from moomoo import (
        RET_OK,
        OrderType,
        ModifyOrderOp,
    )
except ImportError:
    pass

class SmartOrderSlicer:
    """智能切單機 — 結合拆單計算與執行防禦邏輯。"""

    @staticmethod
    def shares_to_lots(shares: int) -> int:
        if shares <= 0:
            raise ValueError(f"股數必須為正整數，收到: {shares}。")
        return math.ceil(shares / TW_LOT_SIZE)

    @staticmethod
    def lots_to_shares(lots: int) -> int:
        if lots <= 0:
            raise ValueError(f"張數必須為正整數，收到: {lots}。")
        return lots * TW_LOT_SIZE

    @staticmethod
    def slice_order(shares: int, market: str) -> list[int]:
        """將委託單依市場規則拆分為多筆子單。"""
        if shares <= 0:
            raise ValueError(f"委託股數必須為正整數，收到: {shares}。")

        if market.upper() == "TW":
            max_shares_per_order = TW_MAX_LOTS_PER_ORDER * TW_LOT_SIZE
        else:
            max_shares_per_order = US_MAX_SHARES_PER_ORDER

        if shares <= max_shares_per_order:
            return [shares]

        slices: list[int] = []
        remaining = shares
        while remaining > 0:
            chunk = min(remaining, max_shares_per_order)
            slices.append(chunk)
            remaining -= chunk

        total_sliced = sum(slices)
        if total_sliced != shares:
            raise RuntimeError(f"切單結果不一致！原始: {shares}, 合計: {total_sliced}")

        logger.info(
            "[%s] 切單完成：%d 股拆分為 %d 筆子單 %s",
            market, shares, len(slices), slices,
        )
        return slices

    @staticmethod
    def execute_sliced_order(
        broker,
        intent,
        market: str = "US",
        slippage_threshold: float = 0.005,
        timeout_seconds: int = 60,
    ) -> list[Any]:
        """
        執行拆單邏輯並與 Broker 互動。
        
        防禦機制：
        1. 重新檢查價格 (Slippage Threshold)
        2. 失敗重試 (Dual-track rate limit)
        3. 等待成交 (Wait for fill)
        """
        from nexus_quant_os.execution.broker_base import OrderResult
        
        slices = SmartOrderSlicer.slice_order(int(intent.qty), market)
        results = []
        
        ctx = broker._get_trade_ctx()
        
        # 1. 取得基準價格 (基準點)
        initial_prices = broker._get_prices([intent.symbol])
        base_price = initial_prices.get(intent.symbol)
        
        # 防禦 P0: 嚴格過濾 0 或極小值，防止 ZeroDivisionError
        if base_price is None or base_price <= 1e-5:
            logger.warning("REJECTED %s %s — 無法取得初始報價", intent.side, intent.symbol)
            return [OrderResult(
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                filled_price=0.0,
                commission=0.0,
                timestamp=datetime.now(timezone.utc),
                order_id="NO_PRICE",
                status="REJECTED",
            )]

        logger.info("開始執行 %s %s, 總量 %d 股, 分 %d 筆, 基準價格 $%.2f", 
                    intent.side, intent.symbol, intent.qty, len(slices), base_price)

        futu_code = broker._to_futu_code(intent.symbol)
        from moomoo import TrdSide
        side = TrdSide.BUY if intent.side == "BUY" else TrdSide.SELL

        for i, chunk_qty in enumerate(slices):
            logger.info("--- 準備遞交第 %d/%d 筆子單 (%d 股) ---", i + 1, len(slices), chunk_qty)
            
            # --- 🛡️ 防禦: Slippage Check ---
            current_prices = broker._get_prices([intent.symbol])
            current_price = current_prices.get(intent.symbol)
            
            # 防禦 P0: 嚴格過濾 0
            if current_price is None or current_price <= 1e-5:
                logger.error("無法取得即時價格，暫停後續拆單")
                break
                
            price_diff = abs(current_price - base_price) / base_price
            if price_diff > slippage_threshold:
                logger.error("🚨 觸發滑價保護! 當前價格 $%.2f 偏離基準 $%.2f 達 %.2f%% > %.2f%%. 中止後續 %d 筆拆單.",
                             current_price, base_price, price_diff * 100, slippage_threshold * 100, len(slices) - i)
                break

            safe_price = round(float(current_price), 2)

            # --- 🛡️ 防禦: Dual-Track Rate Limit ---
            # 防禦 P1: 指數退避 (Exponential Backoff) 撐過 30 秒 API 冷卻期
            max_retries = 5
            order_id = ""
            order_error = ""
            
            # 🛡️ 裝甲：冪等性 UUID (Idempotency)
            # 將它塞入 remark，即使網路斷線重發，券商也會視為同一筆訂單拒絕重複執行
            today_str = datetime.now(timezone.utc).strftime('%Y%m%d')
            idem_id = f"NXOS_{today_str}_{intent.symbol}_{intent.side}_{chunk_qty}_{uuid.uuid4().hex[:4]}"
            
            for attempt in range(max_retries):
                ret, data = ctx.place_order(
                    price=safe_price,
                    qty=chunk_qty,
                    code=futu_code,
                    trd_side=side,
                    order_type=OrderType.NORMAL,
                    trd_env=broker._TRD_ENV,
                    remark=idem_id  # <--- 寫入券商備註欄
                )
                
                if ret == RET_OK:
                    order_id = str(data.iloc[0].get("order_id", ""))
                    logger.info("子單 %d 遞交成功，order_id=%s", i + 1, order_id)
                    break
                else:
                    order_error = str(data)
                    logger.warning("API 拒絕 (Attempt %d/%d): %s", attempt + 1, max_retries, order_error)
                    # 指數退避：2s, 4s, 8s, 16s, 32s (足以覆蓋 30s 鎖定)
                    time.sleep(2 ** (attempt + 1))  
                    
            if not order_id:
                logger.error("子單 %d 重試 %d 次皆失敗，中止拆單流程。", i + 1, max_retries)
                results.append(OrderResult(
                    symbol=intent.symbol, side=intent.side, qty=chunk_qty, filled_price=0.0,
                    commission=0.0, timestamp=datetime.now(timezone.utc),
                    order_id=f"RETRY_FAILED_{order_error[:20]}", status="REJECTED"
                ))
                break

            # 記錄已送出
            results.append(OrderResult(
                symbol=intent.symbol, side=intent.side, qty=float(chunk_qty), filled_price=0.0,
                commission=0.0, timestamp=datetime.now(timezone.utc),
                order_id=order_id, status="SUBMITTED"
            ))

            # --- 🛡️ 防禦: 等待成交 (Wait for fill) ---
            filled = False
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                ret, order_data = ctx.order_list_query(order_id=order_id, trd_env=broker._TRD_ENV)
                if ret == RET_OK and not order_data.empty:
                    status = str(order_data.iloc[0].get("order_status", ""))
                    if status == "FILLED":
                        filled = True
                        # 更新 results 裡面的狀態
                        results[-1] = dataclasses.replace(
                            results[-1], 
                            status="FILLED",
                            filled_price=float(order_data.iloc[0].get("dealt_avg_price", safe_price))
                        )
                        logger.info("子單 %d 已完全成交! (FILLED)", i + 1)
                        break
                    elif status in ("CANCELLED", "REJECTED", "CANCEL_ALL", "FAILED"):
                        logger.error("子單 %d 狀態異常 (%s)，中止後續拆單。", i + 1, status)
                        results[-1] = dataclasses.replace(results[-1], status=status)
                        break
                time.sleep(0.5)  # Polling interval
                
            if not filled and results[-1].status == "SUBMITTED":
                logger.error("🚨 子單 %d 逾時未成交 (Timeout %ds)，自動取消並中止拆單。", i + 1, timeout_seconds)
                ret, data = ctx.modify_order(
                    modify_order_op=ModifyOrderOp.CANCEL,
                    order_id=order_id, qty=0, price=0, trd_env=broker._TRD_ENV
                )
                # 防禦 P0: Phantom Fills，必須確認 Cancel 真的成功了
                if ret == RET_OK:
                    results[-1] = dataclasses.replace(results[-1], status="CANCELLED")
                else:
                    logger.critical("💀 災難: 取消訂單失敗! 訂單 %s 可能變成幽靈倉位 (Phantom Fill): %s", order_id, data)
                    results[-1] = dataclasses.replace(results[-1], status="UNKNOWN_OR_PHANTOM")
                break
                
        ctx.close()
        
        # 把尚未執行的子單記錄為 CANCELLED
        executed_qty = sum(r.qty for r in results)
        remaining_qty = intent.qty - executed_qty
        if remaining_qty > 0:
            results.append(OrderResult(
                symbol=intent.symbol, side=intent.side, qty=remaining_qty, filled_price=0.0,
                commission=0.0, timestamp=datetime.now(timezone.utc),
                order_id="ABORTED", status="CANCELLED"
            ))
            
        return results
