"""
alpha_hunter/order_slicer.py — 台股切單機

防禦 Edge Case #30: 單筆委託上限 499 張（499,000 股）
防禦 Edge Case #15: 一股 vs 一張（1000股）的度量衡陷阱

台股交易所規定，單筆委託最多只能下 499 張（即 499,000 股）。
超過此限制的委託必須自動拆分成多筆，否則會被交易所拒絕。

注意：
    - 台股的「一張」= 1000 股，這是台股特有的交易單位。
    - 美股沒有「張」的概念，直接以「股」為單位，且無單筆上限。
    - 此模組只做純計算，不涉及任何 I/O 或 API 呼叫。
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# ── 台股交易所限制常數 ──────────────────────────────────────
# 單筆委託最大張數（依據台灣證券交易所規則）
TW_MAX_LOTS_PER_ORDER: int = 499

# 每張包含的股數（台股「一張」= 1000 股）
# Edge Case #15: 避免「股」與「張」混淆造成的度量衡錯誤
TW_LOT_SIZE: int = 1000


class OrderSlicer:
    """台股切單機 — 將超過上限的委託自動拆分。

    設計原則：
        - 所有方法皆為 @staticmethod，無狀態、可直接呼叫。
        - 純計算邏輯，不做任何 I/O。
        - 向後相容：美股市場直接回傳原始股數，不做任何處理。

    使用範例：
        >>> OrderSlicer.slice_order(shares=600_000, market="TW")
        [499000, 101000]
        >>> OrderSlicer.slice_order(shares=100, market="US")
        [100]
    """

    @staticmethod
    def shares_to_lots(shares: int) -> int:
        """將台股股數轉換為張數（無條件進位）。

        Edge Case #15 防禦：確保股數正確轉換為張數，避免計算錯誤。

        Args:
            shares: 股數（必須為正整數）

        Returns:
            對應的張數（無條件進位，確保不遺漏零股部分）

        Raises:
            ValueError: 當 shares <= 0 時拋出

        範例：
            >>> OrderSlicer.shares_to_lots(499_000)
            499
            >>> OrderSlicer.shares_to_lots(500)
            1
        """
        if shares <= 0:
            raise ValueError(
                f"股數必須為正整數，收到: {shares}。"
                f"請確認是否混淆了「股」與「張」的單位。"
            )
        # 使用無條件進位，確保零股部分不會被截斷
        return math.ceil(shares / TW_LOT_SIZE)

    @staticmethod
    def lots_to_shares(lots: int) -> int:
        """將台股張數轉換為股數。

        Edge Case #15 防禦：提供明確的張→股轉換，避免手動乘法出錯。

        Args:
            lots: 張數（必須為正整數）

        Returns:
            對應的股數

        Raises:
            ValueError: 當 lots <= 0 時拋出

        範例：
            >>> OrderSlicer.lots_to_shares(499)
            499000
        """
        if lots <= 0:
            raise ValueError(
                f"張數必須為正整數，收到: {lots}。"
            )
        return lots * TW_LOT_SIZE

    @staticmethod
    def slice_order(shares: int, market: str) -> list[int]:
        """將委託單依市場規則拆分為多筆子單。

        Edge Case #30 防禦：台股單筆委託不得超過 499 張（499,000 股），
        超過時自動拆分為多筆委託。

        Args:
            shares: 總委託股數（必須為正整數）
            market: 市場代碼（"TW" 或 "US"）

        Returns:
            拆分後的股數列表，每個元素代表一筆子委託的股數。
            列表中所有元素的總和等於原始委託股數。

        Raises:
            ValueError: 當 shares <= 0 時拋出

        範例：
            >>> OrderSlicer.slice_order(shares=600_000, market="TW")
            [499000, 101000]
            >>> OrderSlicer.slice_order(shares=499_000, market="TW")
            [499000]
            >>> OrderSlicer.slice_order(shares=10_000, market="US")
            [10000]
        """
        # ── 前置驗證 ──
        if shares <= 0:
            raise ValueError(
                f"委託股數必須為正整數，收到: {shares}。"
                f"請確認下單邏輯是否產生了負數或零。"
            )

        # ── 美股：無單筆委託上限，直接回傳 ──
        if market.upper() != "TW":
            return [shares]

        # ── 台股切單邏輯 ──
        # 計算單筆委託最大股數（499 張 × 1000 股/張 = 499,000 股）
        max_shares_per_order = TW_MAX_LOTS_PER_ORDER * TW_LOT_SIZE

        # 若未超過上限，不需要拆分
        if shares <= max_shares_per_order:
            return [shares]

        # 拆分為多筆子單
        slices: list[int] = []
        remaining = shares
        while remaining > 0:
            # 每筆子單不超過最大上限
            chunk = min(remaining, max_shares_per_order)
            slices.append(chunk)
            remaining -= chunk

        # ── 驗證拆分結果完整性 ──
        total_sliced = sum(slices)
        if total_sliced != shares:
            # 這不應該發生，但作為防禦性檢查
            raise RuntimeError(
                f"切單結果不一致！原始: {shares}, 拆分後合計: {total_sliced}。"
                f"這是內部邏輯錯誤，請回報。"
            )

        logger.info(
            "台股切單完成：%d 股拆分為 %d 筆子單 %s",
            shares, len(slices), slices,
        )
        return slices
