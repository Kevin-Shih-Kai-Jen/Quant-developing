"""
llm/sentiment_aggregator.py — Multi-Model Sentiment Aggregation Engine
========================================================================

整合多個 LLM 模型的情緒分析結果，產出統一的市場情緒特徵。

設計原則：
    1. 三模型融合 — Gemini 2.0 Flash (主) + Gemma 4 (備) + DeepSeek (可選)
    2. 優雅降級 — 依序嘗試 Gemini → Gemma → DeepSeek，至少保留一個
    3. 滾動緩衝 — 內部維護長度為 5 的 deque 計算 MA5
    4. PiT 合規 — 僅處理傳入的新聞標題，不引入未來資訊
    5. 統一特徵介面 — 回傳 [n_assets, 3] 矩陣供下游 MoE 使用

Author : Nexus Quant OS — LLM Engineering Division
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nexus_quant_os.llm.deepseek_client import DeepSeekClient
from nexus_quant_os.llm.gemini_client import GeminiClient
from nexus_quant_os.llm.gemma_client import GemmaClient

logger = logging.getLogger("nexus_quant_os.llm.sentiment_aggregator")


@dataclass
class SentimentResult:
    """多模型聚合情緒分析結果。

    Attributes
    ----------
    sentiment_daily : float
        當日加權聚合情緒分數，[-1, 1]。
    sentiment_ma5 : float
        過去 5 日移動平均，[-1, 1]。
    sentiment_dispersion : float
        模型間分歧度：max-min / 2，[0, 1]。
    sources : dict
        各模型的個別分數。
    reasons : dict
        各模型的推理說明。
    """
    sentiment_daily: float = 0.0
    sentiment_ma5: float = 0.0
    sentiment_dispersion: float = 0.0
    sources: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


class SentimentAggregator:
    """多模型情緒分析聚合器。

    整合 Gemini 2.0 Flash（主）、Gemma 4（備用）、DeepSeek（可選本地）
    三個 LLM 的情緒分析結果，透過加權平均產出統一的市場情緒指標。

    Gemini 與 Gemma 共用同一個 Google AI Studio API Key（GEMINI_API_KEY），
    DeepSeek 使用本地 Ollama（需另行啟動）。

    Usage
    -----
    >>> agg = SentimentAggregator(use_gemini=True, use_gemma=True)
    >>> result = agg.get_daily_sentiment(["NVDA surges on AI demand"])
    >>> print(result.sentiment_daily)
    0.65
    >>> features = agg.get_features_for_assets(headlines, ASSETS)
    >>> print(features.shape)  # [7, 3]
    """

    _DEFAULT_ASSETS: list[str] = [
        "AVGO", "GLD", "IWM", "NVDA", "QQQ", "SPY", "TLT",
    ]

    def __init__(
        self,
        gemini_weight: float = 0.50,
        gemma_weight: float = 0.30,
        deepseek_weight: float = 0.20,
        use_gemini: bool = True,
        use_gemma: bool = True,
        use_deepseek: bool = False,
        gemini_client: GeminiClient | None = None,
        gemma_client: GemmaClient | None = None,
        deepseek_client: DeepSeekClient | None = None,
        ma_window: int = 5,
    ) -> None:
        self.gemini_weight = gemini_weight
        self.gemma_weight = gemma_weight
        self.deepseek_weight = deepseek_weight
        self.use_gemini = use_gemini
        self.use_gemma = use_gemma
        self.use_deepseek = use_deepseek

        self._gemini: GeminiClient | None = None
        self._gemma: GemmaClient | None = None
        self._deepseek: DeepSeekClient | None = None

        if self.use_gemini:
            self._gemini = gemini_client or GeminiClient()
        if self.use_gemma:
            self._gemma = gemma_client or GemmaClient()
        if self.use_deepseek:
            self._deepseek = deepseek_client or DeepSeekClient()

        self._ma_window = ma_window
        self._history: deque[float] = deque(maxlen=ma_window)

        active_models = []
        if self.use_gemini:
            active_models.append(f"Gemini(w={self.gemini_weight:.2f})")
        if self.use_gemma:
            active_models.append(f"Gemma(w={self.gemma_weight:.2f})")
        if self.use_deepseek:
            active_models.append(f"DeepSeek(w={self.deepseek_weight:.2f})")

        if not active_models:
            logger.warning("所有 LLM 模型均未啟用！情緒分析將回傳中性分數。")
        else:
            logger.info(
                "SentimentAggregator 初始化 | models=[%s]  ma=%d",
                ", ".join(active_models), self._ma_window,
            )

    def get_daily_sentiment(self, headlines: list[str]) -> SentimentResult:
        """查詢所有可用模型並聚合情緒分數。

        Parameters
        ----------
        headlines : list[str]
            當日新聞標題（須為 PiT 合規）。

        Returns
        -------
        SentimentResult
        """
        sources: dict[str, float] = {}
        reasons: dict[str, str] = {}
        weights: dict[str, float] = {}

        # 查詢 Gemini 2.0 Flash（主模型）
        if self.use_gemini and self._gemini is not None:
            gm = self._safe_analyze(self._gemini, "gemini", headlines)
            if gm is not None:
                sources["gemini"] = gm["sentiment_score"]
                reasons["gemini"] = gm["reason"]
                weights["gemini"] = self.gemini_weight

        # 查詢 Gemma 4（備用模型，共用 API Key）
        if self.use_gemma and self._gemma is not None:
            ga = self._safe_analyze(self._gemma, "gemma", headlines)
            if ga is not None:
                sources["gemma"] = ga["sentiment_score"]
                reasons["gemma"] = ga["reason"]
                weights["gemma"] = self.gemma_weight

        # 查詢 DeepSeek（本地 Ollama，可選）
        if self.use_deepseek and self._deepseek is not None:
            ds = self._safe_analyze(self._deepseek, "deepseek", headlines)
            if ds is not None:
                sources["deepseek"] = ds["sentiment_score"]
                reasons["deepseek"] = ds["reason"]
                weights["deepseek"] = self.deepseek_weight

        # 加權平均
        sentiment_daily = self._weighted_average(sources, weights)

        # 分歧度
        sentiment_dispersion = self._compute_dispersion(sources)

        # MA5
        self._history.append(sentiment_daily)
        sentiment_ma5 = float(np.mean(list(self._history)))

        logger.info(
            "情緒分析 | daily=%.4f  ma5=%.4f  dispersion=%.4f  "
            "sources=%s",
            sentiment_daily, sentiment_ma5, sentiment_dispersion,
            {k: f"{v:.4f}" for k, v in sources.items()},
        )

        return SentimentResult(
            sentiment_daily=sentiment_daily,
            sentiment_ma5=sentiment_ma5,
            sentiment_dispersion=sentiment_dispersion,
            sources=sources,
            reasons=reasons,
        )

    def get_features_for_assets(
        self,
        headlines: list[str],
        assets: list[str] | None = None,
    ) -> np.ndarray:
        """回傳供 MoE Router 使用的情緒特徵矩陣。

        所有資產共用相同的市場層級情緒。

        Parameters
        ----------
        headlines : list[str]
        assets : list[str] or None

        Returns
        -------
        np.ndarray
            shape [n_assets, 3]: [sentiment_daily, ma5, dispersion]
        """
        if assets is None:
            assets = self._DEFAULT_ASSETS

        result = self.get_daily_sentiment(headlines)
        features = np.zeros((len(assets), 3), dtype=np.float32)
        features[:, 0] = result.sentiment_daily
        features[:, 1] = result.sentiment_ma5
        features[:, 2] = result.sentiment_dispersion

        return features

    @staticmethod
    def _safe_analyze(
        client: DeepSeekClient | GeminiClient | GemmaClient,
        name: str,
        headlines: list[str],
    ) -> dict[str, Any] | None:
        """安全地呼叫模型，捕捉所有例外。"""
        try:
            return client.analyze_sentiment(headlines)
        except Exception as e:
            logger.warning("%s 分析失敗（優雅降級）: %s", name, e)
            return None

    @staticmethod
    def _weighted_average(
        sources: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """加權平均，自動重新正規化權重。"""
        if not sources:
            return 0.0
        total_weight = sum(weights[k] for k in sources)
        if total_weight <= 0:
            return 0.0
        weighted_sum = sum(
            sources[k] * (weights[k] / total_weight) for k in sources
        )
        return float(np.clip(weighted_sum, -1.0, 1.0))

    @staticmethod
    def _compute_dispersion(sources: dict[str, float]) -> float:
        """模型間分歧度 [0, 1]。"""
        if len(sources) < 2:
            return 0.0
        scores = list(sources.values())
        return float(np.clip((max(scores) - min(scores)) / 2.0, 0.0, 1.0))
