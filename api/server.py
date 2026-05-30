import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
# Load .env file from the root directory so API keys are available
load_dotenv(_ROOT / ".env")

from nexus_quant_os.data_pipelines.data_loader import load_all_data
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.risk_firewall.firewall_core import (
    IntelligentRiskFirewall, HMMConfig, OODConfig, FirewallConfig
)
from nexus_quant_os.training.train_moe import (
    ASSET_UNIVERSE, DATA_START, FRED_API_KEY,
    build_daily_dataset, find_latest_checkpoint, load_checkpoint
)
from nexus_quant_os.data_pipelines.feature_engineer import engineer_features
import pandas as pd
import numpy as np
import torch

from nexus_quant_os.portfolio.weight_smoother import WeightSmoother, SmootherConfig
from nexus_quant_os.portfolio.regime_allocator import RegimeAllocator, RegimeAllocatorConfig
from nexus_quant_os.portfolio.optimizer import PortfolioOptimizer, OptimizerConfig
from nexus_quant_os.monitoring.health_check import check_weight_sanity, Severity
from nexus_quant_os.llm.sentiment_aggregator import SentimentAggregator
from nexus_quant_os.llm.news_fetcher import NewsFetcher
from nexus_quant_os.execution.broker_router import SimulatedBroker
from nexus_quant_os.execution.trade_logger import TradeLogger

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus_quant_os.api")

app = FastAPI(title="Nexus Quant OS API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup paths
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── v2.0: 全域平滑器和配置器狀態（跨請求保持） ─────────────────────
_smoother: WeightSmoother | None = None
_allocator: RegimeAllocator | None = None
_last_assets: list[str] | None = None  # 追蹤 assets 變化以重建 allocator/smoother
_optimizer: PortfolioOptimizer | None = None
_sentiment: SentimentAggregator | None = None
_news_fetcher: NewsFetcher | None = None

# ── v2.1: 模擬交易引擎（Paper Trading） ──────────────────────────────
_broker: SimulatedBroker | None = None
_trade_logger: TradeLogger | None = None

_pipeline_lock = asyncio.Lock()
_init_lock = asyncio.Lock()


async def _get_broker() -> SimulatedBroker:
    """Lazy-init SimulatedBroker singleton."""
    global _broker
    if _broker is None:
        async with _init_lock:
            if _broker is None:  # double-check
                _broker = SimulatedBroker(initial_capital=100_000.0)
    return _broker


async def _get_trade_logger() -> TradeLogger:
    """Lazy-init TradeLogger singleton."""
    global _trade_logger
    if _trade_logger is None:
        async with _init_lock:
            if _trade_logger is None:  # double-check
                _trade_logger = TradeLogger()
    return _trade_logger


class PipelineResponse(BaseModel):
    status: str
    message: str
    allocations: list[Dict[str, Any]]
    reasoning: Dict[str, Any]


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


# ═════════════════════════════════════════════════════════════════════
# v3.0: Gemini AI Advisor (Standalone — no Antigravity SDK needed)
# ═════════════════════════════════════════════════════════════════════

_gemini_advisor = None


async def _get_advisor():
    """Lazy-init GeminiAdvisor singleton."""
    global _gemini_advisor
    if _gemini_advisor is None:
        async with _init_lock:
            if _gemini_advisor is None:  # double-check
                from nexus_quant_os.advisor.advisor_v2 import GeminiAdvisor
                _gemini_advisor = GeminiAdvisor()
    return _gemini_advisor


@app.post("/api/advisor/chat", response_model=ChatResponse)
async def advisor_chat(req: ChatRequest):
    """Conversational AI Financial Advisor (Gemini 2.5 Flash)."""
    try:
        advisor = await _get_advisor()
        response_text = advisor.chat(req.message)
        return ChatResponse(response=response_text, conversation_id="gemini-session")
    except Exception as e:
        logger.error("Advisor chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/advisor/upload")
async def advisor_upload(
    file: UploadFile = File(...),
    prompt: str = "請分析這張券商截圖，列出所有持倉",
):
    """Upload a broker screenshot for AI analysis."""
    tmp_path = None
    try:
        advisor = await _get_advisor()
        suffix = Path(file.filename or "img.png").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        response_text = advisor.chat(prompt, image_path=tmp_path)
        return {"response": response_text}
    except Exception as e:
        logger.error("Advisor upload error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/advisor/profile")
async def advisor_profile():
    """Get stored user profile."""
    try:
        advisor = await _get_advisor()
        return {"profile": advisor.handle_profile()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════
# v3.0: Dashboard & Health Endpoints
# ═════════════════════════════════════════════════════════════════════


@app.get("/api/dashboard")
async def get_dashboard():
    """Unified dashboard data: account + model + health summary."""
    try:
        from nexus_quant_os.advisor.pipeline_bridge import PipelineBridge
        bridge = PipelineBridge()

        account = bridge.get_account_status()
        checkpoint = bridge.get_checkpoint_info()

        return {
            "account": account,
            "model": checkpoint,
            "assets": list(ASSET_UNIVERSE),
        }
    except Exception as e:
        logger.error("Dashboard error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def get_health():
    """Run pre-trade health checks and return results."""
    try:
        from nexus_quant_os.monitoring.health_check import (
            run_pre_trade_checks, Severity,
        )
        from nexus_quant_os.data_pipelines.data_loader import load_all_data

        end_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        daily_prices, macro_data = load_all_data(
            tickers=ASSET_UNIVERSE,
            fred_api_key=FRED_API_KEY,
            start=DATA_START,
            end=end_str,
        )

        report = run_pre_trade_checks(
            prices_df=daily_prices,
            macro_df=macro_data,
        )

        checks = []
        for c in report.checks:
            checks.append({
                "name": c.name,
                "severity": c.severity.value,
                "message": c.message,
            })

        return {
            "is_healthy": report.is_healthy,
            "summary": report.summary(),
            "n_ok": report.n_ok,
            "n_warnings": report.n_warnings,
            "n_critical": report.n_critical,
            "checks": checks,
        }
    except Exception as e:
        logger.error("Health check error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return "<h1>Index.html not found!</h1>"
    with open(index_file, "r") as f:
        return HTMLResponse(
            content=f.read(),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )


@app.post("/api/run_pipeline", response_model=PipelineResponse)
async def run_pipeline():
    """Execute the full Nexus Quant OS DAG pipeline and return structured JSON results."""
    async with _pipeline_lock:
      try:
        logger.info("Starting Pipeline execution via API")
        
        if not FRED_API_KEY:
            raise HTTPException(status_code=500, detail="FRED_API_KEY is not set.")

        # 1. Load Data
        end_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        daily_prices, macro_data = load_all_data(
            tickers=ASSET_UNIVERSE,
            fred_api_key=FRED_API_KEY,
            start=DATA_START,  # Use full data so HMM can properly learn regimes
            end=end_str,
        )

        # 2. PiT Alignment
        assets_sorted = sorted(ASSET_UNIVERSE)
        frames = []
        for asset in assets_sorted:
            ap = daily_prices[daily_prices["asset_id"] == asset].copy()
            am = macro_data[macro_data["asset_id"] == asset].copy()
            aligned, _ = enforce_pit_alignment(
                ap,
                am,
                timestamp_col="timestamp",
                asset_col="asset_id",
                max_drift_days=45,
                drop_unmatched=True,
                preserve_right_timestamp=True
            )
            frames.append(aligned)
        aligned_df = pd.concat(frames, ignore_index=True).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

        # 3. Dataset Construction（推論模式：保留最後一天）
        X, y_norm, _, dates, assets = build_daily_dataset(aligned_df, inference_mode=True)
        if len(X) == 0:
            raise HTTPException(status_code=500, detail="Not enough data to construct features for today.")

        latest_X = X[-1].reshape(1, -1)
        latest_date = dates[-1]

        # 4. Load Checkpoint
        ckpt_path = find_latest_checkpoint()
        if not ckpt_path:
            raise HTTPException(status_code=500, detail="No trained MoE checkpoint found.")

        router, scaler, trained_assets, firewall = load_checkpoint(ckpt_path)
        
        # 5. Inference
        X_scaled = scaler.transform(latest_X).astype(np.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            out = router(X_tensor)
            raw_weights = out.combined_output[0].numpy()
            expert_util = out.expert_utilisation.numpy()

        # Scale weights to sum(abs(weights)) == 1.0
        abs_sum = np.abs(raw_weights).sum() + 1e-8
        norm_weights = raw_weights / abs_sum

        # ── v2.0 Phase 2.5: Risk Firewall (Run before optimizer to get HMM state) ─────
        sp500_df = daily_prices[daily_prices["asset_id"] == "SPY"].copy()
        sp500_df.set_index("timestamp", inplace=True)

        # Filter SPY specifically for firewall
        spy_df = aligned_df[aligned_df["asset_id"] == "SPY"].copy().reset_index(drop=True)
        spy_feature_matrix, spy_feature_names = engineer_features(spy_df)
        
        if firewall is None:
            raise HTTPException(status_code=500, detail="Checkpoint does not contain a trained firewall.")
        
        # Extract the observation window (last 20 days)
        observation_window = spy_feature_matrix[-20:]
        firewall_result = firewall.evaluate(
            market_features=observation_window,
            raw_weights=norm_weights
        )

        # ── v2.0 Phase 3: 投組優化器 (Risk Parity / Constrained MVO) ─────
        global _optimizer, _sentiment, _allocator, _smoother, _last_assets
        assets_list = list(assets)  # 提前定義，供下方 allocator 判斷使用
        if _optimizer is None or _last_assets != assets_list:
            _optimizer = PortfolioOptimizer(
                n_assets=len(assets),
                config=OptimizerConfig(
                    max_single_weight=0.35,
                    gross_exposure=1.0,
                    min_cash=0.05,
                    dispersion_threshold=0.45,
                ),
                asset_names=assets_list,
            )
        # 用最近 60 天報酬估計共變異數矩陣
        if len(y_norm) >= 60:
            cov_matrix = np.cov(y_norm[-60:], rowvar=False)
        else:
            cov_matrix = np.eye(len(assets)) * 0.01
        optimized_weights = _optimizer.optimize(
            moe_weights=norm_weights,
            expert_utilisation=expert_util,
            cov_matrix=cov_matrix,
            hmm_bear_prob=firewall_result.hmm_bear_prob,
        )

        # Apply Firewall Scaling
        safe_weights = optimized_weights * firewall_result.scale_factor

        # ── v2.0 Phase 5: LLM 情緒引擎調整（可選） ──────────────────────
        sentiment_adjustment = 1.0  # 預設無調整
        try:
            if _sentiment is None:
                _sentiment = SentimentAggregator(
                    use_gemini=True,   # Gemini 2.0 Flash（主模型，免費 API）
                    use_gemma=True,    # Gemma 4（備用模型，同一 API Key）
                    use_deepseek=False,
                )
            # 即時財經新聞 RSS 抓取（30 分鐘快取）
            global _news_fetcher
            if _news_fetcher is None:
                _news_fetcher = NewsFetcher(cache_ttl_seconds=1800)
            live_headlines = _news_fetcher.fetch_latest(max_headlines=15)
            logger.info("即時新聞標題: %d 條", len(live_headlines))
            sentiment_result = _sentiment.get_daily_sentiment(
                headlines=live_headlines
            )
            # 情緒極端時微調權重（±10% 範圍）
            sentiment_adjustment = 1.0 + sentiment_result.sentiment_daily * 0.10
            logger.info(
                "情緒調整因子: %.4f (daily=%.4f)",
                sentiment_adjustment, sentiment_result.sentiment_daily,
            )
        except Exception as e:
            logger.warning("LLM 情緒引擎不可用，跳過調整: %s", e)
            sentiment_adjustment = 1.0

        # ── v2.0 Phase 2: 政體自適應配置 ─────────────────────────
        # 若 assets 變化（不同 checkpoint），重建 allocator 和 smoother
        if _last_assets != assets_list:
            _allocator = None
            _smoother = None
            _last_assets = assets_list
        if _allocator is None:
            _allocator = RegimeAllocator(
                assets=assets_list,
                config=RegimeAllocatorConfig(
                    min_confidence=0.65,
                    max_prior_blend=0.25,
                    transition_smoothing=0.8,
                ),
            )
        regime_blended = _allocator.blend(
            moe_weights=safe_weights,
            regime_label=firewall_result.hmm_regime_label,
            regime_confidence=max(
                firewall_result.hmm_bear_prob,
                firewall_result.hmm_danger_prob,
            ),
            hmm_bear_prob=firewall_result.hmm_bear_prob,
            hmm_danger_prob=firewall_result.hmm_danger_prob,
        )

        regime_blended = regime_blended * sentiment_adjustment

        # ── v2.0 Phase 1: 權重平滑器 ──────────────────────────────
        if _smoother is None:
            _smoother = WeightSmoother(
                n_assets=len(assets),
                config=SmootherConfig(
                    alpha=0.30,
                    min_rebalance_threshold=0.04,
                    max_single_turnover=0.08,
                    min_hold_days=5,
                    signal_stability_window=3,
                ),
            )
            
        vol_idx = spy_feature_names.index("realised_vol")
        latest_vol = spy_feature_matrix[-1, vol_idx]
        vix_proxy = float(latest_vol * np.sqrt(252) * 100)
        
        if len(spy_feature_matrix) >= 5:
            vol_5ma = spy_feature_matrix[-5:, vol_idx].mean()
            vix_5ma = float(vol_5ma * np.sqrt(252) * 100)
        else:
            vix_5ma = vix_proxy
        
        final_weights = _smoother.smooth(
            regime_blended,
            expert_utilization=expert_util,
            vix_value=vix_proxy,
            vix_5ma=vix_5ma,
        )

        # ── HEALTH GATE 2: Weight Sanity (Layer 2) ───────────────────
        weight_check = check_weight_sanity(
            final_weights, asset_names=list(assets), max_single_weight=0.40,
        )
        if weight_check.severity == Severity.CRITICAL:
            logger.error("Weight sanity CRITICAL: %s", weight_check.message)
            raise HTTPException(status_code=500, detail=f"Model weight sanity FAILED: {weight_check.message}")

        # Formatting Output
        allocations = []
        for i, asset in enumerate(assets):
            rw = float(norm_weights[i])
            sw = float(final_weights[i])
            action = "BUY" if sw > 0.05 else ("SELL" if sw < -0.05 else "HOLD")
            allocations.append({
                "asset": asset,
                "raw_weight": rw,
                "scale": firewall_result.scale_factor,
                "regime_weight": float(regime_blended[i]),
                "safe_weight": sw,
                "action": action
            })

        # Formatting Reasoning
        # 使用實際執行日期（今天或上一個交易日），而非 dataset 的 dates[-1]
        pipeline_date = pd.Timestamp.today().normalize()
        # 若今天非交易日（週末），回退到上一個交易日
        if pipeline_date.dayofweek >= 5:  # 5=週六, 6=週日
            pipeline_date = pipeline_date - pd.tseries.offsets.BDay(1)
        reasoning = {
            "date": pipeline_date.strftime("%Y-%m-%d"),
            "risk_tier": firewall_result.risk_tier.name,
            "scale_factor": firewall_result.scale_factor,
            "hmm_regime": firewall_result.hmm_regime_label,
            "hmm_extreme_prob": float(firewall_result.hmm_danger_prob),
            "hmm_bear_prob": float(firewall_result.hmm_bear_prob),
            "ood_score": float(firewall_result.ood_combined_score),
            "ood_flagged": firewall_result.ood_is_flagged,
            "expert_utilization": {
                f"Expert-{i}": float(u) for i, u in enumerate(expert_util)
            },
            "firewall_reason": firewall_result.veto_reason,
            "optimizer_mode": "RiskParity" if float(np.std(expert_util)) > 0.30 else "MVO",
            "sentiment_adjustment": float(sentiment_adjustment),
        }

        return PipelineResponse(
            status="success",
            message="Pipeline executed successfully.",
            allocations=allocations,
            reasoning=reasoning
        )

      except HTTPException:
        raise
      except (KeyError, FileNotFoundError, EnvironmentError) as e:
        logger.exception("Pipeline configuration/service error")
        raise HTTPException(status_code=503, detail=str(e))
      except (ValueError, TypeError) as e:
        logger.exception("Pipeline input validation error")
        raise HTTPException(status_code=422, detail=str(e))
      except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════
# v2.1: 模擬交易 API (Paper Trading Endpoints)
# ═════════════════════════════════════════════════════════════════════


@app.post("/api/execute_trades")
async def execute_trades(pipeline_result: PipelineResponse):
    """
    接收 /api/run_pipeline 的輸出，執行模擬交易。

    流程:
    1. 從 allocations 提取目標權重
    2. SimulatedBroker.reconcile() 計算差額
    3. SimulatedBroker.execute() 模擬成交
    4. TradeLogger 記錄交易日誌
    5. 回傳交易結果
    """
    async with _pipeline_lock:
      try:
        broker = await _get_broker()
        trade_logger = await _get_trade_logger()

        # 檢查市場是否開盤（模擬模式允許非開盤時段執行，但會記錄）
        market_open = broker.is_market_open()
        logger.info("市場狀態: %s", "開盤" if market_open else "休市")

        # 從 pipeline 結果提取目標權重
        target_weights = {}
        for alloc in pipeline_result.allocations:
            weight = alloc.get("safe_weight", 0.0)
            if weight >= 0.0:  # include zero weights so broker can sell
                target_weights[alloc["asset"]] = weight

        # Include 0.0 target for positions currently held but not in pipeline output
        current_positions = broker.get_positions()
        for pos in current_positions:
            if pos.symbol not in target_weights and pos.symbol != 'USD':
                target_weights[pos.symbol] = 0.0  # signal to sell

        if not target_weights:
            return {
                "status": "skipped",
                "message": "No meaningful target weights from pipeline.",
                "trades": [],
                "portfolio_value": broker.get_portfolio_value(),
            }

        # 正規化權重確保總和 = 1.0
        total = sum(target_weights.values())
        if total > 0:
            target_weights = {k: v / total for k, v in target_weights.items()}

        # 對帳 + 執行
        intents = broker.reconcile(target_weights)
        results = broker.execute(intents)

        # 記錄每日快照
        broker.take_daily_snapshot()

        # 寫入交易日誌
        account = broker.get_account()
        positions = broker.get_positions()
        trade_logger.log_daily_performance(account, positions)
        for result in results:
            trade_logger.log_trade(result, pipeline_result.reasoning)

        return {
            "status": "success",
            "message": f"Executed {len(results)} trades.",
            "market_open": market_open,
            "trades": [
                {
                    "symbol": r.symbol,
                    "side": r.side,
                    "qty": r.qty,
                    "filled_price": r.filled_price,
                    "commission": r.commission,
                    "order_id": r.order_id,
                    "status": r.status,
                }
                for r in results
            ],
            "portfolio_value": broker.get_portfolio_value(),
        }

      except Exception as e:
        logger.exception("Trade execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio")
async def get_portfolio():
    """回傳當前模擬交易組合的即時持倉。"""
    try:
        broker = await _get_broker()
        account = broker.get_account()
        positions = broker.get_positions()

        return {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "timestamp": account.timestamp.isoformat(),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pl": p.unrealized_pl,
                    "unrealized_pl_pct": p.unrealized_pl_pct,
                }
                for p in positions
            ],
        }

    except Exception as e:
        logger.exception("Portfolio fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
async def get_performance():
    """回傳歷史績效曲線（每日淨值快照）。"""
    try:
        broker = await _get_broker()
        history = broker.get_performance_history()

        if not history:
            return {"status": "empty", "message": "No performance history yet.", "data": []}

        initial_value = history[0].get("equity", 100_000.0)
        result_history = []
        for entry in history:
            new_entry = {**entry, "cumulative_return": (entry["equity"] / initial_value - 1) * 100}
            result_history.append(new_entry)

        return {
            "status": "success",
            "data": result_history,
            "summary": {
                "initial_value": initial_value,
                "current_value": history[-1].get("equity", 0),
                "total_return_pct": history[-1].get("cumulative_return", 0),
                "trading_days": len(history),
            },
        }

    except Exception as e:
        logger.exception("Performance fetch failed")
        raise HTTPException(status_code=500, detail=str(e))

# ── Alpha Hunter 端點 ─────────────────────────────────────

# 全域單例（lazy-init，與其他模組一致）
_alpha_generator = None

async def _get_alpha_generator():
    global _alpha_generator
    if _alpha_generator is None:
        async with _init_lock:
            if _alpha_generator is None:
                from nexus_quant_os.alpha_hunter.signal_generator import AlphaSignalGenerator
                _alpha_generator = AlphaSignalGenerator(
                    gemini_api_key=os.environ.get("GEMINI_API_KEY"),
                )
    return _alpha_generator

@app.get("/api/alpha/signals")
async def get_alpha_signals():
    """回傳最新的 Alpha 信號列表。"""
    import asyncio
    try:
        generator = await _get_alpha_generator()
        from nexus_quant_os.alpha_hunter.models import SignalStrength
        
        # 在 thread pool 中執行阻塞操作，避免卡住 event loop
        signals = await asyncio.to_thread(
            generator.generate_signals,
            include_supply_chain=False,
            include_ai_analysis=True,
        )
        return {
            "status": "success",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": [s.to_dict() for s in signals if s.signal_strength != SignalStrength.AVOID],
            "total": len(signals),
            "strong_buy_count": sum(1 for s in signals if s.signal_strength == SignalStrength.STRONG_BUY),
            "buy_count": sum(1 for s in signals if s.signal_strength == SignalStrength.BUY),
        }
    except Exception as e:
        logger.exception("Alpha signal generation failed")
        raise HTTPException(status_code=500, detail=str(e))

import re

def _sanitize_ticker(ticker: str) -> str:
    """防呆：嚴格檢查 Ticker 格式，防止路徑穿越與記憶體炸彈"""
    ticker = ticker.upper().strip()
    if not re.match(r"^[A-Z0-9\-\.]{1,10}$", ticker):
        raise HTTPException(
            status_code=400,
            detail=f"不合法的股票代號：{ticker[:20]}"
        )
    return ticker

@app.get("/api/alpha/scan/{ticker}")
async def scan_single_ticker(ticker: str):
    """掃描單一公司的完整 Alpha 分析。"""
    ticker = _sanitize_ticker(ticker)
    import asyncio
    try:
        generator = await _get_alpha_generator()
        signal = await asyncio.to_thread(
            generator.generate_single, ticker.upper()
        )
        return {
            "status": "success",
            "signal": signal.to_dict(),
        }
    except Exception as e:
        logger.exception("Alpha scan failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alpha/supply-chain/{ticker}")
async def get_supply_chain(ticker: str):
    """取得一家公司的供應鏈圖譜。"""
    ticker = _sanitize_ticker(ticker)
    import asyncio
    try:
        generator = await _get_alpha_generator()
        graph = await asyncio.to_thread(
            generator.get_supply_chain, ticker.upper()
        )
        return {
            "status": "success",
            "center": graph.center_ticker,
            "edges": [
                {
                    "source": e.source_ticker,
                    "target": e.target_ticker,
                    "relation": e.relation.value,
                    "revenue_pct": e.revenue_pct,
                    "confidence": e.confidence,
                }
                for e in graph.edges
            ],
            "total_connections": len(graph.edges),
        }
    except Exception as e:
        logger.exception("Supply chain build failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alpha/supply-chain/{ticker}/recursive")
async def get_recursive_supply_chain(
    ticker: str,
    depth: int = 3,
    max_calls: int = 30,
):
    """取得一家公司的遞迴供應鏈圖譜。"""
    ticker = _sanitize_ticker(ticker)
    depth = max(1, min(depth, 3))            # 限制 1~3
    max_calls = max(5, min(max_calls, 50))   # 限制 5~50
    import asyncio

    try:
        generator = await _get_alpha_generator()
        graph = await asyncio.to_thread(
            generator._tracker.build_recursive_graph,
            ticker, depth, max_calls
        )
        return {"status": "success", "data": graph.to_dict()}
    except Exception as e:
        logger.exception("Recursive supply chain failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alpha/supply-chain/{ticker}/recursive/stream")
async def stream_recursive_supply_chain(
    ticker: str,
    request: Request,
    depth: int = 3,
    max_calls: int = 30,
):
    """取得一家公司的遞迴供應鏈圖譜（SSE 進度回報）。"""
    ticker = _sanitize_ticker(ticker)
    depth = max(1, min(depth, 3))
    max_calls = max(5, min(max_calls, 50))
    import asyncio

    q = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress_callback(data: dict):
        # We need to run the queue.put in the event loop thread
        try:
            loop.call_soon_threadsafe(q.put_nowait, data)
        except Exception:
            pass

    async def event_generator():
        generator = await _get_alpha_generator()
        
        # Start the blocking build_recursive_graph in a thread
        build_task = asyncio.create_task(
            asyncio.to_thread(
                generator._tracker.build_recursive_graph,
                ticker, depth, max_calls, 0.5, progress_callback
            )
        )
        
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected from SSE stream")
                    build_task.cancel()
                    break

                # Drain all queued progress messages first
                while not q.empty():
                    try:
                        data = q.get_nowait()
                        yield f"data: {json.dumps(data)}\n\n"
                    except asyncio.QueueEmpty:
                        break

                if build_task.done():
                    try:
                        graph = build_task.result()
                        yield f"data: {json.dumps({'type': 'done', 'data': graph.to_dict()})}\n\n"
                    except Exception as e:
                        logger.exception("Recursive supply chain task failed")
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    return

                # Wait briefly for new queue items or task completion
                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.5)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    pass  # Loop back to check build_task and disconnection
        except Exception as e:
            logger.exception("SSE stream error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/alpha/history")
async def get_alpha_history():
    """取得最近一次的 Alpha 信號掃描歷史與前次分數。"""
    import json
    from nexus_quant_os.alpha_hunter._constants import SIGNALS_HISTORY_DIR
    
    try:
        if not SIGNALS_HISTORY_DIR.exists():
            return {"status": "success", "data": None, "message": "No history found"}
            
        files = sorted(SIGNALS_HISTORY_DIR.glob("signals_*.json"), reverse=True)
        if not files:
            return {"status": "success", "data": None, "message": "No history found"}
            
        latest_file = files[0]
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        previous_scores = {}
        if len(files) > 1:
            try:
                with open(files[1], "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                    for s in prev_data.get("signals", []):
                        previous_scores[s.get("ticker")] = s.get("composite_score")
            except Exception:
                pass
                
        data["previous_scores"] = previous_scores
            
        return {
            "status": "success",
            "data": data,
            "filename": latest_file.name
        }
    except Exception as e:
        logger.exception("Failed to load signal history")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alpha/signals/stream")
async def stream_alpha_signals():
    """使用 Server-Sent Events (SSE) 串流 Alpha 信號生成進度。"""
    import asyncio
    import json
    from fastapi.responses import StreamingResponse
    
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancelled = False

    def progress_callback(msg: str):
        if cancelled:
            return
        try:
            asyncio.run_coroutine_threadsafe(queue.put(f"data: {json.dumps({'message': msg})}\n\n"), loop)
        except Exception:
            pass

    async def event_generator():
        nonlocal cancelled
        generator = await _get_alpha_generator()
        
        # 在背景執行緒啟動同步的 generate_signals，並傳入 progress_callback
        task = asyncio.create_task(
            asyncio.to_thread(
                generator.generate_signals,
                universe=None,
                include_supply_chain=True,
                include_ai_analysis=True,
                progress_callback=progress_callback
            )
        )
        
        try:
            while not task.done():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield msg
                except asyncio.TimeoutError:
                    # 保持連線
                    yield ": keep-alive\n\n"
                    
            # 確保佇列中的所有訊息都被送出
            while not queue.empty():
                yield await queue.get()
                
            try:
                task.result()
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.exception("Alpha signals stream failed")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except asyncio.CancelledError:
            cancelled = True
            logger.info("Client disconnected during SSE stream")
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")
