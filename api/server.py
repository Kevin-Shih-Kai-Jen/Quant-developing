import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from nexus_quant_os.data_pipelines.data_loader import load_all_data
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.risk_firewall.firewall_core import (
    IntelligentRiskFirewall, HMMConfig, OODConfig, FirewallConfig
)
from nexus_quant_os.training.train_moe import (
    ASSET_UNIVERSE, DATA_START, FRED_API_KEY,
    build_daily_dataset, find_latest_checkpoint, load_checkpoint
)
from main import engineer_features
import pandas as pd
import numpy as np
import torch

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexus_quant_os.api")

app = FastAPI(title="Nexus Quant OS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup paths
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PipelineResponse(BaseModel):
    status: str
    message: str
    allocations: list[Dict[str, Any]]
    reasoning: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return "<h1>Index.html not found!</h1>"
    with open(index_file, "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/run_pipeline", response_model=PipelineResponse)
async def run_pipeline():
    """Execute the full Nexus Quant OS DAG pipeline and return structured JSON results."""
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

        # 3. Dataset Construction
        X, _, _, dates, assets = build_daily_dataset(aligned_df)
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

        # 6. Risk Firewall (Loaded from Checkpoint)
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

        # Apply Firewall Scaling
        safe_weights = norm_weights * firewall_result.scale_factor

        # Formatting Output
        allocations = []
        for i, asset in enumerate(assets):
            rw = float(norm_weights[i])
            sw = float(safe_weights[i])
            action = "BUY" if sw > 0.05 else ("SELL" if sw < -0.05 else "HOLD")
            allocations.append({
                "asset": asset,
                "raw_weight": rw,
                "scale": firewall_result.scale_factor,
                "safe_weight": sw,
                "action": action
            })

        # Formatting Reasoning
        reasoning = {
            "date": latest_date.strftime("%Y-%m-%d"),
            "risk_tier": firewall_result.risk_tier.name,
            "scale_factor": firewall_result.scale_factor,
            "hmm_regime": firewall_result.hmm_regime_label,
            "hmm_extreme_prob": float(firewall_result.hmm_danger_prob),
            "hmm_bear_prob": float(getattr(firewall_result, 'hmm_bear_prob', 0.0)),
            "ood_score": float(firewall_result.ood_combined_score),
            "ood_flagged": firewall_result.ood_is_flagged,
            "expert_utilization": {
                f"Expert-{i}": float(u) for i, u in enumerate(expert_util)
            },
            "firewall_reason": firewall_result.veto_reason
        }

        return PipelineResponse(
            status="success",
            message="Pipeline executed successfully.",
            allocations=allocations,
            reasoning=reasoning
        )

    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=str(e))
