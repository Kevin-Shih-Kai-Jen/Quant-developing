#!/usr/bin/env python3
"""
Nexus Quant OS — Automated Project Scaffolding Script
=====================================================

Generates the complete directory tree, __init__.py module markers, and
placeholder source files as specified in the System Architecture Report
§6.1 (High-Modularity Physical Directory Tree).

Usage:
    python setup_project.py            # creates under ./nexus_quant_os/
    python setup_project.py /my/path   # creates under /my/path/nexus_quant_os/

Idempotent: safe to re-run — existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# 1. Directory & File Manifest — single source of truth
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SourceFile:
    """Describes a single Python source file to scaffold."""
    filename: str
    docstring: str
    imports: list[str] = field(default_factory=list)


# Top-level package docstring
ROOT_PACKAGE_DOC = textwrap.dedent('''\
    """
    Nexus Quant OS — Next-Generation Quantitative Trading System
    =============================================================

    A microkernel-based, DAG-enforced quantitative platform featuring:
      • Strict unidirectional data flow with Point-in-Time alignment
      • Mixture-of-Experts (MoE) dynamic model routing
      • Probabilistic intelligent risk firewall (HMM + OOD + Bayesian)
      • Plugin ecosystem for macro & alternative data

    Architecture Reference: System Architecture Report v1.0
    """
''')

# Manifest keyed by *relative directory path* → list of source files
# Directories that contain only __init__.py can map to an empty list.
MANIFEST: dict[str, list[SourceFile]] = {
    # ── Root package ──────────────────────────────────────────────
    ".": [],

    # ── Data Pipelines ────────────────────────────────────────────
    "data_pipelines": [],
    "data_pipelines/fetchers": [
        SourceFile(
            "sec_fetcher.py",
            "Asynchronous SEC EDGAR 10-K / 10-Q filing fetcher.\n"
            "Retrieves and caches corporate filings for supply-chain graph construction.",
        ),
        SourceFile(
            "macro_fetcher.py",
            "Macro-economic data fetcher (FRED, BLS, central bank APIs).\n"
            "Handles revision-aware Point-in-Time ingestion for yield curves & indicators.",
        ),
        SourceFile(
            "price_fetcher.py",
            "High-frequency / daily price & volume data fetcher.\n"
            "Supports equities, futures, FX and crypto via pluggable broker adapters.",
        ),
    ],
    "data_pipelines/point_in_time": [
        SourceFile(
            "pit_database.py",
            "Point-in-Time historical database manager.\n"
            "Maintains immutable revision chains for every macro & fundamental data point,\n"
            "guaranteeing that no future revisions leak into past decision timestamps.",
        ),
        SourceFile(
            "revision_tracker.py",
            "Tracks and versions data revisions (e.g., GDP revisions, earnings restates).\n"
            "Ensures the aligner always references the *first-published* value at time T.",
        ),
    ],
    "data_pipelines": [              # additional file in the parent dir
        SourceFile(
            "aligner.py",
            "Core feature aligner implementing ``pandas.merge_asof``.\n\n"
            "Enforces ``direction='backward'`` with configurable ``tolerance`` to\n"
            "physically prevent any Look-Ahead Bias at the data-engineering level.\n"
            "See Architecture Report §5.1.",
            imports=["import pandas as pd"],
        ),
    ],

    # ── Plugin Ecosystem ──────────────────────────────────────────
    "plugins": [
        SourceFile(
            "base_plugin.py",
            "Abstract base class defining the standardised JSON IPC interface\n"
            "and lifecycle contract (``initialize → fetch → transform → emit``)\n"
            "for all Macro and Asset plugins.\n\n"
            "Every plugin output carries ``lookahead_safe``, ``execution_timestamp``,\n"
            "and ``data_granularity`` fields for downstream PiT validation.",
            imports=["from abc import ABC, abstractmethod"],
        ),
    ],
    "plugins/macro": [
        SourceFile(
            "fed_policy_plugin.py",
            "Federal Reserve policy plugin.\n"
            "Scrapes FOMC minutes & statements, runs NLP sentiment extraction,\n"
            "and emits hawk/dove index + policy uncertainty score.",
        ),
        SourceFile(
            "yield_curve_plugin.py",
            "Yield curve plugin.\n"
            "Analyses Treasury yield curve slope, curvature & spread via PCA\n"
            "to extract macro liquidity factors.",
        ),
        SourceFile(
            "alternative_data_plugin.py",
            "Alternative data plugin.\n"
            "Integrates satellite-derived freight throughput, Geopolitical Risk\n"
            "Index (GPR), and sector-specific network graph features.",
        ),
    ],
    "plugins/assets": [
        SourceFile(
            "supply_chain_momentum_plugin.py",
            "Supply-chain customer momentum plugin (Cohen & Frazzini, 2008).\n"
            "Parses SEC 10-K filings to build dynamic customer-supplier network\n"
            "graphs, then computes information-diffusion speed & institutional\n"
            "cross-holding ratio as structured features.",
        ),
        SourceFile(
            "social_sentiment_plugin.py",
            "Social & news sentiment plugin.\n"
            "Aggregates real-time social-media and news-wire sentiment signals\n"
            "and normalises them into the standard plugin JSON contract.",
        ),
    ],

    # ── Deep-Learning Expert Models ───────────────────────────────
    "models": [
        SourceFile(
            "moe_router.py",
            "Mixture-of-Experts (MoE) dynamic gating router.\n\n"
            "Implements Top-K and Expert-Choice routing with auxiliary\n"
            "load-balancing loss to prevent expert collapse.\n"
            "See Architecture Report §3.",
            imports=[
                "import torch",
                "import torch.nn as nn",
                "import torch.nn.functional as F",
            ],
        ),
    ],
    "models/experts": [
        SourceFile(
            "timeseries_xlstm.py",
            "xLSTM / StoxLSTM expert for long-horizon financial time series.\n"
            "Features exponential gating, stochastic latent variables, and\n"
            "negative-Sharpe-ratio loss objective.",
            imports=["import torch", "import torch.nn as nn"],
        ),
        SourceFile(
            "timeseries_mamba.py",
            "TSMamba / ms-Mamba expert — linear-complexity Selective State Space\n"
            "Model for multi-variate high-dimensional time series.\n"
            "Supports bidirectional encoding and channel-compressed attention.",
            imports=["import torch", "import torch.nn as nn"],
        ),
        SourceFile(
            "timeseries_patchtst.py",
            "PatchTST expert — channel-independent patch-based Transformer\n"
            "for cross-asset covariance and long-horizon forecasting.\n"
            "Reduces memory quadratically via local semantic patching.",
            imports=["import torch", "import torch.nn as nn"],
        ),
        SourceFile(
            "nlp_finllama.py",
            "FinLlama expert (Llama-2 7B + LoRA fine-tune).\n"
            "Generator-classifier architecture for long-form financial text\n"
            "(10-K filings, FOMC minutes) producing sentiment direction\n"
            "and calibrated confidence scores.",
            imports=["import torch", "import torch.nn as nn"],
        ),
        SourceFile(
            "nlp_finbert.py",
            "FinBERT expert for ultra-low-latency headline sentiment.\n"
            "Achieves ≥86.66% accuracy on ESG & short-news classification\n"
            "with industry-specific fine-tuning capability.",
            imports=["import torch", "import torch.nn as nn"],
        ),
        SourceFile(
            "tabular_tabnet.py",
            "TabNet expert for structured fundamental & supply-chain features.\n"
            "Sequential attention masking provides white-box interpretability\n"
            "while matching deep-network representation power.",
            imports=["import torch", "import torch.nn as nn"],
        ),
    ],

    # ── Intelligent Risk Firewall ─────────────────────────────────
    "risk_firewall": [
        SourceFile(
            "hmm_regime_detector.py",
            "Gaussian Hidden Markov Model regime detector.\n"
            "Identifies latent market states (bull / bear / extreme-inflation)\n"
            "via EM-trained transition & emission matrices.  Triggers regime-\n"
            "switching when posterior probability exceeds dynamic threshold.\n"
            "See Architecture Report §4.1.",
            imports=["import numpy as np"],
        ),
        SourceFile(
            "ood_anomaly_detector.py",
            "Out-of-Distribution anomaly detection module.\n"
            "Combines deep Autoencoder reconstruction error with Isolation\n"
            "Forest space-partitioning to flag unseen market structures.\n"
            "See Architecture Report §4.2.",
            imports=["import torch", "import torch.nn as nn", "import numpy as np"],
        ),
        SourceFile(
            "bayesian_stress_tester.py",
            "Bayesian Network stress tester.\n"
            "Constructs a DAG of conditional dependencies among MoE predictions,\n"
            "HMM states and fundamental shocks.  Runs Gibbs-sampler MCMC to\n"
            "estimate posterior VaR and Expected Shortfall at 99%% confidence.\n"
            "See Architecture Report §4.3.",
            imports=["import numpy as np"],
        ),
    ],

    # ── Portfolio Optimisation ────────────────────────────────────
    "portfolio": [
        SourceFile(
            "optimizer.py",
            "Posterior-probability-driven portfolio optimiser.\n"
            "Maximises Sharpe ratio subject to Bayesian VaR constraints,\n"
            "with dynamic position scaling based on regime probabilities.",
            imports=["import numpy as np"],
        ),
    ],

    # ── Execution ─────────────────────────────────────────────────
    "execution": [
        SourceFile(
            "broker_router.py",
            "Smart order router supporting FIX protocol and REST broker APIs.\n"
            "Handles slippage control, TWAP / VWAP scheduling, and\n"
            "position reconciliation.",
        ),
    ],
}


# ─────────────────────────────────────────────────────────────────────
# 2. Sub-package docstrings for __init__.py files
# ─────────────────────────────────────────────────────────────────────

INIT_DOCS: dict[str, str] = {
    ".":
        ROOT_PACKAGE_DOC.strip(),
    "data_pipelines":
        '"""High-dimensional data extraction, PiT storage & feature alignment."""',
    "data_pipelines/fetchers":
        '"""Async API data fetchers (SEC, macro, high-frequency prices)."""',
    "data_pipelines/point_in_time":
        '"""Point-in-Time historical database & revision management."""',
    "plugins":
        '"""Microkernel plugin ecosystem — macro & asset alternative-data plugins."""',
    "plugins/macro":
        '"""Macro-economic, yield-curve & Fed-policy plugins."""',
    "plugins/assets":
        '"""Asset-specific plugins: supply-chain momentum, social sentiment, etc."""',
    "models":
        '"""Deep-learning & ML core engine — expert models and MoE router."""',
    "models/experts":
        '"""Heterogeneous expert model cluster (xLSTM, Mamba, FinLlama, TabNet …)."""',
    "risk_firewall":
        '"""Probabilistic intelligent risk-control defence modules."""',
    "portfolio":
        '"""Portfolio construction & optimisation."""',
    "execution":
        '"""Trade execution & broker interface."""',
}


# ─────────────────────────────────────────────────────────────────────
# 3. Code generation helpers
# ─────────────────────────────────────────────────────────────────────

def _make_module_header(docstring: str, imports: list[str]) -> str:
    """Return full source text for a placeholder module."""
    lines: list[str] = []
    lines.append('"""')
    for line in docstring.strip().splitlines():
        lines.append(line)
    lines.append('"""')
    lines.append("")

    if imports:
        for imp in imports:
            lines.append(imp)
        lines.append("")

    lines.append("")
    lines.append("# ── Implementation will be added in subsequent build phases ──")
    lines.append("")
    return "\n".join(lines)


def _make_init(docstring: str) -> str:
    """Return source text for an __init__.py with a package docstring."""
    return docstring + "\n"


# ─────────────────────────────────────────────────────────────────────
# 4. Scaffolding engine
# ─────────────────────────────────────────────────────────────────────

def scaffold(root: Path) -> dict[str, int]:
    """Create the full project tree under *root*.

    Returns a summary dict with counts of created dirs, inits, and modules.
    """
    stats = {"dirs": 0, "inits": 0, "modules": 0, "skipped": 0}

    # Collect all unique directory paths (including those implied by MANIFEST keys)
    all_dirs: set[str] = set()
    for rel_dir in MANIFEST:
        all_dirs.add(rel_dir)
    for rel_dir in INIT_DOCS:
        all_dirs.add(rel_dir)

    # ── Create directories ────────────────────────────────────────
    for rel_dir in sorted(all_dirs):
        abs_dir = root / rel_dir
        if not abs_dir.exists():
            abs_dir.mkdir(parents=True, exist_ok=True)
            stats["dirs"] += 1
            print(f"  📁  Created directory:  {abs_dir.relative_to(root.parent)}/")

    # ── Write __init__.py files ───────────────────────────────────
    for rel_dir in sorted(all_dirs):
        init_path = root / rel_dir / "__init__.py"
        if init_path.exists():
            stats["skipped"] += 1
            continue
        doc = INIT_DOCS.get(rel_dir, '""""""')
        init_path.write_text(_make_init(doc), encoding="utf-8")
        stats["inits"] += 1
        print(f"  📄  Created __init__.py: {init_path.relative_to(root.parent)}")

    # ── Write source modules ─────────────────────────────────────
    # Aggregate files per directory (MANIFEST may have duplicate keys)
    aggregated: dict[str, list[SourceFile]] = {}
    for rel_dir, files in MANIFEST.items():
        aggregated.setdefault(rel_dir, []).extend(files)

    for rel_dir, files in sorted(aggregated.items()):
        for sf in files:
            mod_path = root / rel_dir / sf.filename
            if mod_path.exists():
                stats["skipped"] += 1
                continue
            mod_path.write_text(
                _make_module_header(sf.docstring, sf.imports),
                encoding="utf-8",
            )
            stats["modules"] += 1
            print(f"  🐍  Created module:     {mod_path.relative_to(root.parent)}")

    return stats


# ─────────────────────────────────────────────────────────────────────
# 5. CLI entry-point
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold the Nexus Quant OS project directory tree.",
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Parent directory under which 'nexus_quant_os/' will be created "
             "(default: current working directory).",
    )
    args = parser.parse_args()

    root = Path(args.base_dir).resolve() / "nexus_quant_os"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'═' * 64}")
    print(f"  Nexus Quant OS — Project Scaffolding")
    print(f"  Target : {root}")
    print(f"  Time   : {timestamp}")
    print(f"{'═' * 64}\n")

    stats = scaffold(root)

    print(f"\n{'─' * 64}")
    print(f"  ✅ Done!  dirs={stats['dirs']}  __init__={stats['inits']}  "
          f"modules={stats['modules']}  skipped(existing)={stats['skipped']}")
    print(f"{'─' * 64}\n")


if __name__ == "__main__":
    main()
