"""
data_pipelines/data_loader.py — Real Market Data Ingestion Engine
=================================================================

資料來源：
    高頻 (Daily OHLCV)  : Yahoo Finance via yfinance
    低頻 (Macro)        : FRED (Federal Reserve Economic Data) via fredapi

PiT (Point-in-Time) 核心設計：
    FRED API 回傳的日期是「參考月份」的第一天，不是「公開日期」。
    例如：1 月 CPI (CPIAUCSL) 在 FRED 顯示日期為 2024-01-01，
          但實際上是 2024-02-13 才公開。

    本模組對每個指標套用已知的「發佈延遲 (Publication Delay)」，
    將 FRED 的參考日期偏移至近似的實際公開日，確保與 aligner.py 的
    merge_asof(direction='backward') 配合後不產生前瞻偏誤。

Author : Nexus Quant OS — Data Engineering Division
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus_quant_os.data_pipelines.data_loader")

# 確保 data 目錄存在
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# FRED 指標配置表
# ═════════════════════════════════════════════════════════════════════

# pub_delay_days：從 FRED 參考日期到實際公開日的日曆天數
FRED_SERIES_CONFIG: dict[str, dict] = {
    "CPIAUCSL": {
        "col":          "cpi_yoy",
        "pub_delay_days": 45,        # CPI 約在參考月份結束後 6 週公開
        "transform":    "yoy_pct",   # 從原始水平換算為年增率 (%)
    },
    "UNRATE": {
        "col":          "unemployment_rate",
        "pub_delay_days": 35,        # 在參考月份的下個月第一個週五公開
        "transform":    None,
    },
    "INDPRO": {
        "col":          "pmi_manufacturing",   # 工業生產指數作為製造業活動代理
        "pub_delay_days": 17,                  # 發佈於參考月份結束後約 2-3 週
        "transform":    None,
    },
    "DFF": {
        "col":          "fed_funds_rate",
        "pub_delay_days": 1,         # 每日更新，次日公開
        "transform":    None,
    },
    "BAMLH0A0HYM2": {
        "col":          "credit_spread",
        "pub_delay_days": 1,         # 每日更新，次日公開
        "transform":    None,
    },
    "T10Y2Y": {
        "col":          "yield_curve_slope",
        "pub_delay_days": 1,         # 每日更新（10年-2年公債殖利率利差）
        "transform":    None,        # 直接使用原始 basis points
    },
}


# ═════════════════════════════════════════════════════════════════════
# 1. 高頻價格數據 (Yahoo Finance)
# ═════════════════════════════════════════════════════════════════════

def load_price_data(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """從 Yahoo Finance 下載每日 OHLCV 數據。

    Parameters
    ----------
    tickers : list[str]   e.g. ["SPY", "NVDA", "AVGO"]
    start   : str         "YYYY-MM-DD"
    end     : str         "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame  columns: timestamp, asset_id, close, high, low, volume
                  sorted by (asset_id, timestamp) ascending
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance 未安裝。請在 requirements.txt 加入 yfinance>=0.2.38"
        )

    cache_path = os.path.join(DATA_DIR, "price_cache.csv")
    cached_df = None
    fetch_start = start

    # 嘗試讀取快取並決定增量抓取的起點
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            cached_tickers = set(cached_df["asset_id"].unique())
            
            # 檢查是否所有要求的標的都在快取中
            if not set(tickers).issubset(cached_tickers):
                logger.warning("發現新標的加入，捨棄快取，重新完整下載所有資料。")
                cached_df = None
            # 檢查要求的起始日期是否比快取還早
            elif pd.Timestamp(fetch_start) < cached_df["timestamp"].min():
                logger.warning("要求的起始日期早於快取紀錄，捨棄快取，重新完整下載。")
                cached_df = None
            else:
                last_date = cached_df["timestamp"].max()
                if pd.Timestamp(fetch_start) <= last_date:
                    # 只需要抓取快取最後一天的隔天到 end
                    fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info("找到價格快取，最後更新日: %s，增量下載起點: %s", last_date.strftime("%Y-%m-%d"), fetch_start)
        except Exception as e:
            logger.warning("無法讀取價格快取，將重新下載: %s", e)
            cached_df = None

    # 如果增量起點已經超過或等於結束日，直接回傳快取
    if cached_df is not None and pd.Timestamp(fetch_start) >= pd.Timestamp(end):
        # 過濾出本次需要的 tickers
        return cached_df[cached_df["asset_id"].isin(tickers)].reset_index(drop=True)

    logger.info(
        "下載價格數據 | tickers=%s | %s -> %s", tickers, fetch_start, end
    )

    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=fetch_start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                logger.warning("  %s: %s -> %s 無數據回傳，跳過", ticker, fetch_start, end)
                continue

            # yfinance 新版可能回傳 MultiIndex 欄位，統一壓平
            if isinstance(raw.columns, pd.MultiIndex):
                # For single-ticker downloads, take the price-type level
                raw.columns = raw.columns.get_level_values(0)

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(raw.index),
                "asset_id":  ticker,
                "close":     raw["Close"].values.astype(float),
                "high":      raw["High"].values.astype(float),
                "low":       raw["Low"].values.astype(float),
                "volume":    raw["Volume"].values.astype(float),
            }).dropna(subset=["close"])

            rows.append(df)
            logger.info(
                "  ✅ %s: %d 個交易日已下載 (%.2f -> %.2f)",
                ticker, len(df), df["close"].iloc[0], df["close"].iloc[-1],
            )

        except Exception as exc:
            logger.error("  ❌ %s 下載失敗: %s", ticker, exc)

    new_data = None
    if rows:
        new_data = pd.concat(rows, ignore_index=True)
    
    # 合併快取與新資料
    if cached_df is not None and new_data is not None:
        result = pd.concat([cached_df, new_data], ignore_index=True)
    elif cached_df is not None:
        result = cached_df
    elif new_data is not None:
        result = new_data
    else:
        raise RuntimeError("所有標的均下載失敗，且無快取可用。請檢查網路連線。")

    # 排序與去重
    result = result.drop_duplicates(subset=["asset_id", "timestamp"], keep="last")
    result = result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    
    # 存回快取
    result.to_csv(cache_path, index=False)
    logger.info("價格數據快取已更新。")
    return result


# ═════════════════════════════════════════════════════════════════════
# 2. 低頻總經數據 (FRED API) — 含 PiT 公開日偏移
# ═════════════════════════════════════════════════════════════════════

def _fetch_fred_series(fred_api_key: str, series_id: str) -> pd.Series:
    """從 FRED 下載單一指標序列。"""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi 未安裝。請在 requirements.txt 加入 fredapi>=0.5.1"
        )
    fred   = Fred(api_key=fred_api_key)
    series = fred.get_series(series_id)
    series.index = pd.to_datetime(series.index)
    return series.sort_index().dropna()


def load_macro_data(
    fred_api_key: str,
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """從 FRED 下載總經指標，套用 PiT 公開日偏移，並廣播至所有標的。

    關鍵機制：
        1. 對每個月度指標加上 pub_delay_days，將 FRED 參考日期轉換為實際公開日
        2. 在交易日日曆上 forward-fill（已公開的值持續有效，直到下次更新）
        3. 廣播至所有 tickers（總經是全市場共享的）

    Returns
    -------
    pd.DataFrame  columns: timestamp, asset_id, cpi_yoy, unemployment_rate,
                           pmi_manufacturing, fed_funds_rate, credit_spread,
                           yield_curve_slope
    """
    logger.info("下載 FRED 總經數據 | %s -> %s", start, end)

    # 建立交易日日曆（作為 forward-fill 的索引基礎）
    trading_days = pd.bdate_range(start=start, end=end)
    macro_daily  = pd.DataFrame(index=trading_days)
    macro_daily.index.name = "timestamp"

    for series_id, cfg in FRED_SERIES_CONFIG.items():
        col_name   = cfg["col"]
        delay_days = cfg["pub_delay_days"]
        transform  = cfg["transform"]

        try:
            series = _fetch_fred_series(fred_api_key, series_id)

            # 轉換：月度水平值 → 年增率 (%)
            if transform == "yoy_pct":
                series = series.pct_change(12) * 100.0
                series = series.dropna()

            # 套用 PiT 公開日偏移
            series.index = series.index + pd.Timedelta(days=delay_days)

            # 截取到研究時間範圍（稍微寬一點以覆蓋偏移）
            series = series[
                (series.index >= pd.Timestamp(start)) &
                (series.index <= pd.Timestamp(end) + pd.Timedelta(days=60))
            ]

            # 重新索引至交易日 + forward-fill
            series_daily = series.reindex(trading_days, method="ffill")
            macro_daily[col_name] = series_daily

            n_valid = int(series_daily.notna().sum())
            logger.info(
                "  ✅ %-20s -> %-22s: %d/%d 交易日有效 (延遲 %dd)",
                series_id, f"'{col_name}'", n_valid, len(trading_days), delay_days,
            )

        except Exception as exc:
            logger.error("  ❌ %s 下載失敗: %s — 填入 NaN", series_id, exc)
            macro_daily[col_name] = np.nan

    # 只要有「任何一個」特徵出現 NaN (例如部分 API 失敗、或是全數失敗)
    macro_cols = [cfg["col"] for cfg in FRED_SERIES_CONFIG.values()]
    cache_path = os.path.join(DATA_DIR, "macro_cache.csv")
    
    if macro_daily[macro_cols].isna().any().any():
        if os.path.exists(cache_path):
            logger.warning("⚠️ FRED API 抓取異常 (部分或全數失敗)。啟動【真實歷史快取】降級模式...")
            cached_macro = pd.read_csv(cache_path, parse_dates=["timestamp"])
            # 將快取轉為以 timestamp 為 index，並 reindex 到當前 trading_days，然後 ffill
            cached_macro = cached_macro.drop_duplicates("timestamp").set_index("timestamp")
            macro_daily = cached_macro.reindex(trading_days, method="ffill")
            macro_daily = macro_daily.bfill()  # backfill any leading NaNs
            macro_daily.index.name = "timestamp"
            macro_daily = macro_daily.reset_index()
        else:
            logger.warning("⚠️ FRED API 異常且找不到本地快取！啟動合成數據降級模式 (Synthetic Fallback)...")
            macro_daily["cpi_yoy"] = 3.0
            macro_daily["unemployment_rate"] = 4.0
            macro_daily["pmi_manufacturing"] = 50.0
            macro_daily["fed_funds_rate"] = 5.0
            macro_daily["credit_spread"] = 2.0
            macro_daily["yield_curve_slope"] = -0.5
            macro_daily = macro_daily.reset_index()
            macro_daily["timestamp"] = pd.to_datetime(macro_daily["timestamp"])
    else:
        # 下載成功，將最新的正確資料存入快取
        macro_daily = macro_daily.reset_index()
        macro_daily["timestamp"] = pd.to_datetime(macro_daily["timestamp"])
        macro_daily_unique = macro_daily.drop_duplicates("timestamp")
        macro_daily_unique.to_csv(cache_path, index=False)
        logger.info("總經數據快取已更新。")


    # 廣播至所有標的（總經數據對所有資產相同）
    frames = [
        macro_daily.assign(asset_id=asset)
        for asset in tickers
    ]
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    logger.info(
        "總經數據完成: %d rows (%d 標的 × %d 交易日)",
        len(result), len(tickers), len(trading_days),
    )
    return result


# ═════════════════════════════════════════════════════════════════════
# 3. 公開入口
# ═════════════════════════════════════════════════════════════════════

def load_all_data(
    tickers: list[str],
    fred_api_key: str,
    start: str = "2020-01-02",
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """主入口：同時載入價格與總經數據。

    Parameters
    ----------
    tickers      : list[str]   e.g. ["SPY","QQQ","IWM","TLT","GLD","NVDA","AVGO"]
    fred_api_key : str         FRED API Key (免費申請於 fred.stlouisfed.org)
    start        : str         歷史起始日 (建議 2020-01-02 以涵蓋 COVID 崩盤)
    end          : str | None  結束日 (None = 今天)

    Returns
    -------
    daily_prices : pd.DataFrame
    macro_data   : pd.DataFrame
    """
    if end is None:
        end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    daily_prices = load_price_data(tickers, start, end)
    macro_data   = load_macro_data(fred_api_key, tickers, start, end)
    return daily_prices, macro_data
