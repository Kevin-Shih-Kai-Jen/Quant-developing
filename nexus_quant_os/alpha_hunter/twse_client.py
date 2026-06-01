"""
alpha_hunter/twse_client.py — 台股資料客戶端 (基於 FinMind)

實作 FinancialDataClient Protocol，並處理台股特有的資料情境。
防禦 Edge Cases:
    - B10 (千元/百萬元單位)：FinMind API 回傳的財報金額已為絕對數字（元），
      但我們在轉換成 FinancialStatement 時需確認對齊格式。
    - 台股特有的月營收高頻數據 (get_monthly_revenue)。
    - Edge Case #32: 集合競價期間（13:25-13:30）行為異常。
    - Edge Case #40: FinMind API 斷線時的熔斷機制。
"""

import logging
import time
from datetime import datetime, timezone, timedelta, tzinfo as _tzinfo
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .models import FinancialStatement
from .interfaces import FinancialDataClient
from .ticker_resolver import TickerResolver

logger = logging.getLogger(__name__)


# ── 台北時區常數（用於集合競價判斷） ──
_TW_TZ: _tzinfo = ZoneInfo("Asia/Taipei")

# ── 熔斷器常數 (Edge Case #40) ──
# 連續失敗次數超過此閾值後啟動熔斷，避免持續打一個已經斷線的 API
_CIRCUIT_BREAKER_THRESHOLD: int = 3


class TWSEClient:
    """台股 API 客戶端。"""

    BASE_URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self._session = requests.Session()

        # ── Edge Case #40: 熔斷器狀態 ──
        # 當 FinMind API 連續失敗 _CIRCUIT_BREAKER_THRESHOLD 次後，
        # 設定 _circuit_open = True，後續請求直接回傳空 DataFrame，
        # 避免無效的重試消耗時間與 rate limit 配額。
        self._circuit_open: bool = False
        self._failure_count: int = 0

    def _clean_number(self, value) -> Optional[float]:
        """清理 FinMind API 回傳的數值。

        處理情境：
            - 字串帶逗號："1,234,567" → 1234567.0
            - 字串帶空白：" 123 " → 123.0
            - 空字串/None/"" → None
            - 已經是 int/float → 直接轉型

        ⚠️ 不得將 None/NaN 轉為 0.0（會汙染財報計算）
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            import math
            if math.isnan(value) or math.isinf(value):
                return None
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if not cleaned or cleaned == "-":
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _roc_to_ad(roc_year: int) -> int:
        """民國年轉西元年。

        範例：113 → 2024, 114 → 2025
        ⚠️ 若 roc_year > 1900，假設已經是西元年，直接回傳。
        """
        if roc_year > 1900:
            return roc_year
        return roc_year + 1911

    @staticmethod
    def _detect_encoding(raw_bytes: bytes) -> str:
        """偵測位元組的編碼。先試 UTF-8，失敗試 Big5。

        用途：MOPS 舊資料可能是 Big5 編碼。
        """
        try:
            raw_bytes.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            try:
                raw_bytes.decode("big5")
                return "big5"
            except UnicodeDecodeError:
                return "utf-8"  # fallback

    def _get_data(self, dataset: str, data_id: str, start_date: str) -> pd.DataFrame:
        """向 FinMind API 請求資料，內建熔斷機制 (Edge Case #40)。

        熔斷邏輯：
            - 每次請求失敗時 _failure_count += 1
            - 連續失敗達到 _CIRCUIT_BREAKER_THRESHOLD 次 → 開啟熔斷
            - 熔斷開啟後，所有請求直接回傳空 DataFrame，不再嘗試打 API
            - 任何一次成功 → 重置計數器並關閉熔斷
        """
        # ── 熔斷器檢查：若已熔斷，直接回傳空 DataFrame ──
        if self._circuit_open:
            logger.warning(
                "熔斷器已開啟（連續 %d 次失敗），跳過 FinMind API 請求: dataset=%s, data_id=%s",
                self._failure_count, dataset, data_id,
            )
            return pd.DataFrame()

        params = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
        }
        if self.api_token:
            params["token"] = self.api_token

        for attempt in range(3):
            try:
                resp = self._session.get(self.BASE_URL, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("msg") == "success" and data.get("data"):
                    # ── 成功：重置熔斷器 ──
                    self._failure_count = 0
                    self._circuit_open = False
                    return pd.DataFrame(data["data"])
                # 沒資料（API 正常回應但無資料，不算失敗）
                return pd.DataFrame()
            except Exception as e:
                logger.warning("FinMind request failed (attempt %d): %s", attempt + 1, e)
                self._failure_count += 1
                time.sleep(2)

        # ── 所有重試皆失敗：檢查是否應開啟熔斷器 ──
        if self._failure_count >= _CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open = True
            logger.error(
                "FinMind API 連續失敗 %d 次，啟動熔斷器。"
                "後續請求將直接回傳空資料，直到手動重置。",
                self._failure_count,
            )
        return pd.DataFrame()

    def get_latest_financials(self, ticker: str) -> Optional[FinancialStatement]:
        """取得最新一季財報。"""
        history = self.get_financials_history(ticker, n_quarters=1)
        return history[0] if history else None

    def get_financials_history(
        self, ticker: str, n_quarters: int = 8
    ) -> list[FinancialStatement]:
        """取得最近 N 季的財報。"""
        stock_id = TickerResolver.to_canonical(ticker).replace(".TW", "")

        # 抓取過去幾年的資料以涵蓋 n_quarters 以及 YoY 的前一年
        year_span = max(3, (n_quarters // 4) + 2)
        start_date = f"{datetime.now().year - year_span}-01-01"

        df = self._get_data("TaiwanStockFinancialStatements", stock_id, start_date)
        if df.empty:
            return []

        df["date"] = pd.to_datetime(df["date"])

        # 將原本 flat 的結構 pivot 成 column based
        # FinMind 的 type 包含: Revenue, GrossProfit, OperatingIncome, IncomeAfterTaxes 等
        pivot_df = df.pivot_table(
            index="date", columns="type", values="value", aggfunc="first"
        ).reset_index()
        pivot_df = pivot_df.sort_values("date", ascending=False)

        statements = []
        for _, row in pivot_df.iterrows():
            date_val = row["date"]

            def get_val(col_name: str):
                if col_name in row and pd.notna(row[col_name]):
                    return row[col_name]
                return None

            # ── FinMind IFRS 科目映射（完整版）──
            # 損益表
            rev = self._clean_number(get_val("Revenue"))
            gp = self._clean_number(get_val("GrossProfit"))
            op_inc = self._clean_number(get_val("OperatingIncome"))
            ni = self._clean_number(get_val("IncomeAfterTaxes"))
            eps = self._clean_number(get_val("EPS"))

            # 資產負債表
            assets = self._clean_number(get_val("TotalAssets"))
            liab = self._clean_number(get_val("TotalLiabilities"))
            equity = self._clean_number(get_val("Equity"))
            # 備用映射：有些公司用 EquityAttributableToOwnersOfParent
            if equity is None:
                equity = self._clean_number(get_val("EquityAttributableToOwnersOfParent"))
            cash = self._clean_number(get_val("CashAndCashEquivalents"))
            total_debt = self._clean_number(get_val("TotalNonCurrentLiabilities"))

            # 現金流量表
            op_cf = self._clean_number(get_val("CashFlowsFromOperatingActivities"))
            invest_cf = self._clean_number(get_val("CashProvidedByInvestingActivities"))
            # capex = 投資現金流的負值（FinMind 投資現金流通常為負數）
            capex = abs(invest_cf) if invest_cf is not None else None
            fcf = (op_cf - capex) if (op_cf is not None and capex is not None) else None

            quarter = (date_val.month - 1) // 3 + 1

            stmt = FinancialStatement(
                ticker=ticker,
                company_name=stock_id,
                market="TW",
                currency="TWD",
                stock_id=stock_id,
                filing_date=date_val.replace(tzinfo=timezone.utc),
                period_end=date_val.replace(tzinfo=timezone.utc),
                fiscal_year=date_val.year,
                fiscal_quarter=quarter,
                revenue=rev,
                gross_profit=gp,
                operating_income=op_inc,
                net_income=ni,
                eps_diluted=eps,
                total_assets=assets,
                total_liabilities=liab,
                total_equity=equity,
                cash_and_equivalents=cash,
                operating_cash_flow=op_cf,
                capex=capex,
                free_cash_flow=fcf,
                total_debt=total_debt,
            )

            # 計算衍生指標
            if equity and total_debt:
                stmt.debt_to_equity = total_debt / equity if abs(equity) > 1e-6 else None

            # ── Edge Case: KY 股幣別偵測 ──
            # KY 股（如 91APP-KY, 5765-KY）的財報幣別可能是 USD 或 CNY，
            # 不是 TWD。若不偵測，會把美金營收跟台幣營收混在一起比較。
            if "-KY" in ticker.upper() or "-KY" in stock_id.upper():
                # FinMind 的 KY 股財報通常仍以 TWD 呈現（已轉換），
                # 但若有 functional_currency 欄位則應採用。
                functional_currency = get_val("FunctionalCurrency")
                if functional_currency and str(functional_currency).upper() in ("USD", "CNY", "RMB"):
                    stmt.currency = str(functional_currency).upper()
                    logger.info("KY 股 %s 偵測到非台幣幣別: %s", ticker, stmt.currency)

            # 計算 margins
            if rev and gp:
                stmt.gross_margin = gp / rev
            if rev and op_inc:
                stmt.operating_margin = op_inc / rev

            statements.append(stmt)

        # 計算 YoY growth
        for i, stmt in enumerate(statements):
            if stmt.revenue is None:
                continue
            target_year = stmt.fiscal_year - 1
            target_quarter = stmt.fiscal_quarter
            for prev_stmt in statements[i + 1 :]:
                if prev_stmt.fiscal_year == target_year and prev_stmt.fiscal_quarter == target_quarter:
                    if prev_stmt.revenue is not None and abs(prev_stmt.revenue) > 1e-6:
                        stmt.revenue_yoy_growth = (
                            stmt.revenue - prev_stmt.revenue
                        ) / abs(prev_stmt.revenue)
                    break

        return statements[:n_quarters]

    def get_filing_text(
        self, ticker: str, filing_type: str = "年報", sections: list[str] | None = None
    ) -> dict[str, str]:
        """抓取台股財報或法說會相關文字。
        
        與 MOPSScraper 整合，若抓取失敗或 Timeout，優雅回傳空資料避免阻礙核心流程。
        """
        try:
            from .ticker_resolver import TickerResolver
            from .mops_scraper import MOPSScraper
            
            stock_id = TickerResolver.to_canonical(ticker).replace(".TW", "")
            scraper = MOPSScraper()
            
            text = scraper.get_investor_conference_text(stock_id)
            if text:
                return {"mda": text, "risk_factors": ""}
            else:
                return {"mda": "", "risk_factors": ""}
        except Exception as e:
            logger.warning("MOPS Unavailable [%s]: %s", ticker, e)
            return {"mda": "", "risk_factors": ""}

    def get_monthly_revenue(self, ticker: str, months: int = 12) -> pd.DataFrame:
        """台股專屬高頻動能指標：取得月營收。"""
        stock_id = TickerResolver.to_canonical(ticker).replace(".TW", "")
        start_year = datetime.now().year - (months // 12) - 1
        start_date = f"{start_year}-01-01"

        df = self._get_data("TaiwanStockMonthRevenue", stock_id, start_date)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        # 清理數值欄位（FinMind 月營收可能含逗號）
        for col in ["revenue", "revenue_month", "revenue_year"]:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_number)
        df = df.sort_values("date", ascending=False).head(months)
        return df

    def get_institutional_flow(self, ticker: str, days: int = 5) -> pd.DataFrame:
        """取得三大法人買賣超資料 (Smart Money Veto 用)。"""
        stock_id = TickerResolver.to_canonical(ticker).replace(".TW", "")
        start_date = (datetime.now() - pd.Timedelta(days=days*3)).strftime("%Y-%m-%d") # 多抓幾天確保有交易日
        
        df = self._get_data("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_date)
        if df.empty:
            return pd.DataFrame()
            
        df["date"] = pd.to_datetime(df["date"])
        # name 欄位有 '外資及陸資(不含外資自營商)', '投信' 等
        # 加總每日外資+投信賣超
        # 篩選外資與投信
        target_names = ["外資及陸資(不含外資自營商)", "外資及陸資", "投信"]
        df_filtered = df[df["name"].isin(target_names)]
        
        daily_flow = df_filtered.groupby("date")["buy_sell"].sum().reset_index()
        daily_flow = daily_flow.sort_values("date", ascending=False).head(days)
        return daily_flow

    # ── Edge Case #32: 集合競價期間偵測 ──
    # 台股在 13:25-13:30 為集合競價（收盤競價），此期間成交量與價格行為
    # 與盤中連續競價完全不同。若在此期間發送限價單，可能以非預期的價格成交。
    # 策略引擎應在此期間暫停或切換下單邏輯。
    @staticmethod
    def is_call_auction_period(dt: datetime) -> bool:
        """判斷指定時間是否在台股集合競價期間（13:25 ~ 13:30 台北時間）。

        Args:
            dt: 需要判斷的時間（必須帶有 timezone 資訊）

        Returns:
            True 表示在集合競價期間，策略應暫停或切換邏輯

        Raises:
            ValueError: 當 dt 不帶 timezone 資訊時拋出

        範例：
            >>> from datetime import datetime
            >>> from zoneinfo import ZoneInfo
            >>> tw_tz = ZoneInfo("Asia/Taipei")
            >>> dt_auction = datetime(2026, 6, 2, 13, 26, 0, tzinfo=tw_tz)
            >>> TWSEClient.is_call_auction_period(dt_auction)
            True
            >>> dt_normal = datetime(2026, 6, 2, 13, 0, 0, tzinfo=tw_tz)
            >>> TWSEClient.is_call_auction_period(dt_normal)
            False
        """
        if dt.tzinfo is None:
            raise ValueError(
                "datetime 必須帶有 timezone 資訊。"
                "請使用 datetime(..., tzinfo=ZoneInfo('Asia/Taipei'))。"
            )

        # 轉換為台北時間以進行比較
        tw_time = dt.astimezone(_TW_TZ)
        hour_minute = tw_time.hour * 100 + tw_time.minute  # 例：13:25 → 1325

        # 集合競價期間：13:25（含）到 13:30（不含）
        return 1325 <= hour_minute < 1330

    def reset_circuit_breaker(self) -> None:
        """手動重置熔斷器。

        當外部確認 FinMind API 已恢復時，呼叫此方法重新啟用 API 請求。
        """
        self._circuit_open = False
        self._failure_count = 0
        logger.info("熔斷器已手動重置，FinMind API 請求已恢復。")
