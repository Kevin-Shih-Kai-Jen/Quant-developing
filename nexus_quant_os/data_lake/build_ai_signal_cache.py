import os
import json
import logging
import sqlite3
import argparse
import random
import polars as pl
from datetime import datetime
from typing import List, Dict, Any

from nexus_quant_os.core.models import OfflineAISignal, DummyTextProvider
from nexus_quant_os.alpha_hunter.ai_analyst import AIAnalyst

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OfflineCruncher")

import pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DB_PATH = str(BASE_DIR / "cache_manifest.db")
OUTPUT_PARQUET_PATH = str(BASE_DIR / "Alpha_Signal_Matrix.parquet")

class QuantScreenerFilter:
    """協定一：量化漏斗海選"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_core_universe(self, date: str) -> List[str]:
        # 實務上會使用 Polars 掃描毛利率等指標
        # 這裡為了測試，回傳核心候選股
        return ["2330.TW", "NVDA", "3231.TW", "3017.TW", "1449.TW"]

class StateMachineDB:
    """協定三：冪等性與狀態機"""
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_signals (
                    ticker TEXT,
                    publish_date TEXT,
                    status TEXT,
                    response_json TEXT,
                    PRIMARY KEY (ticker, publish_date)
                )
            """)

    def upsert_task(self, ticker: str, publish_date: str, status: str, response_json: str = None):
        with self.conn:
            self.conn.execute("""
                INSERT INTO ai_signals (ticker, publish_date, status, response_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker, publish_date) DO UPDATE SET
                    status=excluded.status,
                    response_json=excluded.response_json
            """, (ticker, publish_date, status, response_json))

    def get_status(self, ticker: str, publish_date: str) -> str:
        cur = self.conn.execute("SELECT status FROM ai_signals WHERE ticker=? AND publish_date=?", (ticker, publish_date))
        row = cur.fetchone()
        return row['status'] if row else None
        
    def get_all_completed(self) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM ai_signals WHERE status='COMPLETED'")
        return [dict(r) for r in cur.fetchall()]

class MockBatchResponseGenerator:
    """測試用的假 Batch API 回應產生器"""
    def process(self, request_file: str) -> str:
        response_file = request_file.replace("requests.jsonl", "responses.jsonl")
        with open(request_file, 'r', encoding='utf-8') as fin, open(response_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                req = json.loads(line)
                custom_id = req.get("custom_id", "")
                
                # 模擬 5% 機率失敗或幻覺 (NaN 傳染測試)
                if random.random() < 0.05:
                    fake_res = {
                        "custom_id": custom_id,
                        "error": "Simulated Batch Error or High Entropy"
                    }
                else:
                    fake_res = {
                        "custom_id": custom_id,
                        "response": {
                            "body": {
                                "turnaround_signal": random.choice([True, False]),
                                "capex_expansion_confidence": round(random.uniform(0.1, 0.9), 2),
                                "management_tone_shift": round(random.uniform(-0.8, 0.8), 2),
                                "ai_score": round(random.uniform(0.1, 0.9), 2),
                                "information_entropy": round(random.uniform(0.1, 0.5), 2)
                            }
                        }
                    }
                fout.write(json.dumps(fake_res) + "\n")
        return response_file

class AnomalyImputer:
    """協定十六：靜默缺口與 NaN 傳染防禦"""
    @staticmethod
    def get_neutral_penalty() -> dict:
        return {
            "turnaround_signal": False,
            "capex_expansion_confidence": 0.0,
            "management_tone_shift": 0.0,
            "ai_score": 0.5,
            "information_entropy": 1.0
        }

def build_offline_cache(dry_run: bool = False):
    logger.info("Starting Offline LLM Cruncher (Phase 12 ETL)")
    
    db = StateMachineDB(CACHE_DB_PATH)
    screener = QuantScreenerFilter(data_dir=str(BASE_DIR / "parquet"))
    text_provider = DummyTextProvider()
    ai_analyst = AIAnalyst() # 僅用來借用 _mask_entity
    
    # 這裡假設每季底發布財報，我們模擬 10 年 40 季
    years = range(2014, 2025)
    quarters = ["03-31", "05-15", "08-14", "11-14"] # 簡化發布日
    
    requests_data = []
    
    # 準備 Tasks
    logger.info("Scanning for tasks and generating context...")
    for y in years:
        for q in quarters:
            publish_date = f"{y}-{q}"
            universe = screener.get_core_universe(publish_date)
            
            for ticker in universe:
                status = db.get_status(ticker, publish_date)
                if status == "COMPLETED":
                    continue
                    
                # 協定四：Context Pruning (這裡用 Mock)
                raw_text = text_provider.get_filing_text(ticker, publish_date)
                
                # 實體盲化
                masked_text = ai_analyst._mask_entity(raw_text, ticker)
                
                # 協定十七：Lexical Drift Bias 補救 (Time-Aware Anchoring)
                prompt = f'''Current evaluation year is {y}. Evaluate the company's innovation and capex ONLY relative to the technological context and macro environment of {y}.
Extract features and return strictly following the schema.

--- MD&A ---
{masked_text[:5000]}
'''
                
                custom_id = f"{ticker}_{publish_date}"
                req_obj = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1beta/models/gemini-1.5-pro:generateContent",
                    "body": {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "response_mime_type": "application/json",
                            # 協定五：強制結構化輸出
                            "response_schema": OfflineAISignal.model_json_schema() if hasattr(OfflineAISignal, "model_json_schema") else OfflineAISignal.schema()
                        }
                    }
                }
                requests_data.append(req_obj)
                db.upsert_task(ticker, publish_date, "PENDING_BATCH")
                
    if not requests_data:
        logger.info("All tasks are already COMPLETED.")
        return
        
    # 寫入 JSONL
    req_file = str(BASE_DIR / "batch_requests.jsonl")
    with open(req_file, 'w', encoding='utf-8') as f:
        for req in requests_data:
            f.write(json.dumps(req) + "\n")
            
    logger.info(f"Generated {len(requests_data)} batch tasks to {req_file}")
    
    if dry_run:
        logger.info("[Dry-Run] Invoking MockBatchResponseGenerator...")
        generator = MockBatchResponseGenerator()
        resp_file = generator.process(req_file)
        
        # 模擬 Parse Response
        logger.info("Parsing responses and verifying state machine transitions...")
        with open(resp_file, 'r', encoding='utf-8') as f:
            for line in f:
                res = json.loads(line)
                custom_id = res['custom_id']
                ticker, publish_date = custom_id.split('_', 1)
                
                # 異常處理 (NaN Contagion Prevention)
                if "error" in res or res.get("response", {}).get("body", {}).get("information_entropy", 0.0) > 0.8:
                    logger.warning(f"Anomaly detected for {custom_id}. Applying Imputer.")
                    final_stats = AnomalyImputer.get_neutral_penalty()
                else:
                    final_stats = res["response"]["body"]
                    
                db.upsert_task(ticker, publish_date, "COMPLETED", json.dumps(final_stats))
                
        logger.info("Database state transition completed.")
        
    # 匯出 Parquet
    logger.info(f"Exporting Cache DB to Parquet...")
    records = db.get_all_completed()
    
    if records:
        df_records = []
        for r in records:
            stats = json.loads(r['response_json']) if r['response_json'] else AnomalyImputer.get_neutral_penalty()
            row = {
                "ticker": r['ticker'],
                "publish_date": r['publish_date'],
                "turnaround_signal": stats.get('turnaround_signal', False),
                "capex_expansion_confidence": stats.get('capex_expansion_confidence', 0.0),
                "management_tone_shift": stats.get('management_tone_shift', 0.0),
                "ai_score": stats.get('ai_score', 0.5),
                "information_entropy": stats.get('information_entropy', 1.0)
            }
            df_records.append(row)
            
        df = pl.DataFrame(df_records)
        df.write_parquet(OUTPUT_PARQUET_PATH)
        logger.info(f"Successfully exported {len(df_records)} AI signal records to {OUTPUT_PARQUET_PATH}")
    else:
        logger.warning("No COMPLETED records found to export.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use MockBatchResponseGenerator")
    args = parser.parse_args()
    
    build_offline_cache(dry_run=args.dry_run)
