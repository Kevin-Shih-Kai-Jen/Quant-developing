"""
alpha_hunter/radar.py — 供應鏈 Alpha 雷達 (v2.5)
實作 Graph-RAG, Momentum Spillover 等進階網路分析演算法。
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any

from nexus_quant_os.alpha_hunter.models import EnrichedNode

# 為了能在 asyncio 裡面直接 call，我們會把 dependency injection 交給 server.py 或者直接引入

logger = logging.getLogger("alpha_hunter.radar")


def compute_degree_centrality(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, float]:
    """計算 Degree Centrality."""
    degree_count = defaultdict(int)
    all_tickers = {n["ticker"] for n in nodes}
    for e in edges:
        degree_count[e["source_ticker"]] += 1
        degree_count[e["target_ticker"]] += 1
    
    n = len(all_tickers)
    # 【防禦機制】除以零炸彈：如果圖中只有 1 家公司，分母 n-1 為 0 會導致 ZeroDivisionError
    if n <= 1:
        return {t: 1.0 for t in all_tickers}
    
    return {t: degree_count.get(t, 0) / (n - 1) for t in all_tickers}


def compute_momentum_spillover(enriched_nodes: Dict[str, EnrichedNode], edges: List[Dict[str, Any]], max_iterations: int = 3) -> Dict[str, float]:
    """計算動能傳染."""
    # 【防禦機制】Null 污染：外部 API 失敗時 composite_score 為 None，運算會拋 TypeError
    # 預設給予 0.5 (中性分數) 確保數學公式能運作
    scores = {t: (n.composite_score if n.composite_score is not None else 0.5) for t, n in enriched_nodes.items()}
    supplier_to_customers = defaultdict(list)
    
    for e in edges:
        # 【防禦機制】營收暴走：LLM 幻覺可能給出 revenue_pct="5.0" (500%) 導致分數無限放大
        try:
            rev_pct = max(0.0, min(float(e.get("revenue_pct") or 0.1), 1.0))
        except (ValueError, TypeError):
            rev_pct = 0.1
            
        if e.get("relation") == "SUPPLIER":
            supplier_to_customers[e["source_ticker"]].append((e["target_ticker"], rev_pct))
        elif e.get("relation") == "CUSTOMER":
            supplier_to_customers[e["target_ticker"]].append((e["source_ticker"], rev_pct))
            
    network_scores = dict(scores)
    
    # 【防禦機制】無限月讀：A 是 B 客戶，B 也是 A 客戶。靠 max_iterations=3 強制截斷
    for _ in range(max_iterations):
        delta = {}
        for supplier, customers in supplier_to_customers.items():
            spillover = sum((scores.get(c, 0.5) - 0.5) * pct for c, pct in customers)
            delta[supplier] = 0.5 * spillover
            
        changed = False
        for t, d in delta.items():
            # 【防禦機制】幽靈節點：edge 可能參照 enriched_nodes 之外的 ticker
            current = network_scores.get(t, 0.5)
            # 【防禦機制】分數破表：負能量傳染可能把分數扣到 -0.2，必須嚴格鉗制在 [0, 1]
            new_score = max(0.0, min(1.0, current + d))
            
            # 【防禦機制】浮點數 Epsilon 空轉：使用 0.001 容錯判斷收斂
            if abs(new_score - current) > 0.001:
                changed = True
            network_scores[t] = new_score
            
        if not changed: break
        
    for t in enriched_nodes:
        enriched_nodes[t].network_alpha_score = network_scores.get(t, 0.5)
        enriched_nodes[t].spillover_delta = network_scores.get(t, 0.5) - scores.get(t, 0.5)
    return network_scores


async def generate_graph_rag_analysis(advisor, center: str, enriched_nodes: Dict[str, EnrichedNode], edges: List[Dict[str, Any]], scores: Dict[str, float]) -> str:
    """Graph-RAG 總結."""
    # 【防禦機制】Token Context Window 撐爆：超過 30 個節點轉 Markdown 會被 LLM 拒收 (HTTP 400)
    if len(enriched_nodes) > 30:
        top_tickers = sorted(scores, key=scores.get, reverse=True)[:30]
        enriched_nodes = {t: enriched_nodes[t] for t in top_tickers}
        edges = [e for e in edges if e["source_ticker"] in top_tickers and e["target_ticker"] in top_tickers]
        truncation_warning = "\n*(註：網路過於龐大，已為您截斷並僅分析 Top 30 核心受惠節點)*\n"
    else:
        truncation_warning = ""

    nodes_table = "| Ticker | Company Name | Network Alpha Score | Composite Score | Centrality | Signal |\n|---|---|---|---|---|---|\n"
    for ticker, node in enriched_nodes.items():
        # 【防禦機制】Prompt 注入：清洗公司名稱中的 Markdown 符號與換行，防止格式錯亂
        safe_name = node.company_name.replace("`", "").replace("\n", " ") if node.company_name else "Unknown"
        nodes_table += f"| {ticker} | {safe_name} | {node.network_alpha_score:.2f} | {node.composite_score if node.composite_score is not None else 0.5:.2f} | {node.degree_centrality:.2f} | {node.signal_strength} |\n"
    
    edges_text = "供應鏈關係:\n"
    for e in edges:
        edges_text += f"- {e.get('source_ticker')} -> {e.get('target_ticker')} (Relation: {e.get('relation')}, Rev %: {e.get('revenue_pct')})\n"

    prompt = f"""你是一位管理 50 億美元的對沖基金經理。你會收到「供應鏈網路圖譜」的結構化資料。
## 絕對禁止
- 不要描述圖表的外觀形狀（不要說什麼同心圓）
- 不要使用模糊語句如 "OK" 或 "建議進一步研究"
- 只能輸出以下三個標題（且務必包含對應的 emoji，嚴格符合格式）：

### 🏆 產業咽喉
[指出 Degree Centrality 最高、被最多公司依賴的節點，解釋其定價權]

### 💎 隱藏的 Alpha — Top 3 買入推薦
| 排名 | Ticker | Network Alpha Score | 買入理由 |
|---|---|---|---|
| 1 | TICKER | SCORE | 具體理由：自身估值 + 下游客戶動能傳染 |

### ⚠️ 斷鏈與衰退風險
[列出 composite_score < 0.3 的公司，警告下游可能面臨的砍單風險。無則寫 "目前無重大斷鏈風險"。]

## 資料
中心節點：{center}
{nodes_table}

{edges_text}
"""
    
    try:
        response = advisor.chat(prompt)
    except Exception as e:
        logger.error(f"Graph-RAG LLM error: {e}")
        return f"## ⚠️ AI 分析失敗\n無法生成分析結果。錯誤：{e}\n{truncation_warning}"
    
    # 【防禦機制】傲嬌 AI 叛逆：如果 AI 不輸出指定格式
    if "🏆" not in response and "💎" not in response:
        return f"## ⚠️ AI 分析格式異常\n請直接參考圖表中綠色（高分）與半徑最大（咽喉）的節點。\n\n原始輸出：\n{response}\n{truncation_warning}"
        
    return response + truncation_warning

def node_to_dict(node: EnrichedNode) -> dict:
    return {
        "ticker": node.ticker,
        "depth": node.depth,
        "llm_source": node.llm_source,
        "composite_score": node.composite_score,
        "signal_strength": node.signal_strength,
        "valuation_rating": node.valuation_rating,
        "fundamental_pass": node.fundamental_pass,
        "ai_bullish": node.ai_bullish,
        "technical_confirm": node.technical_confirm,
        "degree_centrality": node.degree_centrality,
        "network_alpha_score": node.network_alpha_score,
        "spillover_delta": node.spillover_delta,
        "scan_status": node.scan_status,
        "top_suppliers": node.top_suppliers,
        "top_customers": node.top_customers,
        "company_name": node.company_name,
    }
