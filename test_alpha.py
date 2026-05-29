"""Alpha Hunter 完整掃描測試"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from nexus_quant_os.alpha_hunter.signal_generator import AlphaSignalGenerator
from nexus_quant_os.alpha_hunter._constants import DEFAULT_SCAN_UNIVERSE

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    print("🚀 Alpha Hunter Agent — 完整掃描")
    print(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"🔑 Gemini API: {'✅ 已設定' if api_key else '⚠️ 未設定（AI/供應鏈分析將跳過）'}")
    print(f"📊 掃描範圍: {len(DEFAULT_SCAN_UNIVERSE)} 檔股票")
    print("=" * 80)

    t0 = time.time()
    generator = AlphaSignalGenerator(gemini_api_key=api_key)

    signals = generator.generate_signals(
        include_supply_chain=False,
        include_ai_analysis=False,  # 避免 429 rate limit
    )
    elapsed = time.time() - t0

    print(f"\n✅ 掃描完成！耗時 {elapsed:.1f} 秒")
    print(f"   候選標的: {len(signals)} / {len(DEFAULT_SCAN_UNIVERSE)}")
    print()

    # 分類
    strong_buy = [s for s in signals if s.signal_strength.value == "STRONG_BUY"]
    buy = [s for s in signals if s.signal_strength.value == "BUY"]
    neutral = [s for s in signals if s.signal_strength.value == "NEUTRAL"]

    def print_signal(s):
        sc = s.scan_result
        st = sc.latest_statement if sc else None
        rev = f"${st.revenue/1e9:.1f}B" if st and st.revenue else "N/A"
        yoy = f"{st.revenue_yoy_growth*100:.0f}%" if st and st.revenue_yoy_growth else "N/A"
        gm  = f"{st.gross_margin*100:.0f}%" if st and st.gross_margin else "N/A"
        eps = f"${st.eps_diluted:.2f}" if st and st.eps_diluted else "N/A"
        de  = f"{st.debt_to_equity:.2f}" if st and st.debt_to_equity else "N/A"
        val = sc.valuation_rating.value if sc else "N/A"
        tech = "📈" if s.technical_confirm else "📉"
        checks = "✅" * s.confirmation_count + "⬜" * (4 - s.confirmation_count)
        print(f"  {s.ticker:<6s} | {s.company_name[:20]:<20s} | {checks} | Score: {s.composite_score:.2f} | Rev: {rev:>8s} YoY: {yoy:>5s} | GM: {gm:>4s} | EPS: {eps:>6s} | D/E: {de:>5s} | {tech} | {val}")

    if strong_buy:
        print(f"🟢 STRONG BUY ({len(strong_buy)})")
        print(f"  {'Ticker':<6s} | {'Company':<20s} | Conf | Score | {'Revenue':>8s} {'YoY':>5s} | {'GM':>4s} | {'EPS':>6s} | {'D/E':>5s} | Tech | Valuation")
        print("  " + "-" * 120)
        for s in strong_buy:
            print_signal(s)
        print()

    if buy:
        print(f"🔵 BUY ({len(buy)})")
        print(f"  {'Ticker':<6s} | {'Company':<20s} | Conf | Score | {'Revenue':>8s} {'YoY':>5s} | {'GM':>4s} | {'EPS':>6s} | {'D/E':>5s} | Tech | Valuation")
        print("  " + "-" * 120)
        for s in buy:
            print_signal(s)
        print()

    if neutral:
        print(f"⚪ NEUTRAL ({len(neutral)})")
        print(f"  {'Ticker':<6s} | {'Company':<20s} | Conf | Score | {'Revenue':>8s} {'YoY':>5s} | {'GM':>4s} | {'EPS':>6s} | {'D/E':>5s} | Tech | Valuation")
        print("  " + "-" * 120)
        for s in neutral:
            print_signal(s)
        print()

    # 排除的股票
    avoid = [s for s in signals if s.signal_strength.value == "AVOID"]
    if avoid:
        print(f"🔴 AVOID ({len(avoid)}): {', '.join(s.ticker for s in avoid)}")

    print("=" * 80)
    print(f"📋 掃描摘要: STRONG_BUY={len(strong_buy)} | BUY={len(buy)} | NEUTRAL={len(neutral)} | AVOID={len(avoid)}")

if __name__ == "__main__":
    main()
