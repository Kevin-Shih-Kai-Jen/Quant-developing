import asyncio
import json
import os
from pprint import pprint
from dotenv import load_dotenv

from nexus_quant_os.alpha_hunter.signal_generator import AlphaSignalGenerator

async def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ 未找到 GEMINI_API_KEY，將跳過 AI 與供應鏈分析")
    
    print("🚀 初始化 Alpha Hunter Agent...")
    generator = AlphaSignalGenerator(
        gemini_api_key=api_key
    )
    
    ticker = "NVDA"
    print(f"\n🔍 正在掃描 {ticker}...")
    try:
        signal = generator.generate_single(ticker)
        
        print("\n✅ 掃描完成！分析結果：")
        print("="*50)
        result_dict = signal.to_dict()
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))
        print("="*50)
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(main())
