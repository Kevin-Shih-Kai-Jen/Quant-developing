import asyncio
import httpx
import json

async def main():
    payload = {
        "tickers": [{"ticker": "ORCL"}],
        "center_ticker": "ORCL",
        "edges": []
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream('POST', 'http://127.0.0.1:8000/api/alpha/supply-chain-radar/scan', json=payload) as response:
                async for chunk in response.aiter_text():
                    print(chunk, end='')
        except Exception as e:
            print("Client error:", e)

if __name__ == "__main__":
    asyncio.run(main())
