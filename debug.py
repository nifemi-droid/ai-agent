import asyncio
import traceback
from multi_agent import ask_agent_v2_stream

async def main():
    print("=== Starting debug run ===")
    try:
        async for kind, content in ask_agent_v2_stream("What is the capital of France?", "local"):
            print(f"[{kind}] {content[:200]}")
        print("=== Finished cleanly ===")
    except Exception as e:
        print(f"=== CRASHED: {type(e).__name__}: {e} ===")
        traceback.print_exc()

asyncio.run(main())