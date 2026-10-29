import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

print("=== 1. Checking environment ===")
print(f"GROQ_API_KEY set: {bool(os.getenv('GROQ_API_KEY'))}")
print(f"TAVILY_API_KEY set: {bool(os.getenv('TAVILY_API_KEY'))}")

print("\n=== 2. Importing multi_agent ===")
try:
    from multi_agent import ask_agent_v2_stream, ingest_files
    print("✅ Import OK")
except Exception as e:
    print(f"❌ Import failed: {type(e).__name__}: {e}")
    raise

print("\n=== 3. Calling the agent ===")
async def test():
    try:
        steps = []
        final = None
        async for kind, content in ask_agent_v2_stream("What is 2+2?", "local"):
            print(f"  [{kind}] {content[:100]}")
            steps.append(kind)
            if kind == "answer":
                final = content
        print(f"\n✅ Agent returned. Steps: {steps}")
        print(f"Final answer: {final}")
    except Exception as e:
        print(f"❌ Agent call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test())