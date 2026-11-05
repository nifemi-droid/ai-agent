import asyncio
import time
from multi_agent import ask_agent_v2_stream

# THE GOLDEN DATASET
# These are the strict criteria we expect the bot to meet.
GOLDEN_TESTS = [
    {
        "name": "PRD Generation",
        "prompt": "Write a PRD for a mobile push notification system.",
        "required_sections": ["Problem", "Goals", "User Stories", "Success Metrics", "Risks"],
        "max_latency": 20.0
    },
    {
        "name": "AI/ML Spec",
        "prompt": "Write an AI spec for a RAG-based customer support bot.",
        "required_sections": ["Model Choice", "Evaluation Plan", "Guardrails", "Cost Model"],
        "max_latency": 25.0
    },
    {
        "name": "Platform Spec",
        "prompt": "Write a platform spec for an internal API gateway.",
        "required_sections": ["API Contracts", "SLOs", "Build vs Buy", "Scalability"],
        "max_latency": 20.0
    },
    {
        "name": "Web Search (Agentic)",
        "prompt": "What are the latest trends in AI Product Management for 2026?",
        "required_sections": ["Summary", "Key Findings", "Implications"],
        "max_latency": 30.0
    }
]

async def run_eval():
    print("🚀 Starting Golden Dataset Evaluation...\n")
    passed = 0
    total = len(GOLDEN_TESTS)
    
    for test in GOLDEN_TESTS:
        print(f"--- Testing: {test['name']} ---")
        print(f"Prompt: {test['prompt']}")
        
        start_time = time.time()
        full_response = ""
        
        try:
            # We pass a dummy session ID for testing
            async for kind, content in ask_agent_v2_stream(test['prompt'], "eval_user", [], "eval_session"):
                if kind == "chunk":
                    full_response += content
                    
            latency = time.time() - start_time
            
            # 1. Check Latency
            latency_pass = latency <= test['max_latency']
            print(f"⏱️ Latency: {latency:.2f}s (Max: {test['max_latency']}s) -> {'✅' if latency_pass else '❌'}")
            
            # 2. Check Required Sections
            missing_sections = []
            for section in test['required_sections']:
                if section.lower() not in full_response.lower():
                    missing_sections.append(section)
            
            sections_pass = len(missing_sections) == 0
            if sections_pass:
                print(f"📝 Sections: All {len(test['required_sections'])} required sections found. -> ✅")
            else:
                print(f"📝 Sections: Missing {missing_sections} -> ❌")
                
            # 3. Final Verdict
            if latency_pass and sections_pass:
                print("🏆 PASS\n")
                passed += 1
            else:
                print("❌ FAIL\n")
                
        except Exception as e:
            print(f"❌ CRASHED: {e}\n")
            
    print(f"=====================================")
    print(f"FINAL SCORE: {passed}/{total} tests passed.")
    print(f"=====================================")

if __name__ == "__main__":
    asyncio.run(run_eval())