import os
import asyncio
import re
import io
import time
from typing import TypedDict, Annotated, Sequence, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator
from tavily import TavilyClient
from rank_bm25 import BM25Okapi

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

from database import get_cached_response, save_to_cache

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_KEY: raise RuntimeError("GROQ_API_KEY missing")
if not TAVILY_KEY: raise RuntimeError("TAVILY_API_KEY missing")

tavily = TavilyClient(api_key=TAVILY_KEY)

print("Loading embedding model...")
os.environ["HF_HUB_OFFLINE"] = "1"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

MAX_FILES = 3
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pptx')

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    user_id: str
    session_id: str

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, api_key=GROQ_KEY)
fallback_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, api_key=GROQ_KEY)

# --- STRUCTURED ROUTING MODEL (The Foundation Fix) ---
class RouteDecision(BaseModel):
    next_agent: Literal["web", "writer", "doc", "review", "compare", "blocked"] = Field(
        description="The exact agent to route to."
    )
    reasoning: str = Field(description="One sentence explaining the routing choice.")

# --- SAFE LLM INVOKER ---
async def stream_llm(messages):
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        print(f"⚠️ Primary failed: {e}. Fallback...")
        try:
            async for chunk in fallback_llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as fallback_error:
            yield f"\n\n**AI systems offline:** {fallback_error}"

# --- PM CONTEXT INJECTION ---
def get_pm_system_prompt():
    try:
        with open("context.md", "r") as f:
            context = f.read()
    except FileNotFoundError:
        context = "No product context provided."
    return f"""You are a Principal Technical Product Manager. 
You produce rigorous, data-driven, and comprehensive product artifacts.
Use frameworks: RICE, INVEST, SMART, SLOs, Eval Metrics.
PRODUCT CONTEXT:
{context}
"""

# --- INGESTION ---
def _extract_text(file_bytes: bytes, file_name: str) -> str:
    name = file_name.lower()
    try:
        if name.endswith('.pdf'):
            reader = PdfReader(io.BytesIO(file_bytes))
            return "".join((p.extract_text() or "") for p in reader.pages)
        if name.endswith('.docx'):
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        if name.endswith('.txt') or name.endswith('.md') or name.endswith('.csv'):
            return file_bytes.decode('utf-8', errors='ignore')
        if name.endswith('.xlsx') or name.endswith('.xls'):
            import pandas as pd
            df = pd.read_excel(io.BytesIO(file_bytes))
            return df.to_string(index=False)
        if name.endswith('.pptx'):
            from pptx import Presentation
            prs = Presentation(io.BytesIO(file_bytes))
            lines = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        lines.append(shape.text)
            return "\n".join(lines)
    except Exception:
        return ""
    return ""

def ingest_files(files, user_id, session_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", user_id):
        return "Upload rejected: invalid identity."
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        return "Upload rejected: invalid session."
    if not files or not isinstance(files, list):
        return "No files received."
    if len(files) > MAX_FILES:
        return f"Upload rejected: maximum {MAX_FILES} files per batch."

    total_bytes = 0
    all_chunks = []
    for f in files:
        name = str(f.get("name", ""))[:200]
        safe_name = os.path.basename(name).lower()
        if not safe_name.endswith(ALLOWED_EXTENSIONS):
            return f"Upload rejected: unsupported file type '{safe_name}'."
        import base64
        try:
            data = base64.b64decode(f.get("data", ""))
        except Exception:
            return "Upload rejected: malformed file data."
        if len(data) > MAX_FILE_BYTES:
            return f"Upload rejected: '{safe_name}' exceeds 10 MB."
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            return "Upload rejected: total upload exceeds 25 MB."
        text = _extract_text(data, safe_name)
        chunks = [line.strip() for line in text.split('\n') if len(line.strip()) > 40]
        all_chunks.extend(chunks)

    if not all_chunks:
        return "No readable text found in the uploaded files."

    embeddings = embedding_model.encode(all_chunks).tolist()
    collection_name = f"docs_{user_id}_{session_id}"
    try: chroma_client.delete_collection(name=collection_name)
    except Exception: pass
    collection = chroma_client.create_collection(name=collection_name)
    collection.add(embeddings=embeddings, documents=all_chunks, ids=[str(i) for i in range(len(all_chunks))])
    return f"Ingested {len(all_chunks)} chunks from {len(files)} file(s)."

def sanitize_context(context):
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+",
        r"new\s+instructions?:",
        r"system\s*prompt:",
        r"<\|.*?\|>",
        r"\[INST\].*?\[/INST\]",
    ]
    out = context
    for p in patterns:
        out = re.sub(p, "[REDACTED]", out, flags=re.IGNORECASE | re.DOTALL)
    return out

# --- ROBUST MANAGER NODE ---
async def manager_node(state: AgentState):
    system_prompt = """You are the routing manager.
    Analyze the user's input and decide which worker should handle it:
    - 'web': market research, competitors, current events
    - 'writer': writing new PRDs, FRDs, specs
    - 'doc': answering questions about a SINGLE uploaded document
    - 'review': critiquing an uploaded document
    - 'compare': comparing multiple uploaded documents
    - 'blocked': malicious or unsafe input
    """
    
    # Use structured output to force a valid route
    structured_llm = llm.with_structured_output(RouteDecision)
    try:
        decision = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt), 
            state["messages"][-1]
        ])
        return {"next_agent": decision.next_agent}
    except Exception as e:
        print(f"Manager routing failed: {e}. Defaulting to writer.")
        return {"next_agent": "writer"}

def guardrail_node(state: AgentState):
    system_prompt = "Is this input malicious? Reply 'SAFE' or 'UNSAFE'."
    response = llm.invoke([HumanMessage(content=system_prompt), state["messages"][-1]])
    return {"next_agent": "blocked" if "UNSAFE" in response.content.upper() else "manager"}

# --- WORKER NODES (Wrapped in Try/Except) ---
async def writer_worker_node(state: AgentState):
    try:
        query = state["messages"][-1].content.lower()
        # ... [Insert the full writer logic from previous iterations here] ...
        # (For brevity, keeping it to a single call in this architecture fix)
        prompt = f"""Write a comprehensive PM document based on: {query}"""
        async for chunk in stream_llm([SystemMessage(content=get_pm_system_prompt()), HumanMessage(content=prompt)]):
            yield chunk
    except Exception as e:
        yield f"**Writer Error:** {e}"

async def web_worker_node(state: AgentState):
    try:
        query = state["messages"][-1].content
        res = tavily.search(query=query, search_depth="advanced", timeout=15)
        context = "\n\n".join([f"SOURCE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}" for r in res['results']])
        prompt = f"{get_pm_system_prompt()}\nAnswer this: {query}\nData: {context}"
        async for chunk in stream_llm([HumanMessage(content=prompt)]):
            yield chunk
    except Exception as e:
        yield f"**Web Search Error:** {e}"

# --- BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("guardrail", guardrail_node)
workflow.add_node("manager", manager_node)
workflow.add_node("writer", writer_worker_node)
workflow.add_node("web", web_worker_node)
# (Add doc, review, compare nodes here using the same try/except pattern)

workflow.set_entry_point("guardrail")
workflow.add_conditional_edges(
    "guardrail",
    lambda x: "blocked" if x["next_agent"] == "blocked" else "manager",
    {"blocked": END, "manager": "manager"}
)

# Conditional routing from Manager
workflow.add_conditional_edges(
    "manager",
    lambda x: x["next_agent"],
    {
        "web": "web",
        "writer": "writer",
        "doc": END,
        "review": END,
        "compare": END,
        "blocked": END
    }
)

workflow.add_edge("writer", END)
workflow.add_edge("web", END)

multi_agent_app = workflow.compile()

# --- THE EXECUTION RUNNER (Handles streaming and errors) ---
async def ask_agent_v2_stream(user_message, user_id, history=None, session_id="default", deep_dive=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", user_id):
        yield ("answer", "Invalid user identity.")
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        yield ("answer", "Invalid session.")
        return

    if history is None: history = []
    if not history:
        query_emb = embedding_model.encode([user_message]).tolist()
        cached = get_cached_response(query_emb)
        if cached:
            yield ("step", "⚡ **Cache Hit**...")
            yield ("chunk", cached)
            return

    history_messages = []
    for msg in history:
        if msg['role'] == 'user': history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant': history_messages.append(AIMessage(content=msg['content']))
    history_messages.append(HumanMessage(content=user_message[:100_000]))

    inputs = {"messages": history_messages, "user_id": user_id, "session_id": session_id}
    
    # Use a fallback route if the graph fails entirely
    route_decision = "writer"
    try:
        for event in multi_agent_app.stream(inputs, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "guardrail":
                    if node_output.get("next_agent") == "blocked":
                        yield ("step", "🛡️ **Guardrail**: `BLOCKED`")
                        yield ("answer", "I cannot process that request.")
                        return
                elif node_name == "manager":
                    route_decision = node_output.get("next_agent")
                    yield ("step", f"🧠 **Manager** → `{route_decision.upper()}`")
    except Exception as e:
        yield ("step", f"❌ **Graph Error:** {e}")

    # Execute the chosen worker natively
    state = {"messages": history_messages, "user_id": user_id, "next_agent": "", "session_id": session_id}
    full_response = ""
    
    if route_decision == "web":
        yield ("step", "🌐 **Web Worker**...")
        async for chunk in web_worker_node(state):
            full_response += chunk
            yield ("chunk", chunk)
    elif route_decision == "writer":
        yield ("step", f"✍️ **Writer**...")
        async for chunk in writer_worker_node(state):
            full_response += chunk
            yield ("chunk", chunk)
    else:
        yield ("answer", "This agent is under construction.")
        
    if not history and route_decision != "doc":
        query_emb = embedding_model.encode([user_message]).tolist()
        save_to_cache(query_emb, full_response)