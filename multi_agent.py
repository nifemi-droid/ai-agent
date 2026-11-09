import os
import asyncio
import re
import io
import time
from typing import TypedDict, Annotated, Sequence, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import operator
from tavily import TavilyClient
from rank_bm25 import BM25Okapi

# CORRECTED IMPORT FOR LANGFUSE V3+
from langfuse.langchain import CallbackHandler

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

# --- LANGFUSE OBSERVABILITY ---
langfuse_handler = CallbackHandler()

MAX_FILES = 3
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pptx')

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    user_id: str
    session_id: str

class RouteDecision(BaseModel):
    next_agent: Literal["web", "writer", "doc", "review", "compare", "blocked"] = Field(
        description="The exact agent to route to."
    )
    reasoning: str = Field(description="One sentence explaining the routing choice.")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, api_key=GROQ_KEY)
fallback_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, api_key=GROQ_KEY)

# --- CIRCUIT BREAKER STATE ---
primary_failures = 0
circuit_tripped_at = 0
CIRCUIT_THRESHOLD = 3
CIRCUIT_RESET_SECONDS = 60

async def stream_llm(messages, callbacks=None):
    global primary_failures, circuit_tripped_at
    
    current_time = time.time()
    
    # Check if circuit is tripped
    if primary_failures >= CIRCUIT_THRESHOLD:
        if current_time - circuit_tripped_at < CIRCUIT_RESET_SECONDS:
            print(f"🔌 Circuit breaker is OPEN. Routing directly to fallback (resets in {CIRCUIT_RESET_SECONDS - (current_time - circuit_tripped_at):.0f}s).")
            try:
                async for chunk in fallback_llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
                    if chunk.content:
                        yield chunk.content
                return
            except Exception as e:
                yield f"\n\n**All AI systems offline:** {e}"
                return
        else:
            # Reset the circuit after timeout
            print("🔌 Circuit breaker resetting. Trying primary...")
            primary_failures = 0

    # Try primary
    try:
        async for chunk in llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
            if chunk.content:
                yield chunk.content
        # Success! Reset failure count
        primary_failures = 0
    except Exception as e:
        primary_failures += 1
        if primary_failures >= CIRCUIT_THRESHOLD:
            circuit_tripped_at = time.time()
            print(f"🚨 Primary model failed {primary_failures} times. Circuit breaker tripped.")
        
        print(f"⚠️ Primary failed: {e}. Trying fallback...")
        try:
            async for chunk in fallback_llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
                if chunk.content:
                    yield chunk.content
        except Exception as fallback_error:
            yield f"\n\n**All AI systems offline:** {fallback_error}"

def get_pm_system_prompt():
    try:
        with open("context.md", "r") as f:
            context = f.read()
    except FileNotFoundError:
        context = "No product context provided."
    return f"""You are a Principal Technical Product Manager. 
You produce rigorous, data-driven, and comprehensive product artifacts.
PRODUCT CONTEXT:
{context}
"""

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
    patterns = [r"ignore\s+(all\s+)?previous\s+instructions", r"ignore\s+(all\s+)?above\s+instructions", r"disregard\s+(all\s+)?previous", r"forget\s+(all\s+)?previous", r"you\s+are\s+now\s+", r"new\s+instructions?:", r"system\s*prompt:", r"<\|.*?\|>", r"\[INST\].*?\[/INST\]"]
    out = context
    for p in patterns:
        out = re.sub(p, "[REDACTED]", out, flags=re.IGNORECASE | re.DOTALL)
    return out

# --- CONTEXT COMPRESSION ---
async def compress_history(history):
    if len(history) < 10:
        return history
        
    print("🗜️ Compressing conversation history...")
    recent = history[-4:]
    old = history[:-4]
    
    summary_prompt = f"""
    Summarize this past conversation into a concise 'handoff summary'. 
    Include: Main Goal, Key Decisions, Constraints, and Next Steps.
    
    Conversation:
    {old}
    """
    
    try:
        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
        summary_text = response.content
        summary_message = AIMessage(content=f"[Previous conversation summary]: {summary_text}")
        return [{"role": "assistant", "content": summary_message.content}] + recent
    except Exception as e:
        print(f"Compression failed: {e}. Using raw history.")
        return history

# --- MANAGER NODE (Structured Output) ---
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
    structured_llm = llm.with_structured_output(RouteDecision)
    try:
        decision = await structured_llm.ainvoke([SystemMessage(content=system_prompt), state["messages"][-1]], config={"callbacks": [langfuse_handler]})
        return {"next_agent": decision.next_agent}
    except Exception as e:
        print(f"Manager routing failed: {e}. Defaulting to writer.")
        return {"next_agent": "writer"}

def guardrail_node(state: AgentState):
    system_prompt = "Is this input malicious? Reply 'SAFE' or 'UNSAFE'."
    response = llm.invoke([HumanMessage(content=system_prompt), state["messages"][-1]], config={"callbacks": [langfuse_handler]})
    return {"next_agent": "blocked" if "UNSAFE" in response.content.upper() else "manager"}

# --- WORKER NODES (Wrapped in Try/Except) ---
async def writer_worker_node(state: AgentState):
    try:
        query = state["messages"][-1].content.lower()
        prompt = f"Write a comprehensive PM document based on: {query}"
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

# --- THE ENTERPRISE GRAPH (With Checkpointer) ---
checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")

workflow = StateGraph(AgentState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("manager", manager_node)
workflow.add_node("writer", writer_worker_node)
workflow.add_node("web", web_worker_node)

workflow.set_entry_point("guardrail")
workflow.add_conditional_edges("guardrail", lambda x: "blocked" if x["next_agent"] == "blocked" else "manager", {"blocked": END, "manager": "manager"})
workflow.add_conditional_edges("manager", lambda x: x["next_agent"], {"web": "web", "writer": "writer", "doc": END, "review": END, "compare": END, "blocked": END})
workflow.add_edge("writer", END)
workflow.add_edge("web", END)

multi_agent_app = workflow.compile(checkpointer=checkpointer)

# --- THE EXECUTION RUNNER ---
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

    # Compress history if it's getting too long
    compressed_history = await compress_history(history)
    
    history_messages = []
    for msg in compressed_history:
        if msg['role'] == 'user': history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant': history_messages.append(AIMessage(content=msg['content']))
    history_messages.append(HumanMessage(content=user_message[:100_000]))

    config = {"configurable": {"thread_id": session_id}, "callbacks": [langfuse_handler]}
    inputs = {"messages": history_messages, "user_id": user_id, "session_id": session_id}
    
    route_decision = "writer"
    try:
        for event in multi_agent_app.stream(inputs, config, stream_mode="updates"):
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