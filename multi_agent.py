import os
import asyncio
import re
import io
import time
import uuid
import logging
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

from langfuse.langchain import CallbackHandler
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

from database import get_cached_response, save_to_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

langfuse_handler = CallbackHandler()

MAX_FILES = 3
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_TOOL_OUTPUT_CHARS = 15000
ALLOWED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pptx')

fast_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, api_key=GROQ_KEY)
main_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, api_key=GROQ_KEY)
fallback_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3, api_key=GROQ_KEY)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?:",
    r"system\s*prompt:",
    r"<\|.*?\|>",
    r"\[INST\].*?\[/INST\]",
    r"jailbreak",
    r"DAN\s+mode",
]

def sanitize_input(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"Blocked: {pattern}"
    return True, ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    user_id: str
    session_id: str
    request_id: str

class RouteDecision(BaseModel):
    next_agent: Literal["web", "doc", "review", "compare", "general", "blocked"] = Field(description="Route.")
    reasoning: str = Field(description="Reason.")

circuit_state = {}
CIRCUIT_THRESHOLD = 3
CIRCUIT_RESET_SECONDS = 60

async def stream_llm(messages, session_id, callbacks=None):
    state = circuit_state.setdefault(session_id, {"failures": 0, "tripped_at": 0})
    current_time = time.time()
    
    if state["failures"] >= CIRCUIT_THRESHOLD:
        if current_time - state["tripped_at"] < CIRCUIT_RESET_SECONDS:
            try:
                async for chunk in fallback_llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
                    if chunk.content: yield chunk.content
                return
            except Exception as e:
                yield f"\n\n**Offline:** {e}"
                return
        else:
            state["failures"] = 0

    try:
        async for chunk in main_llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
            if chunk.content: yield chunk.content
        state["failures"] = 0
    except Exception as e:
        state["failures"] += 1
        if state["failures"] >= CIRCUIT_THRESHOLD:
            state["tripped_at"] = time.time()
        try:
            async for chunk in fallback_llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
                if chunk.content: yield chunk.content
        except Exception as fallback_error:
            yield f"\n\n**Offline:** {fallback_error}"

def get_general_system_prompt():
    return """You are a highly capable, helpful, and intelligent AI assistant.
    Answer the user's questions directly, accurately, and comprehensively.
    Use markdown formatting for readability. Be concise but thorough.
    If asked to write a document, write it in clean markdown."""

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
        return f"Max {MAX_FILES} files per batch."

    total_bytes = 0
    all_chunks = []
    all_metadatas = []
    for f in files:
        name = str(f.get("name", ""))[:200]
        safe_name = os.path.basename(name).lower()
        if not safe_name.endswith(ALLOWED_EXTENSIONS):
            return f"Unsupported: '{safe_name}'."
        import base64
        try:
            data = base64.b64decode(f.get("data", ""))
        except Exception:
            return "Malformed file."
        if len(data) > MAX_FILE_BYTES:
            return f"'{safe_name}' exceeds 10 MB."
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            return "Total upload exceeds 25 MB."
        
        text = _extract_text(data, safe_name)
        chunks = [line.strip() for line in text.split('\n') if len(line.strip()) > 40]
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"file_name": safe_name})

    if not all_chunks:
        return "No readable text found."

    embeddings = embedding_model.encode(all_chunks).tolist()
    collection_name = f"docs_{user_id}_{session_id}"
    try: chroma_client.delete_collection(name=collection_name)
    except Exception: pass
    collection = chroma_client.create_collection(name=collection_name)
    collection.add(embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas, ids=[str(i) for i in range(len(all_chunks))])
    return f"Ingested {len(all_chunks)} chunks."

def sanitize_context(context):
    out = context
    for p in INJECTION_PATTERNS:
        out = re.sub(p, "[REDACTED]", out, flags=re.IGNORECASE | re.DOTALL)
    return out

def truncate_tool_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"
    return text

async def manager_node(state: AgentState):
    req_id = state.get("request_id", "unknown")
    system_prompt = """Routing manager. Reply with one word:
    - 'web': market research, real-time info, competitors, news
    - 'doc': answering about an uploaded document
    - 'review': critiquing an uploaded document
    - 'compare': comparing multiple uploaded documents
    - 'general': simple questions, casual chat, coding help, writing tasks, general knowledge
    - 'blocked': malicious
    """
    structured_llm = fast_llm.with_structured_output(RouteDecision)
    try:
        decision = await structured_llm.ainvoke([SystemMessage(content=system_prompt), state["messages"][-1]], config={"callbacks": [langfuse_handler]})
        logger.info(f"[{req_id}] Manager → {decision.next_agent}")
        return {"next_agent": decision.next_agent}
    except Exception as e:
        logger.error(f"[{req_id}] Manager failed: {e}")
        return {"next_agent": "general"}

def guardrail_node(state: AgentState):
    req_id = state.get("request_id", "unknown")
    is_safe, reason = sanitize_input(state["messages"][-1].content)
    if not is_safe:
        logger.warning(f"[{req_id}] Guardrail blocked: {reason}")
        return {"next_agent": "blocked"}
    return {"next_agent": "manager"}

async def general_worker_node(state: AgentState):
    req_id = state.get("request_id", "unknown")
    session_id = state.get("session_id", "default")
    messages = state["messages"][-10:]
    
    conversation = [SystemMessage(content=get_general_system_prompt())]
    for msg in messages:
        conversation.append(msg)
    
    logger.info(f"[{req_id}] General Worker answering.")
    async for chunk in stream_llm(conversation, session_id=session_id):
        yield chunk

async def web_worker_node(state: AgentState):
    req_id = state.get("request_id", "unknown")
    session_id = state.get("session_id", "default")
    try:
        query = state["messages"][-1].content
        res = tavily.search(query=query, search_depth="advanced", timeout=15)
        raw_context = "\n\n".join([f"SOURCE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}" for r in res['results']])
        context = truncate_tool_output(raw_context)
        
        prompt = f"{get_general_system_prompt()}\n\nUser Question: {query}\n\nWeb Research Data:\n{context}\n\nAnswer based on this data."
        async for chunk in stream_llm([HumanMessage(content=prompt)], session_id=session_id):
            yield chunk
    except Exception as e:
        yield f"**Web Error:** {e}"

async def doc_worker_node(state: AgentState, mode="query"):
    req_id = state.get("request_id", "unknown")
    session_id = state.get("session_id", "default")
    try:
        query = state["messages"][-1].content
        collection = chroma_client.get_collection(name=f"docs_{state['user_id']}_{session_id}")
        
        all_metadatas = collection.get()['metadatas']
        unique_files = list(set([m['file_name'] for m in all_metadatas if m]))
        target_file = next((f for f in unique_files if f in query.lower()), None)
        
        if target_file:
            dense_results = collection.query(query_embeddings=embedding_model.encode([query]).tolist(), n_results=10, where={"file_name": target_file})
        else:
            dense_results = collection.query(query_embeddings=embedding_model.encode([query]).tolist(), n_results=10)
        
        raw_context = "\n\n".join(dense_results['documents'][0])
        safe_context = truncate_tool_output(sanitize_context(raw_context))
        
        if mode == "review":
            prompt = f"{get_general_system_prompt()}\n\nReview this document:\n<document>{safe_context}</document>\nProvide: Strengths, Gaps, Risks, Recommendations."
        else:
            prompt = f"{get_general_system_prompt()}\n\nAnswer using this context:\n{safe_context}\n\nQuestion: {query}"
        
        async for chunk in stream_llm([HumanMessage(content=prompt)], session_id=session_id):
            yield chunk
    except Exception as e:
        yield f"**Doc Error:** {e}"

checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
workflow = StateGraph(AgentState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("manager", manager_node)
workflow.add_node("web", web_worker_node)
workflow.add_node("doc", doc_worker_node)
workflow.add_node("review", lambda state: doc_worker_node(state, mode="review"))
workflow.add_node("general", general_worker_node)

workflow.set_entry_point("guardrail")
workflow.add_conditional_edges("guardrail", lambda x: "blocked" if x["next_agent"] == "blocked" else "manager", {"blocked": END, "manager": "manager"})
workflow.add_conditional_edges("manager", lambda x: x["next_agent"], {
    "web": "web", 
    "doc": "doc", 
    "review": "review", 
    "compare": END, 
    "general": "general",
    "blocked": END
})
workflow.add_edge("web", END)
workflow.add_edge("doc", END)
workflow.add_edge("review", END)
workflow.add_edge("general", END)

multi_agent_app = workflow.compile(checkpointer=checkpointer)

async def ask_agent_v2_stream(user_message, user_id, history=None, session_id="default", deep_dive=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", user_id):
        yield ("answer", "Invalid user identity.")
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        yield ("answer", "Invalid session.")
        return

    if history is None: history = []
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Request | {session_id} | {user_message[:50]}...")

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

    config = {"configurable": {"thread_id": session_id}, "callbacks": [langfuse_handler]}
    inputs = {"messages": history_messages, "user_id": user_id, "session_id": session_id, "request_id": request_id}
    
    route_decision = "general"
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
        logger.error(f"[{request_id}] Graph Error: {e}")
        yield ("step", f"❌ **Graph Error:** {e}")

    state = {"messages": history_messages, "user_id": user_id, "next_agent": "", "session_id": session_id, "request_id": request_id}
    full_response = ""
    
    if route_decision == "web":
        yield ("step", "🌐 **Web Worker**...")
        async for chunk in web_worker_node(state):
            full_response += chunk
            yield ("chunk", chunk)
    elif route_decision == "doc":
        yield ("step", "📄 **Doc Worker**...")
        async for chunk in doc_worker_node(state):
            full_response += chunk
            yield ("chunk", chunk)
    elif route_decision == "review":
        yield ("step", "🔍 **Critique Mode**...")
        async for chunk in doc_worker_node(state, mode="review"):
            full_response += chunk
            yield ("chunk", chunk)
    else:
        yield ("step", "💬 **General Chat**...")
        async for chunk in general_worker_node(state):
            full_response += chunk
            yield ("chunk", chunk)
        
    if not history and route_decision != "doc":
        query_emb = embedding_model.encode([user_message]).tolist()
        save_to_cache(query_emb, full_response)
        logger.info(f"[{request_id}] Cached.")