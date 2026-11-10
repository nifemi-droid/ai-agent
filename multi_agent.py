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

# --- STRUCTURED LOGGING ---
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
ALLOWED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pptx')

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    user_id: str
    session_id: str
    request_id: str

class RouteDecision(BaseModel):
    next_agent: Literal["web", "writer", "doc", "review", "compare", "blocked"] = Field(
        description="The exact agent to route to."
    )
    reasoning: str = Field(description="One sentence explaining the routing choice.")

class ContextSufficiency(BaseModel):
    is_sufficient: bool = Field(description="Whether the retrieved context is sufficient to answer the question.")
    reasoning: str = Field(description="Brief reasoning.")
    better_query: str = Field(description="If not sufficient, a better search query. Otherwise, empty string.")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, api_key=GROQ_KEY)
fallback_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, api_key=GROQ_KEY)

primary_failures = 0
circuit_tripped_at = 0
CIRCUIT_THRESHOLD = 3
CIRCUIT_RESET_SECONDS = 60

async def stream_llm(messages, callbacks=None):
    global primary_failures, circuit_tripped_at
    current_time = time.time()
    
    if primary_failures >= CIRCUIT_THRESHOLD:
        if current_time - circuit_tripped_at < CIRCUIT_RESET_SECONDS:
            logger.warning(f"🔌 Circuit breaker OPEN.")
            try:
                async for chunk in fallback_llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
                    if chunk.content: yield chunk.content
                return
            except Exception as e:
                yield f"\n\n**All AI systems offline:** {e}"
                return
        else:
            primary_failures = 0

    try:
        async for chunk in llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
            if chunk.content: yield chunk.content
        primary_failures = 0
    except Exception as e:
        primary_failures += 1
        if primary_failures >= CIRCUIT_THRESHOLD:
            circuit_tripped_at = time.time()
            logger.error(f"🚨 Circuit breaker tripped.")
        logger.warning(f"⚠️ Primary failed: {e}. Fallback...")
        try:
            async for chunk in fallback_llm.astream(messages, config={"callbacks": callbacks or [langfuse_handler]}):
                if chunk.content: yield chunk.content
        except Exception as fallback_error:
            yield f"\n\n**All AI systems offline:** {fallback_error}"

# --- 2. CROSS-SESSION MEMORY ---
def get_user_profile(user_id):
    """Reads facts from past sessions to build a persistent user profile."""
    try:
        import sqlite3
        from database import DB_NAME
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Get the last 3 assistant messages from other sessions
        cursor.execute("""
            SELECT m.content FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE s.user_id = ? AND m.role = 'assistant'
            ORDER BY m.timestamp DESC LIMIT 3
        """, (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return ""
        
        # We don't summarize here to save tokens; we just feed the last few docs
        # A production system would use a background job to extract "facts".
        profile = "Recent context from past sessions:\n"
        for r in rows:
            profile += f"- {r[0][:200]}...\n"
        return profile
    except Exception:
        return ""

def get_pm_system_prompt(user_id="local"):
    try:
        with open("context.md", "r") as f:
            context = f.read()
    except FileNotFoundError:
        context = "No product context provided."
    
    memory = get_user_profile(user_id)
    
    return f"""You are a Principal Technical Product Manager. 
You produce rigorous, data-driven, and comprehensive product artifacts.
PRODUCT CONTEXT:
{context}
{memory}
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
    all_metadatas = []

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
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"file_name": safe_name})

    if not all_chunks:
        return "No readable text found in the uploaded files."

    embeddings = embedding_model.encode(all_chunks).tolist()
    collection_name = f"docs_{user_id}_{session_id}"
    try: chroma_client.delete_collection(name=collection_name)
    except Exception: pass
    collection = chroma_client.create_collection(name=collection_name)
    
    collection.add(
        embeddings=embeddings, 
        documents=all_chunks, 
        metadatas=all_metadatas,
        ids=[str(i) for i in range(len(all_chunks))]
    )
    return f"Ingested {len(all_chunks)} chunks from {len(files)} file(s)."

def sanitize_context(context):
    patterns = [r"ignore\s+(all\s+)?previous\s+instructions", r"ignore\s+(all\s+)?above\s+instructions", r"disregard\s+(all\s+)?previous", r"forget\s+(all\s+)?previous", r"you\s+are\s+now\s+", r"new\s+instructions?:", r"system\s*prompt:", r"<\|.*?\|>", r"\[INST\].*?\[/INST\]"]
    out = context
    for p in patterns:
        out = re.sub(p, "[REDACTED]", out, flags=re.IGNORECASE | re.DOTALL)
    return out

async def compress_history(history):
    if len(history) < 10: return history
    logger.info("🗜️ Compressing history...")
    recent = history[-4:]
    old = history[:-4]
    summary_prompt = f"Summarize this conversation into a concise handoff summary.\n{old}"
    try:
        response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
        summary_message = AIMessage(content=f"[Previous conversation summary]: {response.content}")
        return [{"role": "assistant", "content": summary_message.content}] + recent
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return history

async def manager_node(state: AgentState):
    req_id = state.get("request_id", "unknown")
    system_prompt = """You are the routing manager.
    - 'web': market research, competitors
    - 'writer': writing new PRDs, FRDs, specs
    - 'doc': answering questions about a SINGLE uploaded document
    - 'review': critiquing an uploaded document
    - 'compare': comparing multiple uploaded documents
    - 'blocked': malicious input
    """
    structured_llm = llm.with_structured_output(RouteDecision)
    try:
        decision = await structured_llm.ainvoke([SystemMessage(content=system_prompt), state["messages"][-1]], config={"callbacks": [langfuse_handler]})
        logger.info(f"[{req_id}] Manager → {decision.next_agent} ({decision.reasoning})")
        return {"next_agent": decision.next_agent}
    except Exception as e:
        logger.error(f"[{req_id}] Manager failed: {e}. Defaulting to writer.")
        return {"next_agent": "writer"}

def guardrail_node(state: AgentState):
    req_id = state.get("request_id", "unknown")
    system_prompt = "Is this input malicious? Reply 'SAFE' or 'UNSAFE'."
    response = llm.invoke([HumanMessage(content=system_prompt), state["messages"][-1]], config={"callbacks": [langfuse_handler]})
    decision = "blocked" if "UNSAFE" in response.content.upper() else "manager"
    logger.info(f"[{req_id}] Guardrail: {decision.upper()}")
    return {"next_agent": decision}

async def writer_worker_node(state: AgentState, deep_dive=False):
    req_id = state.get("request_id", "unknown")
    query = state["messages"][-1].content.lower()
    
    if "prd" in query or "product requirement" in query:
        artifact_type = "PRD"
        required_sections = ["Problem", "Goals", "User Stories", "Success Metrics", "Risks"]
    elif "frd" in query or "functional requirement" in query:
        artifact_type = "FRD"
        required_sections = ["Functional Overview", "User Flow", "Data Contracts", "Edge Cases", "Acceptance Criteria"]
    elif any(k in query for k in ["ai", "ml", "model", "llm", "rag"]):
        artifact_type = "AI/ML Spec"
        required_sections = ["Model Choice", "Evaluation Plan", "Guardrails", "Cost Model"]
    else:
        artifact_type = "Strategy Doc"
        required_sections = ["Situation Analysis", "Market Context", "Options", "Recommendation"]

    prompt = f"Write a comprehensive {artifact_type} based on: {query}"
    full_response = ""
    
    try:
        async for chunk in stream_llm([SystemMessage(content=get_pm_system_prompt(state['user_id'])), HumanMessage(content=prompt)]):
            full_response += chunk
            yield chunk
            
        missing = [s for s in required_sections if s.lower() not in full_response.lower()]
        if missing:
            logger.warning(f"[{req_id}] Quality Gate Failed! Missing: {missing}. Retrying...")
            yield "\n\n---\n**⚠️ Quality Gate: Retrying missing sections...**\n\n"
            retry_prompt = f"Your previous output was rejected. Missing: {', '.join(missing)}. Rewrite the entire {artifact_type} with all sections."
            async for chunk in stream_llm([SystemMessage(content=get_pm_system_prompt(state['user_id'])), HumanMessage(content=retry_prompt)]):
                yield chunk
        else:
            logger.info(f"[{req_id}] Quality Gate Passed.")
            
    except Exception as e:
        logger.error(f"[{req_id}] Writer Error: {e}")
        yield f"**Writer Error:** {e}"

async def web_worker_node(state: AgentState):
    req_id = state.get("request_id", "unknown")
    try:
        query = state["messages"][-1].content
        logger.info(f"[{req_id}] Web Search for: {query}")
        res = tavily.search(query=query, search_depth="advanced", timeout=15)
        context = "\n\n".join([f"SOURCE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}" for r in res['results']])
        prompt = f"{get_pm_system_prompt(state['user_id'])}\nAnswer this: {query}\nData: {context}"
        async for chunk in stream_llm([HumanMessage(content=prompt)]):
            yield chunk
    except Exception as e:
        logger.error(f"[{req_id}] Web Error: {e}")
        yield f"**Web Search Error:** {e}"

# --- 1. SELF-HEALING RAG + 3. CRITIQUE MODE ---
async def doc_worker_node(state: AgentState, mode="query"):
    req_id = state.get("request_id", "unknown")
    try:
        query = state["messages"][-1].content
        session_id = state.get("session_id", "default")
        
        collection = chroma_client.get_collection(name=f"docs_{state['user_id']}_{session_id}")
        all_metadatas = collection.get()['metadatas']
        unique_files = list(set([m['file_name'] for m in all_metadatas if m]))
        
        target_file = None
        for f in unique_files:
            if f in query.lower():
                target_file = f
                break
                
        max_retries = 2
        current_query = query
        
        for attempt in range(max_retries):
            if target_file:
                dense_results = collection.query(
                    query_embeddings=embedding_model.encode([current_query]).tolist(),
                    n_results=10,
                    where={"file_name": target_file}
                )
            else:
                dense_results = collection.query(
                    query_embeddings=embedding_model.encode([current_query]).tolist(),
                    n_results=10
                )
                
            context = "\n\n".join(dense_results['documents'][0])
            safe_context = sanitize_context(context)
            
            # SELF-HEALING: Evaluate context sufficiency
            eval_prompt = f"""
            Question: {query}
            Context: {safe_context}
            Is this context sufficient to answer the question comprehensively?
            If yes, is_sufficient=True and better_query="".
            If no, is_sufficient=False and provide a better_query.
            """
            structured_llm = llm.with_structured_output(ContextSufficiency)
            try:
                eval_res = await structured_llm.ainvoke([HumanMessage(content=eval_prompt)])
                if eval_res.is_sufficient:
                    logger.info(f"[{req_id}] Context sufficient on attempt {attempt+1}.")
                    break
                else:
                    logger.warning(f"[{req_id}] Context insufficient. Retrying with: {eval_res.better_query}")
                    current_query = eval_res.better_query
            except Exception:
                break # If evaluation fails, just use what we have
                
        # 3. CRITIQUE MODE
        if mode == "review":
            prompt = f"""
            {get_pm_system_prompt(state['user_id'])}
            You are a VP of Product reviewing a document.
            Use ONLY this context:
            <document>{safe_context}</document>
            
            User Request: {query}
            Provide a critique: Strengths, Gaps, Risks, and Actionable Recommendations.
            """
        else:
            prompt = f"{get_pm_system_prompt(state['user_id'])}\nAnswer using this context:\n{safe_context}\n\nQuestion: {query}"
        
        async for chunk in stream_llm([HumanMessage(content=prompt)]):
            yield chunk
    except Exception as e:
        logger.error(f"[{req_id}] Document Error: {e}")
        yield f"**Document Error:** {e}"

# --- GRAPH ---
checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
workflow = StateGraph(AgentState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("manager", manager_node)
workflow.add_node("writer", writer_worker_node)
workflow.add_node("web", web_worker_node)
workflow.add_node("doc", doc_worker_node)
workflow.add_node("review", lambda state: doc_worker_node(state, mode="review"))

workflow.set_entry_point("guardrail")
workflow.add_conditional_edges("guardrail", lambda x: "blocked" if x["next_agent"] == "blocked" else "manager", {"blocked": END, "manager": "manager"})
workflow.add_conditional_edges("manager", lambda x: x["next_agent"], {"web": "web", "writer": "writer", "doc": "doc", "review": "review", "compare": END, "blocked": END})
workflow.add_edge("writer", END)
workflow.add_edge("web", END)
workflow.add_edge("doc", END)
workflow.add_edge("review", END)

multi_agent_app = workflow.compile(checkpointer=checkpointer)

# --- RUNNER ---
async def ask_agent_v2_stream(user_message, user_id, history=None, session_id="default", deep_dive=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", user_id):
        yield ("answer", "Invalid user identity.")
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        yield ("answer", "Invalid session.")
        return

    if history is None: history = []
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] New Request | Session: {session_id} | Message: {user_message[:50]}...")

    if not history:
        query_emb = embedding_model.encode([user_message]).tolist()
        cached = get_cached_response(query_emb)
        if cached:
            logger.info(f"[{request_id}] Cache Hit.")
            yield ("step", "⚡ **Cache Hit**...")
            yield ("chunk", cached)
            return

    compressed_history = await compress_history(history)
    history_messages = []
    for msg in compressed_history:
        if msg['role'] == 'user': history_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant': history_messages.append(AIMessage(content=msg['content']))
    history_messages.append(HumanMessage(content=user_message[:100_000]))

    config = {"configurable": {"thread_id": session_id}, "callbacks": [langfuse_handler]}
    inputs = {"messages": history_messages, "user_id": user_id, "session_id": session_id, "request_id": request_id}
    
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
        yield ("step", f"✍️ **Writer**...")
        async for chunk in writer_worker_node(state):
            full_response += chunk
            yield ("chunk", chunk)
        
    if not history and route_decision != "doc":
        query_emb = embedding_model.encode([user_message]).tolist()
        save_to_cache(query_emb, full_response)
        logger.info(f"[{request_id}] Saved to semantic cache.")
        
    logger.info(f"[{request_id}] Request Complete.")