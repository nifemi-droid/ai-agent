import os
import asyncio
import re
import io
import time
import uuid
import json
import logging
from typing import TypedDict, Annotated, Sequence, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
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
MAX_TOOL_OUTPUT_CHARS = 8000
MAX_CONTEXT_TOKENS = 6000
ALLOWED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pptx')

fast_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, api_key=GROQ_KEY)
main_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, api_key=GROQ_KEY)
fallback_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3, api_key=GROQ_KEY)

# --- SECURITY: INJECTION BLOCKLIST ---
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

# --- 1. DETERMINISTIC PRE-ROUTER ---
def pre_router(message: str, has_files: bool) -> str:
    """Route deterministically before the LLM is even called."""
    msg = message.lower().strip()
    
    if msg.startswith("/web") or "search the web" in msg or "latest news" in msg:
        return "web"
    if has_files:
        return "doc"
    if msg.startswith("/doc") or "uploaded document" in msg:
        return "doc"
    
    return "auto"

# --- 2. NATIVE TOOL SCHEMAS ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information, market research, competitors, or news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the user's uploaded documents for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "file_name": {"type": "string", "description": "Optional: specific file to search"}
                },
                "required": ["query"]
            }
        }
    }
]

# --- TOOL EXECUTORS ---
async def execute_web_search(query: str) -> str:
    try:
        res = tavily.search(query=query, search_depth="advanced", timeout=15)
        raw = "\n\n".join([f"SOURCE: {r['title']}\nURL: {r['url']}\nCONTENT: {r['content']}" for r in res['results']])
        return re.sub(r'<[^>]+>', '', raw)[:MAX_TOOL_OUTPUT_CHARS]
    except Exception as e:
        return f"Web search failed: {e}"

def execute_search_documents(query: str, user_id: str, session_id: str, file_name: str = None) -> str:
    try:
        collection = chroma_client.get_collection(name=f"docs_{user_id}_{session_id}")
        if file_name:
            results = collection.query(query_embeddings=embedding_model.encode([query]).tolist(), n_results=10, where={"file_name": file_name})
        else:
            results = collection.query(query_embeddings=embedding_model.encode([query]).tolist(), n_results=10)
        return "\n\n".join(results['documents'][0])[:MAX_TOOL_OUTPUT_CHARS]
    except Exception as e:
        return f"Document search failed: {e}"

# --- 3. TOKEN-BOUNDED CONTEXT ---
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4

def compress_history(history: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> list:
    """Drop oldest messages until under the token limit."""
    total_tokens = sum(estimate_tokens(m['content']) for m in history)
    
    if total_tokens <= max_tokens:
        return history
    
    logger.info(f"🗜️ Context too large ({total_tokens} tokens). Compressing...")
    # Keep the most recent messages, drop from the front
    compressed = history[:]
    while estimate_tokens_compressed(compressed) > max_tokens and len(compressed) > 2:
        compressed.pop(0)
    
    return compressed

def estimate_tokens_compressed(history: list) -> int:
    return sum(estimate_tokens(m['content']) for m in history)

# --- AGENT LOOP WITH NATIVE TOOL CALLING ---
async def run_agent(user_message: str, user_id: str, session_id: str, history: list, pre_route: str = "auto") -> str:
    # 3. COMPRESS HISTORY BEFORE SENDING
    compressed_history = compress_history(history)
    
    messages = [
        SystemMessage(content="""You are a helpful, intelligent AI assistant.
        Use the provided tools when needed:
        - web_search: for real-time info, news, or competitors
        - search_documents: for questions about the user's uploaded files
        
        Answer directly when no tools are needed. Be concise but thorough. Use markdown formatting.""")
    ]
    
    for msg in compressed_history[-10:]:
        messages.append(HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content']))
    
    messages.append(HumanMessage(content=user_message))
    
    max_tool_calls = 3
    tool_call_count = 0
    used_tool = False
    
    while tool_call_count < max_tool_calls:
        try:
            # 1. PRE-ROUTER: If deterministic route, force tool on first call
            if tool_call_count == 0 and pre_route == "web":
                response = await main_llm.ainvoke(messages, tools=[tools[0]], tool_choice={"type": "function", "function": {"name": "web_search"}}, config={"callbacks": [langfuse_handler]})
            elif tool_call_count == 0 and pre_route == "doc":
                response = await main_llm.ainvoke(messages, tools=[tools[1]], tool_choice={"type": "function", "function": {"name": "search_documents"}}, config={"callbacks": [langfuse_handler]})
            else:
                response = await main_llm.ainvoke(messages, tools=tools, config={"callbacks": [langfuse_handler]})
        except Exception as e:
            logger.error(f"LLM error: {e}. Trying fallback...")
            try:
                response = await fallback_llm.ainvoke(messages, tools=tools, config={"callbacks": [langfuse_handler]})
            except Exception as fallback_error:
                return f"**AI systems offline:** {fallback_error}"
        
        if not response.tool_calls:
            return response.content
        
        tool_call_count += 1
        used_tool = True
        messages.append(response)
        
        for tool_call in response.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            logger.info(f"🔧 Tool call: {fn_name}({args})")
            
            if fn_name == "web_search":
                result = await execute_web_search(args["query"])
            elif fn_name == "search_documents":
                result = execute_search_documents(args["query"], user_id, session_id, args.get("file_name"))
            else:
                result = "Unknown tool."
            
            messages.append(ToolMessage(content=result, tool_call_id=tool_call.id))
    
    return "Max tool calls reached. Please rephrase."

# --- FILE INGESTION ---
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

# --- MAIN ENTRY POINT ---
async def ask_agent_v2_stream(user_message, user_id, history=None, session_id="default", deep_dive=False, has_files=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", user_id):
        yield ("answer", "Invalid user identity.")
        return
    
    if history is None: history = []
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Request | {session_id} | {user_message[:50]}...")
    
    # SECURITY: Guardrail check
    is_safe, reason = sanitize_input(user_message)
    if not is_safe:
        logger.warning(f"[{request_id}] Blocked: {reason}")
        yield ("step", "🛡️ **Guardrail**: `BLOCKED`")
        yield ("chunk", "I cannot process that request due to safety guidelines.")
        return
    
    # CACHE: Check semantic cache
    if not history:
        query_emb = embedding_model.encode([user_message]).tolist()
        cached = get_cached_response(query_emb)
        if cached:
            yield ("step", "⚡ **Cache Hit**...")
            yield ("chunk", cached)
            return
    
    # 1. DETERMINISTIC PRE-ROUTER
    pre_route = pre_router(user_message, has_files)
    yield ("step", f"🧭 **Pre-Router**: `{pre_route.upper()}`")
    
    # RUN AGENT
    yield ("step", "🤖 **Agent thinking...**")
    
    try:
        response = await run_agent(user_message, user_id, session_id, history, pre_route)
        yield ("chunk", response)
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        yield ("chunk", f"**Error:** {e}")
        return
    
    # CACHE: Save to cache
    if not history:
        query_emb = embedding_model.encode([user_message]).tolist()
        save_to_cache(query_emb, response)
        logger.info(f"[{request_id}] Cached.")