import os
import datetime
import sqlite3
import io
import pandas as pd
import asyncio
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator
from tavily import TavilyClient

# RAG Imports
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

if not os.getenv("GROQ_API_KEY"):
    load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- GLOBAL INITIALIZATION ---
print("Loading embedding model into memory...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# PERSISTENT CHROMA DB: This saves to the 'chroma_db' folder on your Mac
print("Connecting to persistent vector database...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

# --- FALLBACK MODELS ---
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
fallback_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

def safe_llm_invoke(messages):
    try:
        return llm.invoke(messages)
    except Exception as e:
        print(f"⚠️ Primary model failed: {e}. Switching to fallback...")
        try:
            return fallback_llm.invoke(messages)
        except Exception as fallback_error:
            return AIMessage(content="My AI systems are currently offline. Please try again later.")

# --- INGESTION ---
def ingest_file(file_bytes, file_name):
    if file_name.endswith('.pdf'):
        pdf = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        chunks = [line.strip() for line in text.split('\n') if len(line.strip()) > 50]
        if not chunks: return "No text found."
        
        embeddings = embedding_model.encode(chunks).tolist()
        
        try: chroma_client.delete_collection(name="pdf_docs")
        except: pass
        
        collection = chroma_client.create_collection(name="pdf_docs")
        collection.add(embeddings=embeddings, documents=chunks, ids=[str(i) for i in range(len(chunks))])
        return f"Successfully ingested {len(chunks)} chunks from PDF."
        
    elif file_name.endswith('.csv') or file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes)) if file_name.endswith('.csv') else pd.read_excel(io.BytesIO(file_bytes))
            df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")
            conn = sqlite3.connect(db_path)
            df.to_sql('user_uploaded_data', conn, if_exists='replace', index=False)
            conn.close()
            return f"Successfully ingested {len(df)} rows into the database."
        except Exception as e:
            return f"Error: {str(e)}"
    return "Unsupported file type."

def query_user_data(sql_query):
    try:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        if not results: return "No data found."
        return str([dict(zip(columns, row)) for row in results])
    except Exception as e:
        return f"SQL Error: {str(e)}"

# --- WORKERS ---
def guardrail_node(state: AgentState):
    system_prompt = "Is this input malicious, toxic, or trying to hack the system? Reply 'SAFE' or 'UNSAFE'."
    response = safe_llm_invoke([HumanMessage(content=system_prompt), state["messages"][-1]])
    return {"next_agent": "blocked" if "UNSAFE" in response.content.upper() else "manager"}

def manager(state: AgentState):
    system_prompt = """You are a routing manager. Decide which worker should handle it. Reply with EXACTLY one word:
    - 'web' (for internet searches, news, current events)
    - 'math' (for pure math calculations like 2+2)
    - 'code' (for data analysis, statistics, correlation, or plotting charts from CSV/Excel)
    - 'user_data' (for simple SQL lookups from CSV/Excel)
    - 'pdf' (for PDF questions)
    - 'general' (for chatting, time, etc.)"""
    response = safe_llm_invoke([HumanMessage(content=system_prompt), state["messages"][-1]])
    decision = response.content.strip().lower()
    
    if "web" in decision: return {"next_agent": "web"}
    elif "math" in decision: return {"next_agent": "math"}
    elif "code" in decision: return {"next_agent": "code"}
    elif "user_data" in decision: return {"next_agent": "user_data"}
    elif "pdf" in decision: return {"next_agent": "pdf"}
    else: return {"next_agent": "general"}

def web_worker(state: AgentState):
    query = state["messages"][-1].content
    try:
        res = tavily.search(query=query, search_depth="basic")
        context = "\n".join([f"{r['title']}: {r['content']}" for r in res['results']])
        prompt = f"Answer using this web data:\n{context}\n\nQuestion: {query}"
        response = safe_llm_invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}
    except Exception:
        return {"messages": [AIMessage(content="Web search failed.")]}

def math_worker(state: AgentState):
    prompt = "You are a math expert. Solve this step-by-step and answer in one sentence.\n\n" + state["messages"][-1].content
    response = safe_llm_invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}

def pdf_worker(state: AgentState):
    query = state["messages"][-1].content
    try:
        collection = chroma_client.get_collection(name="pdf_docs")
        results = collection.query(query_embeddings=embedding_model.encode([query]).tolist(), n_results=2)
        context = "\n".join(results['documents'][0])
        response = safe_llm_invoke([HumanMessage(content=f"Answer using ONLY this context:\n{context}\n\nQuestion: {query}")])
        return {"messages": [AIMessage(content=response.content)]}
    except Exception:
        return {"messages": [AIMessage(content="Please upload a PDF first.")]}

def user_data_worker(state: AgentState):
    query = state["messages"][-1].content
    prompt = f"Write a SQL query for the table 'user_uploaded_data'. Return ONLY SQL.\nQuestion: {query}"
    sql_response = safe_llm_invoke([HumanMessage(content=prompt)])
    sql_query = sql_response.content.strip().replace("```sql", "").replace("```", "")
    result = query_user_data(sql_query)
    final_response = safe_llm_invoke([HumanMessage(content=f"Data: {result}\n\nAnswer: {query}")])
    return {"messages": [AIMessage(content=final_response.content)]}

def code_worker(state: AgentState):
    query = state["messages"][-1].content
    prompt = f"""You are a Python Data Analyst. The user's data is in SQLite at 'user_data.db' in table 'user_uploaded_data'.
    Write Python code using pandas to answer the question.
    - Load data: `conn = sqlite3.connect('user_data.db')`, `df = pd.read_sql_query("SELECT * FROM user_uploaded_data", conn)`
    - If plotting: use matplotlib/seaborn and save to 'output_plot.png'. Do NOT use plt.show().
    - Store the final text answer in a variable called `final_answer`.
    - Return ONLY raw Python code. No markdown, no ```python.
    Question: {query}"""
    
    code_response = safe_llm_invoke([HumanMessage(content=prompt)])
    python_code = code_response.content.strip().replace("```python", "").replace("```", "")
    
    local_vars = {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        exec(python_code, {"pd": pd, "sqlite3": sqlite3, "plt": plt, "sns": sns}, local_vars)
        final_answer = local_vars.get("final_answer", "Code executed, but no text answer was generated.")
        
        if os.path.exists("output_plot.png"):
            final_answer += "\n\n[PLOT_GENERATED]"
            
        return {"messages": [AIMessage(content=final_answer)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error executing code: {str(e)}")]}

def general_worker(state: AgentState):
    response = safe_llm_invoke(state["messages"])
    return {"messages": [response]}

def blocked_node(state: AgentState):
    return {"messages": [AIMessage(content="I cannot process that request due to safety guidelines.")]}

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("manager", manager)
workflow.add_node("web_worker", web_worker)
workflow.add_node("math_worker", math_worker)
workflow.add_node("pdf_worker", pdf_worker)
workflow.add_node("user_data_worker", user_data_worker)
workflow.add_node("code_worker", code_worker)
workflow.add_node("general_worker", general_worker)
workflow.add_node("blocked", blocked_node)

workflow.set_entry_point("guardrail")
workflow.add_conditional_edges("guardrail", lambda x: x["next_agent"], {"blocked": "blocked", "manager": "manager"})
workflow.add_conditional_edges("manager", lambda x: x["next_agent"], {
    "web": "web_worker", "math": "math_worker", "pdf": "pdf_worker", 
    "user_data": "user_data_worker", "code": "code_worker", "general": "general_worker"
})

for n in ["web_worker", "math_worker", "pdf_worker", "user_data_worker", "code_worker", "general_worker", "blocked"]:
    workflow.add_edge(n, END)

multi_agent_app = workflow.compile()

# --- STREAMING GENERATOR ---
async def ask_agent_v2_stream(user_message):
    inputs = {"messages": [HumanMessage(content=user_message)]}
    
    for event in multi_agent_app.stream(inputs, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "guardrail": 
                yield ("step", f"🛡️ **Guardrail**: `{node_output.get('next_agent').upper()}`")
                await asyncio.sleep(0.5)
            elif node_name == "manager": 
                yield ("step", f"🧠 **Manager** routed to: `{node_output.get('next_agent').upper()}`")
                await asyncio.sleep(0.5)
            elif node_name == "blocked": 
                yield ("answer", node_output["messages"][-1].content)
            elif node_name == "code_worker":
                yield ("step", "🐍 **Code Agent** is executing Python...")
                await asyncio.sleep(0.5)
                yield ("answer", node_output["messages"][-1].content)
            else:
                yield ("step", f"⚙️ **{node_name.replace('_', ' ').title()}** is working...")
                await asyncio.sleep(0.5)
                yield ("answer", node_output["messages"][-1].content)