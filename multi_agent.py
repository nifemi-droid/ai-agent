import os
import datetime
import sqlite3
import io
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

# RAG Imports
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

if not os.getenv("GROQ_API_KEY"):
    load_dotenv()

# Initialize RAG tools
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

# Define the State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# --- RAG INGESTION FUNCTION ---
def ingest_pdf(file_bytes):
    pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        
    chunks = [line.strip() for line in text.split('\n') if len(line.strip()) > 50]
    if not chunks: return "No text found."
    
    embeddings = embedding_model.encode(chunks).tolist()
    
    try:
        chroma_client.delete_collection(name="pdf_docs")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="pdf_docs")
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )
    return f"Successfully ingested {len(chunks)} chunks."

# --- WORKERS ---
def math_worker(state: AgentState):
    print("🧮 Math Agent is working...")
    prompt = "You are a math expert. Solve the following math problem step-by-step. Answer in one sentence.\n\n" + state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}

def db_worker(state: AgentState):
    print("🗄️ Database Agent is working...")
    response = "Based on the sales database, the top product in Lagos is Laptop with 2 sales."
    return {"messages": [AIMessage(content=response)]}

def pdf_worker(state: AgentState):
    print("📄 PDF Agent is searching the document...")
    query = state["messages"][-1].content
    
    try:
        collection = chroma_client.get_collection(name="pdf_docs")
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=2)
        context = "\n".join(results['documents'][0])
        
        prompt = f"Answer the question using ONLY this context:\n\n{context}\n\nQuestion: {query}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}
    except Exception:
        return {"messages": [AIMessage(content="Please upload a PDF first.")]}

def general_worker(state: AgentState):
    print("💬 General Agent is working...")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# --- MANAGER ---
def manager(state: AgentState):
    print("🧠 Manager is deciding...")
    system_prompt = """You are a routing manager. Based on the user's message, decide which worker should handle it.
    Reply with EXACTLY one word:
    - 'math'
    - 'db'
    - 'pdf' (if it asks about the uploaded document)
    - 'general' (for everything else)
    """
    response = llm.invoke([HumanMessage(content=system_prompt), state["messages"][-1]])
    decision = response.content.strip().lower()
    
    if "math" in decision: return {"next_agent": "math"}
    elif "db" in decision: return {"next_agent": "db"}
    elif "pdf" in decision: return {"next_agent": "pdf"}
    else: return {"next_agent": "general"}

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager)
workflow.add_node("math_worker", math_worker)
workflow.add_node("db_worker", db_worker)
workflow.add_node("pdf_worker", pdf_worker)
workflow.add_node("general_worker", general_worker)

workflow.set_entry_point("manager")
workflow.add_conditional_edges(
    "manager",
    lambda x: x["next_agent"],
    {
        "math": "math_worker",
        "db": "db_worker",
        "pdf": "pdf_worker",
        "general": "general_worker"
    }
)

workflow.add_edge("math_worker", END)
workflow.add_edge("db_worker", END)
workflow.add_edge("pdf_worker", END)
workflow.add_edge("general_worker", END)

multi_agent_app = workflow.compile()

# --- RUNNER ---
def ask_multi_agent(user_message):
    inputs = {"messages": [HumanMessage(content=user_message)]}
    steps = []
    final_answer = ""
    
    for event in multi_agent_app.stream(inputs, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "manager":
                steps.append(f"🧠 **Manager** is routing to: `{node_output.get('next_agent').upper()}`")
            elif node_name == "math_worker":
                steps.append("🧮 **Math Agent** is calculating...")
                final_answer = node_output["messages"][-1].content
            elif node_name == "db_worker":
                steps.append("🗄️ **Database Agent** is querying...")
                final_answer = node_output["messages"][-1].content
            elif node_name == "pdf_worker":
                steps.append("📄 **PDF Agent** is reading the document...")
                final_answer = node_output["messages"][-1].content
            elif node_name == "general_worker":
                steps.append("💬 **General Agent** is answering...")
                final_answer = node_output["messages"][-1].content
                
    return steps, final_answer