import streamlit as st
import pandas as pd
from agent import ask_agent, execute_approved_tool
from multi_agent import ask_multi_agent
from database import ensure_dbs_exist, save_message, load_messages, clear_history, log_analytics, get_analytics

st.set_page_config(page_title="AI Tool Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Agent Architectures")

ensure_dbs_exist()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "💬 Standard Agent (Tools)", 
        "🧠 Multi-Agent Router", 
        "📊 Admin Dashboard"
    ])
    st.divider()
    st.header("Settings")
    if st.button("🗑️ Clear Chat History"):
        clear_history()
        st.rerun()

# --- PAGE 1: STANDARD AGENT ---
if page == "💬 Standard Agent (Tools)":
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages()
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.pending_action:
        action = st.session_state.pending_action
        st.warning(f"⚠️ **Human Approval Required**: {action['message']}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve & Send"):
                result = execute_approved_tool(action["tool_name"], action["tool_args"])
                st.session_state.messages.append({"role": "assistant", "content": result})
                save_message("assistant", result)
                st.session_state.pending_action = None
                st.rerun()
        with col2:
            if st.button("❌ Deny"):
                result = "Action denied by user."
                st.session_state.messages.append({"role": "assistant", "content": result})
                save_message("assistant", result)
                st.session_state.pending_action = None
                st.rerun()

    if prompt := st.chat_input("Ask a question...", disabled=bool(st.session_state.pending_action)):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message("user", prompt)
        
        history = st.session_state.messages[:-1] 
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            stream = ask_agent(prompt, history)
            tool_used = "None"
            latency = 0
            
            for chunk in stream:
                if chunk.get("requires_approval"):
                    st.session_state.pending_action = chunk
                    st.rerun()
                if chunk.get("type") == "content":
                    full_response += chunk["text"]
                    response_placeholder.markdown(full_response + "▌")
                    tool_used = chunk.get("tool_used", "None")
                    latency = chunk.get("latency", 0)
            
            response_placeholder.markdown(full_response)
            
        log_analytics(prompt, tool_used, latency)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_message("assistant", full_response)

# --- PAGE 2: MULTI-AGENT ROUTER ---
elif page == "🧠 Multi-Agent Router":
    st.markdown("""
    This mode uses a **Manager Agent** to route your question to a specialist worker.
    *   **Math Agent:** Handles calculations.
    *   **DB Agent:** Handles sales data.
    *   **General Agent:** Handles everything else.
    """)
    
    # We use a separate chat history for the multi-agent to keep it clean
    if "multi_messages" not in st.session_state:
        st.session_state.multi_messages = []

    for message in st.session_state.multi_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask the multi-agent system..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.multi_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Manager is routing..."):
                response = ask_multi_agent(prompt)
            st.markdown(response)
            
        st.session_state.multi_messages.append({"role": "assistant", "content": response})

# --- PAGE 3: ADMIN DASHBOARD ---
elif page == "📊 Admin Dashboard":
    st.header("Analytics Dashboard")
    data = get_analytics()
    if not data:
        st.info("No analytics data yet.")
    else:
        total_queries = len(data)
        avg_latency = sum([row[2] for row in data]) / total_queries
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", total_queries)
        col2.metric("Average Latency", f"{avg_latency:.2f}s")
        col3.metric("Tools Used", len(set([row[1] for row in data])))
        st.divider()
        df = pd.DataFrame(data, columns=["User Query", "Tool Used", "Latency (s)", "Timestamp"])
        st.dataframe(df, use_container_width=True)