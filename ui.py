import streamlit as st
import pandas as pd
import facade
from agent import ask_agent, execute_approved_tool
from multi_agent import ask_multi_agent, ingest_file
from database import (ensure_dbs_exist, create_session, get_sessions, 
                      delete_session, save_message, load_messages, 
                      log_analytics, get_analytics)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nexus AI", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

# --- APPLY FACADE THEME ---
facade.theme.apply(
    preset="carbon-sage", 
    radius="md",
    font_sans="DM Sans",
    font_link="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap"
)

ensure_dbs_exist()

# --- SESSION STATE INITIALIZATION ---
if "current_session_id" not in st.session_state:
    sessions = get_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_session("New Chat")

# --- SIDEBAR (FACADE) ---
with facade.Sidebar(logo="🌌", footer="Nexus AI v1.0"):
    # Use native Streamlit toggle to avoid facade version issues
    is_dark = st.toggle("Dark Mode", key="dark_mode")
    
    if is_dark:
        facade.theme.apply(preset="carbon-dark")
    else:
        facade.theme.apply(preset="carbon-sage")

    facade.Separator()
    
    # Use native Streamlit button to avoid facade version issues
    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_session("New Chat")
        st.session_state.current_session_id = new_id
        st.rerun()

    facade.Separator(label="Recent Chats")
    
    sessions = get_sessions()
    for session_id, title in sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            if session_id == st.session_state.current_session_id:
                st.button(f"👉 {title}", key=session_id, disabled=True, use_container_width=True)
            else:
                if st.button(title, key=session_id, use_container_width=True):
                    st.session_state.current_session_id = session_id
                    st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{session_id}"):
                delete_session(session_id)
                if session_id == st.session_state.current_session_id:
                    st.session_state.current_session_id = create_session("New Chat")
                st.rerun()

    facade.Separator()
    
    with facade.Accordion("⚙️ Settings & Admin"):
        # Use native Streamlit radio to avoid facade version issues
        agent_mode = st.radio("Agent Mode", ["Standard Agent", "Multi-Agent Router"], key="agent_mode")
        
        if st.checkbox("View Analytics"):
            data = get_analytics()
            if not data:
                st.info("No data yet.")
            else:
                total = len(data)
                avg_lat = sum([r[2] for r in data]) / total
                st.metric("Total Queries", total)
                st.metric("Avg Latency", f"{avg_lat:.2f}s")
                df = pd.DataFrame(data, columns=["Query", "Tool", "Latency", "Time"])
                st.dataframe(df, use_container_width=True)

# --- MAIN CHAT AREA ---
facade.Card(
    title="How can I help you today?",
    description="Nexus AI can analyze PDFs, CSVs, Excel files, and answer complex questions."
)

current_messages = load_messages(st.session_state.current_session_id)

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & FILE UPLOADER ---
col1, col2 = st.columns([11, 1])

with col2:
    with st.popover("📎"):
        facade.Card(title="Upload Data", description="PDF, CSV, or Excel")
        uploaded_file = st.file_uploader("Upload", type=["pdf", "csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Ingest File"):
                with st.spinner("Ingesting..."):
                    res = ingest_file(uploaded_file.read(), uploaded_file.name)
                    facade.Alert(res, variant="success")

with col1:
    prompt = st.chat_input("Ask anything...")

# --- HANDLE USER INPUT ---
if prompt:
    save_message(st.session_state.current_session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve agent_mode from native radio
            agent_mode = st.session_state.get("agent_mode", "Standard Agent")
            
            if agent_mode == "Standard Agent":
                response_placeholder = st.empty()
                full_response = ""
                stream = ask_agent(prompt, current_messages)
                tool_used = "None"
                latency = 0
                
                for chunk in stream:
                    if chunk.get("requires_approval"):
                        full_response = chunk["message"]
                        break
                    if chunk.get("type") == "content":
                        full_response += chunk["text"]
                        response_placeholder.markdown(full_response + "▌")
                        tool_used = chunk.get("tool_used", "None")
                        latency = chunk.get("latency", 0)
                
                response_placeholder.markdown(full_response)
                log_analytics(prompt, tool_used, latency)
                final_answer = full_response
            else:
                steps, final_answer = ask_multi_agent(prompt)
                with st.expander("🔍 View Manager's Thought Process"):
                    for step in steps:
                        st.markdown(step)
                st.markdown(final_answer)
    
    save_message(st.session_state.current_session_id, "assistant", final_answer)
    st.rerun()