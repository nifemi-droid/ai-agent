import streamlit as st
import pandas as pd
import base64
from agent import ask_agent, execute_approved_tool
from multi_agent import ask_multi_agent, ingest_file
from database import (ensure_dbs_exist, create_session, get_sessions, 
                      delete_session, save_message, load_messages, 
                      log_analytics, get_analytics)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nexus AI", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

ensure_dbs_exist()

# --- SESSION STATE INITIALIZATION ---
if "current_session_id" not in st.session_state:
    sessions = get_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_session("New Chat")

# --- SIDEBAR ---
with st.sidebar:
    # Custom Header with Logo
    st.markdown("<h1 style='text-align: center;'>🌌 Nexus AI</h1>", unsafe_allow_html=True)
    st.divider()
    
    # Theme Toggle Switch
    theme_toggle = st.toggle("🌙 Dark Mode", value=False)
    st.divider()
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_session("New Chat")
        st.session_state.current_session_id = new_id
        st.rerun()

    st.subheader("Recent Chats")
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

    st.divider()
    
    with st.expander("⚙️ Settings & Admin"):
        agent_mode = st.radio("Agent Mode", ["Standard Agent", "Multi-Agent Router"])
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

# --- APPLY DYNAMIC THEME (STREAMLIT NATIVE) ---
# We use a small JS injection to click the native theme toggle if the switch is on
if theme_toggle:
    st.markdown("""
        <script>
            // This script triggers Streamlit's native dark mode button in the top right menu
            const menu = window.parent.document.querySelector('[data-testid="stMainMenu"]');
            if (menu) { menu.click(); }
        </script>
    """, unsafe_allow_html=True)

# --- MAIN CHAT AREA ---
# Add some styling to the main area
st.markdown("""
<style>
    /* Center the chat container and make it look clean */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stChatMessage { border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .stChatInput { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("How can I help you today?")
st.caption("Nexus AI can analyze PDFs, CSVs, Excel files, and answer complex questions.")

current_messages = load_messages(st.session_state.current_session_id)

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & FILE UPLOADER ---
col1, col2 = st.columns([11, 1])

with col2:
    with st.popover("📎", use_container_width=True):
        st.write("Upload Data (PDF, CSV, Excel)")
        uploaded_file = st.file_uploader("Upload", type=["pdf", "csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Ingest File", use_container_width=True):
                with st.spinner("Ingesting..."):
                    res = ingest_file(uploaded_file.read(), uploaded_file.name)
                    st.success(res)

with col1:
    prompt = st.chat_input("Ask anything...")

# --- HANDLE USER INPUT ---
if prompt:
    save_message(st.session_state.current_session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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