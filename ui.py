import streamlit as st
import pandas as pd
from agent import ask_agent, execute_approved_tool
from multi_agent import ask_multi_agent, ingest_pdf
from database import (ensure_dbs_exist, create_session, get_sessions, 
                      delete_session, save_message, load_messages, 
                      log_analytics, get_analytics)

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to make it look cleaner
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; }
    [data-testid="stSidebar"] { min-width: 300px; max-width: 300px; }
</style>
""", unsafe_allow_html=True)

ensure_dbs_exist()

# --- SESSION STATE INITIALIZATION ---
if "current_session_id" not in st.session_state:
    sessions = get_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_session("New Chat")

# --- SIDEBAR (CHAT HISTORY) ---
with st.sidebar:
    st.title("🤖 AI Agents")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_session("New Chat")
        st.session_state.current_session_id = new_id
        st.rerun()

    st.divider()
    st.subheader("Recent Chats")
    
    sessions = get_sessions()
    for session_id, title in sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            # Highlight the active chat
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
    
    # Settings Expander
    with st.expander("⚙️ Settings & Admin"):
        st.markdown("**Mode Selection**")
        agent_mode = st.radio("Agent Mode", ["Standard Agent", "Multi-Agent Router"])
        
        st.divider()
        if st.button("Clear All Analytics"):
            pass # Add clear analytics logic here if needed
            
        st.markdown("**Admin Dashboard**")
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
st.title("How can I help you today?")

# Get current messages
current_messages = load_messages(st.session_state.current_session_id)

# Display chat history
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & FILE UPLOADER ---
col1, col2 = st.columns([10, 1])

with col2:
    # We use an expander disguised as a file uploader button
    with st.popover("📎"):
        st.write("Upload a PDF")
        uploaded_pdf = st.file_uploader("Upload", type="pdf", label_visibility="collapsed")
        if uploaded_pdf is not None:
            if st.button("Ingest PDF"):
                with st.spinner("Ingesting..."):
                    res = ingest_pdf(uploaded_pdf.read())
                    st.success(res)

with col1:
    prompt = st.chat_input("Ask anything...")

# --- HANDLE USER INPUT ---
if prompt:
    # Save user message to DB
    save_message(st.session_state.current_session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if agent_mode == "Standard Agent":
                # Standard logic (Streaming)
                response_placeholder = st.empty()
                full_response = ""
                stream = ask_agent(prompt, current_messages)
                tool_used = "None"
                latency = 0
                
                for chunk in stream:
                    if chunk.get("requires_approval"):
                        # We need to handle HITL (Keep it simple for now)
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
                # Multi-Agent logic
                steps, final_answer = ask_multi_agent(prompt)
                with st.expander("🔍 View Manager's Thought Process"):
                    for step in steps:
                        st.markdown(step)
                st.markdown(final_answer)
    
    # Save assistant message to DB
    save_message(st.session_state.current_session_id, "assistant", final_answer)
    st.rerun() # Refresh to update sidebar titles