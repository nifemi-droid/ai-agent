import streamlit as st
import pandas as pd
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

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# --- DYNAMIC THEME CSS (FORCES EVERYTHING TO CHANGE) ---
if st.session_state.theme == "Light":
    bg_color = "#FFFFFF"
    text_color = "#31333F"
    sidebar_bg = "#F0F2F6"
    chat_bg = "#F0F2F6"
    border_color = "#E0E0E0"
    input_bg = "#FFFFFF"
else:
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    sidebar_bg = "#262730"
    chat_bg = "#262730"
    border_color = "#444444"
    input_bg = "#262730"

st.markdown(f"""
<style>
    /* 1. Main App Background & Text */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    
    /* 2. Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}
    
    /* 3. Chat Messages */
    [data-testid="stChatMessage"] {{
        background-color: {chat_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px;
        padding: 15px;
    }}
    
    /* 4. Chat Input (Bottom Bar) */
    [data-testid="stChatInput"] {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: {text_color} !important;
    }}
    
    /* 5. Buttons */
    .stButton>button {{
        color: {text_color} !important;
        background-color: {chat_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px;
    }}
    .stButton>button:hover {{
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
    }}

    /* 6. Settings/Radio/Checkbox */
    .stRadio label, .stCheckbox label {{
        color: {text_color} !important;
    }}
    
    /* 7. File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: {chat_bg} !important;
        border-radius: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"<h1 style='text-align: center; color: {text_color};'>🌌 Nexus AI</h1>", unsafe_allow_html=True)
    st.divider()
    
    # Toggle switch
    theme_choice = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    if theme_choice and st.session_state.theme != "Dark":
        st.session_state.theme = "Dark"
        st.rerun()
    elif not theme_choice and st.session_state.theme != "Light":
        st.session_state.theme = "Light"
        st.rerun()

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

# --- MAIN CHAT AREA ---
st.title("How can I help you today?")
st.caption("Nexus AI can analyze PDFs, CSVs, Excel files, and answer complex questions.")

current_messages = load_messages(st.session_state.current_session_id)

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & FILE UPLOADER ---
col1, col2 = st.columns([11, 1])

with col2:
    with st.popover("📎"):
        st.write("Upload Data (PDF, CSV, Excel)")
        uploaded_file = st.file_uploader("Upload", type=["pdf", "csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Ingest File"):
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