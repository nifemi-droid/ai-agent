import streamlit as st
import pandas as pd
import os
from multi_agent import ask_agent_v2, ingest_file
from database import (ensure_dbs_exist, create_session, get_sessions, 
                      delete_session, save_message, load_messages, 
                      log_analytics, get_analytics)

st.set_page_config(page_title="Nexus AI", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

ensure_dbs_exist()

# --- THEME ---
if "theme" not in st.session_state: 
    st.session_state.theme = "Light"

bg = "#FFFFFF" if st.session_state.theme == "Light" else "#0E1117"
text = "#31333F" if st.session_state.theme == "Light" else "#FAFAFA"
sidebar = "#F0F2F6" if st.session_state.theme == "Light" else "#262730"

st.markdown(f"""<style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar} !important; }}
    [data-testid="stSidebar"] * {{ color: {text} !important; }}
    [data-testid="stChatMessage"] {{ background-color: {sidebar} !important; border-radius: 12px; padding: 15px; }}
    .stButton>button {{ color: {text} !important; background-color: {sidebar} !important; border: 1px solid {text}33 !important; }}
</style>""", unsafe_allow_html=True)

# --- STATE ---
if "current_session_id" not in st.session_state:
    s = get_sessions()
    st.session_state.current_session_id = s[0][0] if s else create_session("New Chat")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌌 Nexus AI")
    theme = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
    if theme != (st.session_state.theme == "Dark"):
        st.session_state.theme = "Dark" if theme else "Light"
        st.rerun()

    st.divider()
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = create_session("New Chat")
        st.rerun()

    st.subheader("Recent Chats")
    for sid, title in get_sessions():
        c1, c2 = st.columns([4, 1])
        with c1:
            if sid == st.session_state.current_session_id: 
                st.button(f"👉 {title}", key=sid, disabled=True, use_container_width=True)
            elif st.button(title, key=sid, use_container_width=True): 
                st.session_state.current_session_id = sid
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{sid}"):
                delete_session(sid)
                if sid == st.session_state.current_session_id: 
                    st.session_state.current_session_id = create_session("New Chat")
                st.rerun()

    st.divider()
    with st.expander("⚙️ Admin"):
        if st.checkbox("View Analytics"):
            data = get_analytics()
            if data:
                st.metric("Total Queries", len(data))
                st.dataframe(pd.DataFrame(data, columns=["Query", "Tool", "Latency", "Time"]), use_container_width=True)

# --- MAIN CHAT ---
st.title("How can I help you today?")

for msg in load_messages(st.session_state.current_session_id):
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

c1, c2 = st.columns([11, 1])
with c2:
    with st.popover("📎"):
        file = st.file_uploader("Upload", type=["pdf", "csv", "xlsx", "xls"], label_visibility="collapsed")
        if file and st.button("Ingest"):
            with st.spinner("Ingesting..."): 
                st.success(ingest_file(file.read(), file.name))

with c1:
    prompt = st.chat_input("Ask anything...")

# --- HANDLE INPUT ---
if prompt:
    save_message(st.session_state.current_session_id, "user", prompt)
    with st.chat_message("user"): 
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Routing..."):
            steps, answer = ask_agent_v2(prompt)
            
            # Show thought process
            with st.expander("🔍 View Thought Process"):
                for s in steps: 
                    st.markdown(s)
            
            # Render Answer (and charts if any)
            if "[PLOT_GENERATED]" in answer:
                clean_answer = answer.replace("[PLOT_GENERATED]", "")
                st.markdown(clean_answer)
                if os.path.exists("output_plot.png"):
                    st.image("output_plot.png")
                    os.remove("output_plot.png") # Clean up the file after showing
            else:
                st.markdown(answer)
            
    save_message(st.session_state.current_session_id, "assistant", answer)
    st.rerun()