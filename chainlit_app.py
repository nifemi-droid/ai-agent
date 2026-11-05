import chainlit as cl
from multi_agent import ask_agent_v2_stream, ingest_files
from database import ensure_dbs_exist

ensure_dbs_exist()

@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session starts."""
    cl.user_session.set("history", [])
    cl.user_session.set("verbose_mode", False)
    cl.user_session.set("show_sources", True)
    
    # --- PM TOOLBOX (Quick Actions) ---
    actions = [
        cl.Action(name="tool_prd", value="prd", label="📝 Write a PRD", payload={}),
        cl.Action(name="tool_frd", value="frd", label="📊 Create an FRD", payload={}),
        cl.Action(name="tool_ai", value="ai", label="🤖 Design an AI Spec", payload={}),
        cl.Action(name="tool_platform", value="platform", label="⚙️ Platform Spec", payload={}),
        cl.Action(name="open_settings", value="open", label="⚙️ Settings", payload={})
    ]
    
    await cl.Message(
        content="✦ **Nexus** — Your Technical PM Co-pilot.\n\n"
                "Select a tool below to start, or just type your question.",
        actions=actions
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Called when a user clicks an old chat in the sidebar."""
    cl.user_session.set("history", [])
    cl.user_session.set("verbose_mode", False)
    cl.user_session.set("show_sources", True)
    
    await cl.Message(content="Resumed session. Note: The bot's memory of the previous context has been reset.").send()


# --- TOOLBOX CALLBACKS ---
@cl.action_callback("tool_prd")
async def on_tool_prd(action: cl.Action):
    await cl.Message(content="📝 **PRD Mode Activated**\n\nPlease describe the feature or product you want me to write a PRD for. (e.g., 'A notification system for mobile users')").send()

@cl.action_callback("tool_frd")
async def on_tool_frd(action: cl.Action):
    await cl.Message(content="📊 **FRD Mode Activated**\n\nPlease describe the feature you want me to write an FRD for.").send()

@cl.action_callback("tool_ai")
async def on_tool_ai(action: cl.Action):
    await cl.Message(content="🤖 **AI Spec Mode Activated**\n\nPlease describe the AI/ML product you want to spec out. I will include evaluation metrics, guardrails, and cost modeling.").send()

@cl.action_callback("tool_platform")
async def on_tool_platform(action: cl.Action):
    await cl.Message(content="⚙️ **Platform Spec Mode Activated**\n\nPlease describe the internal platform or API you want to spec out. I will include SLOs, API contracts, and Build vs Buy analysis.").send()


# --- SETTINGS CALLBACKS ---
@cl.action_callback("open_settings")
async def on_settings_action(action: cl.Action):
    verbose = cl.user_session.get("verbose_mode", False)
    sources = cl.user_session.get("show_sources", True)
    
    await cl.Message(
        content=f"**Current Settings**\n\n"
                f"- Agent Routing (Verbose): {'ON' if verbose else 'OFF'}\n"
                f"- Show Sources: {'ON' if sources else 'OFF'}",
        actions=[
            cl.Action(name="toggle_verbose", value="toggle", label=f"Toggle Routing ({'ON' if verbose else 'OFF'})", payload={}),
            cl.Action(name="toggle_sources", value="toggle", label=f"Toggle Sources ({'ON' if sources else 'OFF'})", payload={}),
            cl.Action(name="close_settings", value="close", label="Close", payload={}),
        ]
    ).send()

@cl.action_callback("toggle_verbose")
async def on_toggle_verbose(action: cl.Action):
    current = cl.user_session.get("verbose_mode", False)
    cl.user_session.set("verbose_mode", not current)
    await cl.Message(content=f"Agent Routing is now **{'ON' if not current else 'OFF'}**.").send()

@cl.action_callback("toggle_sources")
async def on_toggle_sources(action: cl.Action):
    current = cl.user_session.get("show_sources", True)
    cl.user_session.set("show_sources", not current)
    await cl.Message(content=f"Show Sources is now **{'ON' if not current else 'OFF'}**.").send()


# --- MAIN MESSAGE HANDLER ---
@cl.on_message
async def on_message(message: cl.Message):
    """Called every time the user sends a message."""
    
    user_id = "local"
    session_id = cl.context.session.id
    history = cl.user_session.get("history") or []
    
    verbose_mode = cl.user_session.get("verbose_mode", False)
    show_sources = cl.user_session.get("show_sources", True)

    # 1. Handle file attachments
    if message.elements:
        files_to_ingest = []
        for element in message.elements:
            if element.type == "file":
                with open(element.path, "rb") as f:
                    file_bytes = f.read()
                
                import base64
                files_to_ingest.append({
                    "name": element.name,
                    "data": base64.b64encode(file_bytes).decode()
                })
        
        if files_to_ingest:
            msg = cl.Message(content=f"📁 Processing {len(files_to_ingest)} file(s)...")
            await msg.send()
            
            res = await cl.make_async(ingest_files)(files_to_ingest, user_id, session_id)
            
            msg.content = res
            await msg.update()
            
            if not message.content or message.content.strip() == "":
                return

    # 2. If there's no text, do nothing
    if not message.content or message.content.strip() == "":
        return

    # 3. Stream the agent response
    final_answer = cl.Message(content="")
    await final_answer.send()
    
    got_answer = False
    
    try:
        async for kind, content in ask_agent_v2_stream(
            message.content, user_id, history, session_id
        ):
            if kind == "step":
                if verbose_mode:
                    await cl.Message(content=f"_{content}_").send()
                else:
                    print(f"[Internal Routing]: {content}")
                    
            elif kind == "chunk":
                got_answer = True
                await final_answer.stream_token(content)
                
            elif kind == "answer":
                got_answer = True
                final_answer.content = content
                await final_answer.update()
                
    except Exception as e:
        final_answer.content = f"**Error:** {type(e).__name__}: {e}"
        await final_answer.update()
        got_answer = True

    if not got_answer:
        final_answer.content = "**No answer was returned.** Please try again."
        await final_answer.update()

    await final_answer.update()
    
    # 4. Clean up sources if the user has toggled them OFF
    if not show_sources and "[SOURCES]" in final_answer.content:
        clean_text = final_answer.content.split("[SOURCES]")[0].strip()
        final_answer.content = clean_text
        await final_answer.update()
    
    # 5. Update the conversation history for the next turn
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": final_answer.content})
    cl.user_session.set("history", history)