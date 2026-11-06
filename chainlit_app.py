import chainlit as cl
from multi_agent import ask_agent_v2_stream, ingest_files
from database import ensure_dbs_exist

ensure_dbs_exist()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("verbose_mode", False)
    cl.user_session.set("show_sources", True)
    cl.user_session.set("last_prompt", "")
    
    actions = [
        cl.Action(name="tool_prd", value="prd", label="📝 Write a PRD", payload={}),
        cl.Action(name="tool_ai", value="ai", label="🤖 Design an AI Spec", payload={}),
        cl.Action(name="open_settings", value="open", label="⚙️ Settings", payload={})
    ]
    
    await cl.Message(
        content="✦ **Nexus** — Your Technical PM Co-pilot.\n\n"
                "Select a tool below to start, or just type your question.",
        actions=actions
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    cl.user_session.set("history", [])
    cl.user_session.set("verbose_mode", False)
    cl.user_session.set("show_sources", True)
    cl.user_session.set("last_prompt", "")
    await cl.Message(content="Resumed session.").send()


# --- TOOLBOX CALLBACKS ---
@cl.action_callback("tool_prd")
async def on_tool_prd(action: cl.Action):
    await cl.Message(content="📝 **PRD Mode Activated**\n\nPlease describe the feature you want a PRD for.").send()

@cl.action_callback("tool_ai")
async def on_tool_ai(action: cl.Action):
    await cl.Message(content="🤖 **AI Spec Mode Activated**\n\nPlease describe the AI/ML product.").send()


# --- DEEP DIVE CALLBACK (NEW) ---
@cl.action_callback("deep_dive")
async def on_deep_dive(action: cl.Action):
    last_prompt = cl.user_session.get("last_prompt", "")
    if not last_prompt:
        await cl.Message(content="No previous prompt found.").send()
        return
        
    await cl.Message(content="🔬 **Running Deep Dive...** (Draft → Critique → Refine). This will take ~10 seconds.").send()
    
    history = cl.user_session.get("history") or []
    session_id = cl.context.session.id
    
    final_answer = cl.Message(content="")
    await final_answer.send()
    
    try:
        async for kind, content in ask_agent_v2_stream(
            last_prompt, "local", history, session_id, deep_dive=True
        ):
            if kind == "chunk":
                await final_answer.stream_token(content)
                
    except Exception as e:
        final_answer.content = f"**Error:** {type(e).__name__}: {e}"
        
    await final_answer.update()


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
            cl.Action(name="toggle_verbose", value="toggle", label=f"Routing ({'ON' if verbose else 'OFF'})", payload={}),
            cl.Action(name="toggle_sources", value="toggle", label=f"Sources ({'ON' if sources else 'OFF'})", payload={}),
        ]
    ).send()

@cl.action_callback("toggle_verbose")
async def on_toggle_verbose(action: cl.Action):
    current = cl.user_session.get("verbose_mode", False)
    cl.user_session.set("verbose_mode", not current)
    await cl.Message(content=f"Routing is now **{'ON' if not current else 'OFF'}**.").send()

@cl.action_callback("toggle_sources")
async def on_toggle_sources(action: cl.Action):
    current = cl.user_session.get("show_sources", True)
    cl.user_session.set("show_sources", not current)
    await cl.Message(content=f"Sources is now **{'ON' if not current else 'OFF'}**.").send()


# --- MAIN MESSAGE HANDLER ---
@cl.on_message
async def on_message(message: cl.Message):
    user_id = "local"
    session_id = cl.context.session.id
    history = cl.user_session.get("history") or []
    
    verbose_mode = cl.user_session.get("verbose_mode", False)
    show_sources = cl.user_session.get("show_sources", True)

    # Save the last prompt so Deep Dive can use it
    if message.content:
        cl.user_session.set("last_prompt", message.content)

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

    if not message.content or message.content.strip() == "":
        return

    final_answer = cl.Message(content="")
    await final_answer.send()
    
    got_answer = False
    
    try:
        # Default to fast mode
        async for kind, content in ask_agent_v2_stream(
            message.content, user_id, history, session_id, deep_dive=False
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
    
    if not show_sources and "[SOURCES]" in final_answer.content:
        clean_text = final_answer.content.split("[SOURCES]")[0].strip()
        final_answer.content = clean_text
        await final_answer.update()
    
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": final_answer.content})
    cl.user_session.set("history", history)
    
    # Add the Deep Dive button after the response
    await cl.Message(
        content="Want a more rigorous version of this?",
        actions=[
            cl.Action(name="deep_dive", value="deep", label="🔬 Run Deep Dive (Draft → Critique → Refine)", payload={})
        ]
    ).send()