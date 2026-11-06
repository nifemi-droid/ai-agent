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
    
    await cl.Message(
        content="✦ **Nexus** — Your Technical PM Co-pilot.\n\n"
                "**Quick Commands** (type these to start):\n"
                "- `/prd` — Write a Product Requirements Doc\n"
                "- `/frd` — Write a Functional Requirements Doc\n"
                "- `/ai` — Write an AI/ML Product Spec\n"
                "- `/platform` — Write a Platform/Systems Spec\n"
                "- `/strategy` — Write a Strategy Memo\n\n"
                "**Special Modes:**\n"
                "- Upload a doc and say **\"Review this\"** to get a VP-level critique.\n"
                "- Upload multiple docs and say **\"Compare these\"** to get a synthesis matrix.\n\n"
                "Or just type your question directly."
    ).send()

# ... (keep all the @cl.action_callback functions for Deep Dive and Settings exactly as they are) ...

@cl.on_message
async def on_message(message: cl.Message):
    user_id = "local"
    session_id = cl.context.session.id
    history = cl.user_session.get("history") or []
    
    verbose_mode = cl.user_session.get("verbose_mode", False)
    show_sources = cl.user_session.get("show_sources", True)

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

    final_answer = cl.Message(content="")
    await final_answer.send()
    
    got_answer = False
    
    try:
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
    
    await cl.Message(
        content="Want a more rigorous version of this?",
        actions=[
            cl.Action(name="deep_dive", value="deep", label="🔬 Run Deep Dive", payload={})
        ]
    ).send()