import chainlit as cl
import asyncio
from multi_agent import ask_agent_v2_stream, ingest_files
from database import ensure_dbs_exist

ensure_dbs_exist()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("last_prompt", "")
    cl.user_session.set("active_task", None)
    
    await cl.Message(
        content="✦ **Nexus** — Your generalist AI assistant.\n\n"
                "Ask me anything. I can search the web, answer questions about your uploaded documents, and chat.\n\n"
                "Use the 📎 paperclip to upload a PDF, DOCX, or CSV."
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    cl.user_session.set("history", [])
    cl.user_session.set("last_prompt", "")
    cl.user_session.set("active_task", None)
    await cl.Message(content="Resumed session.").send()


# --- STOP CONTROL: Called when the user clicks the Stop button ---
@cl.on_stop
async def on_stop():
    task = cl.user_session.get("active_task")
    if task and not task.done():
        task.cancel()
        await cl.Message(content="⏹️ **Generation stopped.**").send()


# --- REGENERATE: Re-runs the last prompt ---
@cl.action_callback("regenerate")
async def on_regenerate(action: cl.Action):
    last_prompt = cl.user_session.get("last_prompt", "")
    if not last_prompt:
        await cl.Message(content="No previous prompt to regenerate.").send()
        return
    
    await cl.Message(content="🔄 **Regenerating...**").send()
    await process_message(last_prompt, is_regenerate=True)


# --- MAIN MESSAGE HANDLER ---
@cl.on_message
async def on_message(message: cl.Message):
    # Handle file uploads first
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
            session_id = cl.context.session.id
            msg = cl.Message(content=f"📁 Processing {len(files_to_ingest)} file(s)...")
            await msg.send()
            res = await cl.make_async(ingest_files)(files_to_ingest, "local", session_id)
            msg.content = res
            await msg.update()
    
    if not message.content or message.content.strip() == "":
        return
    
    cl.user_session.set("last_prompt", message.content)
    await process_message(message.content, is_regenerate=False)


# --- THE STREAMING PROCESSOR (Handles both new messages and regenerates) ---
async def process_message(prompt: str, is_regenerate: bool = False):
    user_id = "local"
    session_id = cl.context.session.id
    history = cl.user_session.get("history") or []
    
    # Determine if the session has uploaded files (for pre-routing)
    has_files = False
    try:
        from multi_agent import chroma_client
        collection = chroma_client.get_collection(name=f"docs_{user_id}_{session_id}")
        if collection.count() > 0:
            has_files = True
    except Exception:
        has_files = False
    
    # Create the assistant message (this will be streamed into)
    msg = cl.Message(content="")
    await msg.send()
    
    full_response = ""
    current_step_msg = None
    current_step_text = ""
    
    try:
        # Create the async generator task so we can cancel it
        agent_gen = ask_agent_v2_stream(
            prompt, user_id, history, session_id, has_files=has_files
        )
        active_task = asyncio.create_task(anext_wrapper(agent_gen))
        cl.user_session.set("active_task", active_task)
        
        while True:
            try:
                kind, content = await active_task
            except StopAsyncIteration:
                break
            
            if kind == "step":
                # Step messages appear as short italic lines above the answer
                if content == current_step_text:
                    pass
                else:
                    current_step_text = content
                    if current_step_msg is not None:
                        # Update the existing step message in place
                        current_step_msg.content = content
                        await current_step_msg.update()
                    else:
                        current_step_msg = cl.Message(content=content, parent_id=msg.id)
                        await current_step_msg.send()
                        
            elif kind == "chunk":
                full_response += content
                await msg.stream_token(content)
                
            elif kind == "answer":
                full_response = content
                await msg.stream_token(content)
            
            # Prep the next iteration of the generator
            active_task = asyncio.create_task(anext_wrapper(agent_gen))
            cl.user_session.set("active_task", active_task)
    
    except asyncio.CancelledError:
        if not full_response:
            full_response = "⏹️ *Generation stopped by user.*"
            msg.content = full_response
            await msg.update()
        else:
            # Keep what was generated so far
            await msg.update()
    
    except Exception as e:
        msg.content = f"**Error:** {type(e).__name__}: {e}"
        await msg.update()
        full_response = msg.content
    
    finally:
        cl.user_session.set("active_task", None)
    
    # Update history
    if not is_regenerate:
        history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)
    
    # Add the "Regenerate" button below the response
    await cl.Message(
        content="",
        actions=[
            cl.Action(name="regenerate", value="regen", label="🔄 Regenerate", payload={})
        ]
    ).send()


# --- WRAPPER: Helper to make asyncio.create_task work on async generators ---
async def anext_wrapper(gen):
    """Wraps anext(gen) so it can be cancelled via asyncio.create_task."""
    return await gen.__anext__()