import chainlit as cl
from multi_agent import ask_agent_v2_stream, ingest_files
from database import ensure_dbs_exist

ensure_dbs_exist()

@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session starts."""
    # Initialize the chat history for this session
    cl.user_session.set("history", [])
    
    # Initialize settings (if you want to add custom PM settings later)
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Switch(id="show_sources", label="Show Sources", initial=True),
            cl.input_widget.Switch(id="verbose_mode", label="Verbose Mode (Debug)", initial=False),
        ]
    ).send()
    
    cl.user_session.set("settings", settings)
    
    # Send a welcome message
    await cl.Message(
        content="✦ **Nexus** — Your Technical PM Co-pilot.\n\n"
                "Ask me to write a PRD, research a market, or analyze a document. "
                "Use the attach button below to upload a PDF, DOCX, or CSV."
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    """Called when the user changes settings."""
    cl.user_session.set("settings", settings)
    await cl.Message(content="Settings updated.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Called every time the user sends a message."""
    
    user_id = "local"
    # Chainlit gives each chat session a unique ID. 
    # Clicking "New Chat" in the UI generates a new session_id, isolating documents.
    session_id = cl.context.session.id
    history = cl.user_session.get("history") or []
    settings = cl.user_session.get("settings") or {"show_sources": True, "verbose_mode": False}

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
            # Show a subtle message that ingestion is happening
            msg = cl.Message(content=f"📁 Processing {len(files_to_ingest)} file(s)...")
            await msg.send()
            
            # Call existing ingest_files with the session-scoped ID
            res = await cl.make_async(ingest_files)(files_to_ingest, user_id, session_id)
            
            # Update the message with the result
            msg.content = res
            await msg.update()
            
            # If the user only uploaded a file without a text prompt, stop here
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
                # SILENTLY LOG TO TERMINAL (For debugging)
                if settings.get("verbose_mode", False):
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
    
    # 4. Update the conversation history for the next turn
    # This is what makes the bot remember context within the SAME chat.
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": final_answer.content})
    cl.user_session.set("history", history)


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Called when a user resumes an old chat from the sidebar."""
    # Chainlit automatically handles the session state for resumed chats, 
    # but we initialize history to be safe.
    cl.user_session.set("history", [])