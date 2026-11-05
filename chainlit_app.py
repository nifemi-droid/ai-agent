import chainlit as cl
from multi_agent import ask_agent_v2_stream, ingest_files
from database import ensure_dbs_exist

ensure_dbs_exist()

# Chainlit manages sessions natively. We use cl.user_session to store per-chat state.
# The session_id from Chainlit is used to scope documents (like ChatGPT does).

@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session starts."""
    # Initialize the chat history for this session
    cl.user_session.set("history", [])
    
    # Send a welcome message
    await cl.Message(
        content="✦ **Nexus** — Your Technical PM Co-pilot.\n\n"
                "Ask me to write a PRD, research a market, or analyze a document. "
                "Use the attach button below to upload a PDF, DOCX, or CSV."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Called every time the user sends a message."""
    
    user_id = "local"
    # Chainlit gives each chat session a unique ID. We use this to scope documents.
    session_id = cl.context.session.id
    history = cl.user_session.get("history") or []

    # 1. Handle file attachments (spontaneous upload)
    # Chainlit automatically puts uploaded files in message.elements
    if message.elements:
        files_to_ingest = []
        for element in message.elements:
            if element.type == "file":
                # Read the file bytes from the path Chainlit provides
                with open(element.path, "rb") as f:
                    file_bytes = f.read()
                
                import base64
                files_to_ingest.append({
                    "name": element.name,
                    "data": base64.b64encode(file_bytes).decode()
                })
        
        if files_to_ingest:
            async with cl.Step(name="Document Ingestion", type="tool") as step:
                step.input = f"Uploading {len(files_to_ingest)} file(s)"
                # Call your existing ingest_files with the session-scoped ID
                res = await cl.make_async(ingest_files)(files_to_ingest, user_id, session_id)
                step.output = res
                await step.update()
            
            await cl.Message(content=f"📁 {res}").send()
            
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
        # Call your existing streaming function, passing the session-scoped ID
        async for kind, content in ask_agent_v2_stream(
            message.content, user_id, history, session_id
        ):
            if kind == "step":
                # Show the routing step as a collapsible panel
                async with cl.Step(name=content, type="tool") as step:
                    step.output = "Running..."
                    await step.update()
                    
            elif kind == "chunk":
                got_answer = True
                await final_answer.stream_token(content)
                
            elif kind == "answer":
                # Fallback for non-streaming responses
                got_answer = True
                final_answer.content = content
                await final_answer.update()
                
    except Exception as e:
        final_answer.content = f"**Error:** {type(e).__name__}: {e}"
        await final_answer.update()
        got_answer = True

    # 4. If no answer was produced, show a fallback
    if not got_answer:
        final_answer.content = "**No answer was returned.** Please try again."
        await final_answer.update()

    await final_answer.update()
    
    # 5. Update the conversation history for the next turn
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": final_answer.content})
    cl.user_session.set("history", history)


# Optional: A custom "New Chat" button in the UI header
@cl.on_chat_resume
async def on_chat_resume(thread):
    """Called when a user resumes an old chat."""
    # We could load history here, but for now, start fresh
    cl.user_session.set("history", [])