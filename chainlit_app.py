import chainlit as cl
import asyncio
import io
import os
from multi_agent import ask_agent_v2_stream, ingest_files
from database import ensure_dbs_exist
from faster_whisper import WhisperModel

ensure_dbs_exist()

# --- LOAD WHISPER MODEL (Runs locally on your Mac for free) ---
print("Loading local Whisper model (base)...")
# 'base' is a good balance of speed and accuracy for CPU
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded.")

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("last_prompt", "")
    cl.user_session.set("active_task", None)
    
    await cl.Message(
        content="✦ **Nexus** — Your generalist AI assistant.\n\n"
                "Ask me anything. You can type, upload files, or **click the microphone icon to speak**."
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    cl.user_session.set("history", [])
    cl.user_session.set("last_prompt", "")
    cl.user_session.set("active_task", None)
    await cl.Message(content="Resumed session.").send()


# --- STOP CONTROL ---
@cl.on_stop
async def on_stop():
    task = cl.user_session.set("active_task")
    if task and not task.done():
        task.cancel()
        await cl.Message(content="⏹️ **Generation stopped.**").send()


# --- REGENERATE ---
@cl.action_callback("regenerate")
async def on_regenerate(action: cl.Action):
    last_prompt = cl.user_session.get("last_prompt", "")
    if not last_prompt:
        await cl.Message(content="No previous prompt to regenerate.").send()
        return
    
    await cl.Message(content="🔄 **Regenerating...**").send()
    await process_message(last_prompt, is_regenerate=True)


# --- AUDIO INPUT (Speech-to-Text) ---
@cl.on_audio_start
async def on_audio_start():
    # Initialize the audio buffer for this recording
    cl.user_session.set("audio_buffer", io.BytesIO())
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_buffer = cl.user_session.get("audio_buffer")
    if audio_buffer is not None:
        audio_buffer.write(chunk.data)

@cl.on_audio_end
async def on_audio_end():
    audio_buffer = cl.user_session.get("audio_buffer")
    if audio_buffer is None:
        return
    
    # Reset buffer
    audio_buffer.seek(0)
    
    # Notify user
    msg = cl.Message(content="🎙️ **Transcribing voice...**")
    await msg.send()
    
    try:
        # Run Whisper transcription in a separate thread to not block the UI
        segments, info = await cl.make_async(whisper_model.transcribe)(audio_buffer)
        transcription = "".join([segment.text for segment in segments]).strip()
        
        if not transcription:
            msg.content = "🎙️ *Could not hear anything. Please try again.*"
            await msg.update()
            return
        
        # Show what was heard
        msg.content = f"🎙️ **You said:** {transcription}"
        await msg.update()
        
        # Pass the transcribed text into the agent
        cl.user_session.set("last_prompt", transcription)
        await process_message(transcription, is_regenerate=False)
        
    except Exception as e:
        msg.content = f"**Transcription error:** {e}"
        await msg.update()


# --- MAIN MESSAGE HANDLER (Text) ---
@cl.on_message
async def on_message(message: cl.Message):
    # Handle file uploads
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


# --- THE STREAMING PROCESSOR ---
async def process_message(prompt: str, is_regenerate: bool = False):
    user_id = "local"
    session_id = cl.context.session.id
    history = cl.user_session.get("history") or []
    
    has_files = False
    try:
        from multi_agent import chroma_client
        collection = chroma_client.get_collection(name=f"docs_{user_id}_{session_id}")
        if collection.count() > 0:
            has_files = True
    except Exception:
        has_files = False
    
    msg = cl.Message(content="")
    await msg.send()
    
    full_response = ""
    current_step_msg = None
    current_step_text = ""
    
    try:
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
                if content != current_step_text:
                    current_step_text = content
                    if current_step_msg is not None:
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
            
            active_task = asyncio.create_task(anext_wrapper(agent_gen))
            cl.user_session.set("active_task", active_task)
    
    except asyncio.CancelledError:
        if not full_response:
            full_response = "⏹️ *Generation stopped by user.*"
            msg.content = full_response
            await msg.update()
        else:
            await msg.update()
    
    except Exception as e:
        msg.content = f"**Error:** {type(e).__name__}: {e}"
        await msg.update()
        full_response = msg.content
    
    finally:
        cl.user_session.set("active_task", None)
    
    if not is_regenerate:
        history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history)
    
    await cl.Message(
        content="",
        actions=[
            cl.Action(name="regenerate", value="regen", label="🔄 Regenerate", payload={})
        ]
    ).send()


async def anext_wrapper(gen):
    return await gen.__anext__()