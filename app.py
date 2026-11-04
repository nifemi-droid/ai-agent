from nicegui import ui, app
from multi_agent import ask_agent_v2_stream, generate_chat_title, ingest_files
from database import (ensure_dbs_exist, create_session, get_sessions, load_messages,
                      save_message, delete_session, update_user_password,
                      get_user_email, get_all_sessions_for_user,
                      delete_last_message, check_rate_limit, get_rate_limit_status)
import base64
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
storage_secret = os.getenv("NICEGUI_STORAGE_SECRET")
if not storage_secret or len(storage_secret) < 32:
    raise RuntimeError("Set NICEGUI_STORAGE_SECRET (32+ chars) in your .env")

ensure_dbs_exist()
current_task = None
USER_ID = "local"

GLOBAL_CSS = """
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
    body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
    
    /* Make the native uploader look like a small clean icon */
    .clean-uploader .q-uploader__header {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        min-height: auto !important;
        width: auto !important;
    }
    .clean-uploader .q-uploader__list { display: none !important; }
    .clean-uploader .q-uploader__header-content { padding: 0 !important; }
    .clean-uploader .q-btn {
        padding: 4px 8px !important;
        min-height: auto !important;
        font-size: 0.875rem;
    }
    .clean-uploader .q-uploader__title { display: none !important; }
    .clean-uploader .q-uploader__subtitle { display: none !important; }

    /* --- TABLE STYLING (PM Specs) --- */
    .prose table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-size: 0.9rem;
        background-color: #1f2937;
        border-radius: 8px;
        overflow: hidden;
    }
    .prose th, .prose td {
        padding: 12px 16px;
        border-bottom: 1px solid #374151;
        text-align: left;
        color: #e5e7eb;
    }
    .prose th {
        background-color: #111827;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        color: #9ca3af;
    }
    .prose tr:hover {
        background-color: #1f2937;
    }
    .prose tr:last-child td {
        border-bottom: none;
    }

    /* --- CODE BLOCK STYLING --- */
    .prose pre {
        background-color: #0d1117 !important;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        overflow-x: auto;
        position: relative;
        margin: 1.5rem 0;
    }
    .prose code {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
        color: #e6edf3;
    }
</style>
"""

@ui.page('/')
def main_page(session: str = None):
    global current_task
    ui.add_head_html(GLOBAL_CSS, shared=True)
    ui.dark_mode().enable()

    valid_sessions = [s[0] for s in get_sessions(USER_ID)]
    
    if session and session in valid_sessions:
        app.storage.user['current_session'] = session
    elif 'current_session' not in app.storage.user or app.storage.user['current_session'] not in valid_sessions:
        if valid_sessions:
            app.storage.user['current_session'] = valid_sessions[0]
        else:
            app.storage.user['current_session'] = create_session(USER_ID, "New Chat")

    # Header
    with ui.header().classes('bg-gray-900 border-b border-gray-800 px-4 py-2 items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.label('✦').classes('text-blue-400 text-lg')
            ui.label('Nexus').classes('text-lg font-bold text-white')
        with ui.row().classes('items-center gap-3'):
            remaining = get_rate_limit_status(USER_ID)
            color = 'text-green-400' if remaining > 5 else 'text-yellow-400' if remaining > 0 else 'text-red-400'
            ui.label(f'⚡ {remaining}').classes(f'text-sm {color}')
            ui.button('⚙️ Settings', on_click=lambda: ui.navigate.to('/settings')).props('flat color=white size=sm')

    # Layout
    with ui.row().classes('w-full no-wrap h-[calc(100vh-56px)] bg-gray-900'):
        
        # Sidebar
        with ui.column().classes('w-72 bg-gray-950 h-full overflow-y-auto border-r border-gray-800 p-3'):
            ui.button('+ New Chat', on_click=lambda: new_chat()).props('flat color=blue-400 align=left').classes('w-full mb-4')
            chat_list = ui.column().classes('w-full gap-1')

            def render_sidebar():
                chat_list.clear()
                for sid, title in get_sessions(USER_ID):
                    active = sid == app.storage.user['current_session']
                    cls = 'bg-gray-800 text-white' if active else 'text-gray-300 hover:bg-gray-800'
                    with chat_list:
                        with ui.row().classes(f'w-full items-center justify-between rounded p-2 cursor-pointer {cls}'):
                            ui.label(title).classes('truncate text-sm flex-grow').on('click', lambda s=sid: switch_session(s))
                            ui.button('×', on_click=lambda s=sid: remove_chat(s)).props('flat dense size=sm color=grey-5')

            def new_chat():
                new_id = create_session(USER_ID, "New Chat")
                ui.navigate.to(f'/?session={new_id}')

            def switch_session(sid):
                if sid == app.storage.user.get('current_session'):
                    return
                ui.navigate.to(f'/?session={sid}')

            def remove_chat(sid):
                delete_session(sid, USER_ID)
                if app.storage.user.get('current_session') == sid:
                    sessions = get_sessions(USER_ID)
                    if sessions:
                        ui.navigate.to(f'/?session={sessions[0][0]}')
                    else:
                        new_id = create_session(USER_ID, "New Chat")
                        ui.navigate.to(f'/?session={new_id}')

            render_sidebar()

        # Main Chat Area
        with ui.column().classes('flex-grow h-full flex flex-col relative'):
            
            # JS for tracking scroll AND adding copy buttons to code blocks
            ui.add_head_html('''
                <script>
                    window.scrollToBottom = function() {
                        const el = document.querySelector('.q-page-container');
                        if (el) el.scrollTop = el.scrollHeight;
                        const btn = document.getElementById('scroll-btn');
                        if (btn) btn.style.display = 'none';
                    };
                    
                    window.checkScroll = function() {
                        const el = document.querySelector('.q-page-container');
                        const btn = document.getElementById('scroll-btn');
                        if (el && btn) {
                            if (el.scrollHeight - el.scrollTop - el.clientHeight > 200) {
                                btn.style.display = 'flex';
                            } else {
                                btn.style.display = 'none';
                            }
                        }
                    };

                    function addCopyButtons() {
                        document.querySelectorAll('pre').forEach(pre => {
                            if (!pre.querySelector('.copy-code-btn')) {
                                const btn = document.createElement('button');
                                btn.className = 'copy-code-btn';
                                btn.style.cssText = 'position:absolute; top:8px; right:8px; background:#374151; color:#e5e7eb; border:none; padding:4px 8px; border-radius:4px; font-size:12px; cursor:pointer; transition: background 0.2s;';
                                btn.innerText = 'Copy';
                                btn.onmouseover = () => btn.style.background = '#4b5563';
                                btn.onmouseout = () => btn.style.background = '#374151';
                                btn.onclick = () => {
                                    const code = pre.querySelector('code');
                                    if (code) {
                                        navigator.clipboard.writeText(code.innerText).then(() => {
                                            btn.innerText = 'Copied!';
                                            setTimeout(() => btn.innerText = 'Copy', 2000);
                                        });
                                    }
                                };
                                pre.style.position = 'relative';
                                pre.appendChild(btn);
                            }
                        });
                    }

                    setInterval(addCopyButtons, 1000);
                </script>
            ''')

            chat_box = ui.column().classes('w-full max-w-3xl mx-auto gap-4 overflow-y-auto flex-grow p-4 pb-40')
            
            def render_chat():
                chat_box.clear()
                msgs = load_messages(app.storage.user['current_session'], USER_ID)
                if not msgs:
                    with chat_box:
                        with ui.column().classes('w-full items-center justify-center mt-32 opacity-50'):
                            ui.label('✦').classes('text-6xl text-blue-400 mb-4')
                            ui.label('How can I help you today?').classes('text-2xl text-gray-400')
                
                for i, m in enumerate(msgs):
                    if m['role'] == 'user':
                        with chat_box:
                            with ui.row().classes('w-full justify-end items-start gap-2'):
                                ui.button('Edit', on_click=lambda idx=i, c=m['content']: edit_prompt(idx, c)).props('flat dense size=sm color=grey-5')
                                with ui.card().classes('bg-blue-600 text-white rounded-2xl max-w-[80%] px-4 py-2'):
                                    ui.markdown(m['content'])
                    else:
                        with chat_box:
                            with ui.row().classes('w-full justify-start items-start gap-3'):
                                with ui.avatar().classes('bg-blue-500'):
                                    ui.label('✦').classes('text-white')
                                with ui.column().classes('flex-grow max-w-[85%] prose prose-invert prose-sm'):
                                    
                                    raw_content = m['content']
                                    display_text = raw_content
                                    urls = []
                                    
                                    if "[SOURCES]" in raw_content:
                                        parts = raw_content.split("[SOURCES]")
                                        display_text = parts[0].strip()
                                        if len(parts) > 1:
                                            urls = [u.strip() for u in parts[1].split("|") if u.strip()]

                                    prefix = display_text[:20]
                                    is_err = prefix.startswith("Web search failed") or prefix.startswith("Document error") or prefix.startswith("**Error:**")
                                    color = 'text-red-400' if is_err else 'text-gray-200'
                                    ui.markdown(display_text).classes(color)
                                    
                                    if urls:
                                        with ui.expansion(f'🌐 View {len(urls)} Sources', icon='link').classes('w-full text-xs text-gray-400 mt-4 bg-gray-800/50 rounded'):
                                            for u in urls:
                                                if u.startswith("http"):
                                                    ui.link(u, u).classes('text-blue-400 block truncate mb-1 hover:underline')
                                    
                                    # --- EXPORT & REGENERATE BUTTONS ---
                                    with ui.row().classes('mt-2 gap-2 items-center'):
                                        def download_export(text_content=display_text, idx=i):
                                            ui.download(text_content.encode('utf-8'), f"nexus_export_{idx}.md")
                                            
                                        ui.button('⬇️ Export to Markdown', on_click=download_export)\
                                            .props('flat dense size=sm color=grey-5')\
                                            .tooltip('Download this document as a Markdown file')
                                            
                                        if i == len(msgs) - 1:
                                            ui.button('🔄 Regenerate', on_click=lambda: regenerate())\
                                                .props('flat dense size=sm color=grey-5')
                
                ui.run_javascript('setTimeout(scrollToBottom, 200);')

            render_chat()

            # Scroll Down Button
            scroll_btn = ui.button('↓', on_click=lambda: ui.run_javascript('scrollToBottom()'))\
                .props('round dense')\
                .classes('fixed bottom-28 right-[calc(50%-400px)] bg-gray-800 text-gray-400 border border-gray-700 shadow-lg z-50')\
                .style('display: none;')
                
            chat_box.on('scroll', lambda: ui.run_javascript('checkScroll()'))

            ui.run_javascript('''
                setTimeout(function() {
                    const input = document.querySelector('textarea');
                    if (input) input.focus();
                }, 500);
            ''')

            async def edit_prompt(idx, content):
                while len(load_messages(app.storage.user['current_session'], USER_ID)) > idx:
                    delete_last_message(app.storage.user['current_session'], USER_ID)
                render_chat()
                input_box.value = content
                ui.run_javascript('const t=document.querySelector("textarea"); if(t) t.focus();')

            async def regenerate():
                msgs = load_messages(app.storage.user['current_session'], USER_ID)
                if not msgs or msgs[-1]['role'] != 'assistant':
                    return
                delete_last_message(app.storage.user['current_session'], USER_ID)
                prompt = msgs[-2]['content'] if len(msgs) >= 2 else ""
                render_chat()
                if prompt:
                    await run_agent(prompt)

            # Input Area
            with ui.row().classes('w-full max-w-3xl mx-auto px-4 pb-6 absolute bottom-0 left-0 right-0'):
                with ui.card().classes('w-full bg-gray-800 rounded-2xl flex-col p-2 border border-gray-700'):
                    
                    async def handle_native_upload(e):
                        file_bytes = e.content.read()
                        if len(file_bytes) > 10 * 1024 * 1024:
                            ui.notify('File exceeds 10 MB limit.', type='negative')
                            return
                        
                        res = await ui.run.io_bound(ingest_files, [{"name": e.name, "data": base64.b64encode(file_bytes).decode()}], USER_ID)
                        ui.notify(res, type='info')
                        with chat_box:
                            with ui.row().classes('w-full justify-center'):
                                ui.badge(f"📁 {res}", color='gray-700').classes('text-gray-300 text-xs')
                        ui.run_javascript('setTimeout(scrollToBottom, 100);')

                    # Use the clean native uploader
                    uploader = ui.upload(on_upload=handle_native_upload, auto_upload=True)\
                        .props('accept=".pdf,.docx,.doc,.txt,.md,.csv,.xlsx,.xls,.pptx"')\
                        .classes('clean-uploader')

                    input_box = ui.textarea(placeholder='Ask anything — Enter to send, Shift+Enter for new line')\
                        .props('outlined dense autogrow')\
                        .classes('w-full bg-transparent text-white')

                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('gap-2 items-center'):
                            # The label is clickable and triggers the uploader
                            ui.label('📎 Attach').classes('text-gray-400 text-sm cursor-pointer hover:text-white')\
                                .on('click', lambda: uploader.run_method('pickFiles'))
                            
                        with ui.row().classes('gap-2'):
                            stop_btn = ui.button('Stop', on_click=lambda: stop_generation()).props('flat dense color=red-400').classes('hidden')
                            send_btn = ui.button('Send', on_click=lambda: send()).props('flat dense color=blue-400')

                    async def on_key(e):
                        if e.args.get('key') == 'Enter' and not e.args.get('shiftKey'):
                            await send()

                    input_box.on('keydown', on_key)

                    async def send():
                        prompt = (input_box.value or '').strip()
                        if not prompt:
                            return
                        allowed, remaining = check_rate_limit(USER_ID)
                        if not allowed:
                            ui.notify('Rate limit exceeded. Wait 60s.', type='negative')
                            return
                        input_box.value = ''
                        save_message(app.storage.user['current_session'], 'user', prompt, USER_ID)
                        
                        with chat_box:
                            with ui.row().classes('w-full justify-end items-start gap-2'):
                                idx = len(load_messages(app.storage.user['current_session'], USER_ID)) - 1
                                ui.button('Edit', on_click=lambda idx=idx, c=prompt: edit_prompt(idx, c)).props('flat dense size=sm color=grey-5')
                                with ui.card().classes('bg-blue-600 text-white rounded-2xl max-w-[80%] px-4 py-2'):
                                    ui.markdown(prompt)
                        ui.run_javascript('scrollToBottom()')
                        
                        await run_agent(prompt)

                    async def run_agent(prompt):
                        global current_task
                        stop_btn.classes(remove='hidden')
                        send_btn.classes(add='hidden')
                        
                        with chat_box:
                            with ui.row().classes('w-full justify-start items-start gap-3') as row:
                                with ui.avatar().classes('bg-blue-500'):
                                    ui.label('✦').classes('text-white')
                                with ui.column().classes('flex-grow prose prose-invert prose-sm'):
                                    status = ui.label('Thinking...').classes('text-gray-400 italic text-xs')
                                    answer_container = ui.column().classes('w-full')
                        
                        ui.run_javascript('scrollToBottom()')
                        
                        final = ""
                        got_answer = False
                        try:
                            current_task = asyncio.current_task()
                            async for kind, content in ask_agent_v2_stream(prompt, USER_ID):
                                if kind == "step":
                                    status.text = content
                                elif kind == "chunk":
                                    status.visible = False
                                    got_answer = True
                                    final += content
                                    
                                    visible_text = final.split("[SOURCES]")[0] if "[SOURCES]" in final else final
                                    
                                    answer_container.clear()
                                    with answer_container:
                                        ui.markdown(visible_text).classes('text-gray-200')
                                    ui.run_javascript('scrollToBottom()')
                                elif kind == "answer":
                                    final = content
                                    got_answer = True
                                    status.visible = False
                                    answer_container.clear()
                                    
                                    if "[SOURCES]" in final:
                                        parts = final.split("[SOURCES]")
                                        display_text = parts[0].strip()
                                        urls = [u.strip() for u in parts[1].split("|") if u.strip()]
                                        
                                        with answer_container:
                                            ui.markdown(display_text).classes('text-gray-200')
                                            with ui.expansion(f'🌐 View {len(urls)} Sources', icon='link').classes('w-full text-xs text-gray-400 mt-4 bg-gray-800/50 rounded'):
                                                for u in urls:
                                                    if u.startswith("http"):
                                                        ui.link(u, u).classes('text-blue-400 block truncate mb-1 hover:underline')
                                    else:
                                        with answer_container:
                                            ui.markdown(final).classes('text-gray-200')
                                            
                        except asyncio.CancelledError:
                            status.text = 'Stopped.'
                            if not final:
                                final = 'Generation stopped.'
                            got_answer = True
                        except Exception as e:
                            print(f"AGENT ERROR: {type(e).__name__}: {e}")
                            status.visible = False
                            answer_container.clear()
                            err = f"**Error:** {type(e).__name__}: {e}"
                            with answer_container:
                                ui.markdown(err).classes('text-red-400')
                            final = err
                            got_answer = True
                        finally:
                            current_task = None
                            stop_btn.classes(add='hidden')
                            send_btn.classes(remove='hidden')
                        
                        if not got_answer:
                            status.visible = False
                            answer_container.clear()
                            with answer_container:
                                ui.markdown("**No answer was returned.** Check the terminal for errors.").classes('text-red-400')
                            final = "No answer was returned."
                        
                        # Add Export button to live stream
                        if got_answer and not final.startswith("**Error"):
                            clean_export_text = final.split("[SOURCES]")[0].strip() if "[SOURCES]" in final else final
                            def download_live_export(text_content=clean_export_text):
                                ui.download(text_content.encode('utf-8'), "nexus_export.md")
                                
                            with answer_container:
                                with ui.row().classes('mt-2 gap-2'):
                                    ui.button('⬇️ Export to Markdown', on_click=download_live_export)\
                                        .props('flat dense size=sm color=grey-5')\
                                        .tooltip('Download this document as a Markdown file')

                        save_message(app.storage.user['current_session'], 'assistant', final, USER_ID)
                        msgs = load_messages(app.storage.user['current_session'], USER_ID)
                        if len(msgs) == 2:
                            from database import DB_NAME
                            import sqlite3
                            title = generate_chat_title(prompt)
                            c = sqlite3.connect(DB_NAME); cur = c.cursor()
                            cur.execute("UPDATE sessions SET title=? WHERE id=?", (title, app.storage.user['current_session']))
                            c.commit(); c.close()
                        render_sidebar()

                    def stop_generation():
                        global current_task
                        if current_task:
                            current_task.cancel()

@ui.page('/settings')
def settings_page():
    ui.add_head_html(GLOBAL_CSS, shared=True)
    
    with ui.header().classes('bg-gray-900 px-4 py-3 border-b border-gray-800 items-center justify-between'):
        ui.label('Settings').classes('text-lg font-bold text-white')
        ui.button('Back to Chat', on_click=lambda: ui.navigate.to('/')).props('flat color=white')

    with ui.column().classes('w-full max-w-3xl mx-auto p-6 gap-6 bg-gray-900 min-h-screen'):
        
        with ui.card().classes('w-full bg-gray-800 p-6 rounded-xl'):
            ui.label('My Chat History').classes('text-white font-semibold mb-4 text-lg')
            
            sessions = get_all_sessions_for_user(USER_ID)
            if not sessions:
                ui.label('No chat history yet.').classes('text-gray-500 text-sm italic')
            else:
                columns = [
                    {'name': 'title', 'label': 'Title', 'field': 'title', 'align': 'left'},
                    {'name': 'date', 'label': 'Created', 'field': 'date'},
                    {'name': 'actions', 'label': '', 'field': 'id'},
                ]
                rows = [{'id': s[0], 'title': s[1], 'date': s[2]} for s in sessions]
                t = ui.table(columns=columns, rows=rows, row_key='id').classes('w-full bg-gray-800 text-white')
                t.add_slot('body-cell-actions', '''
                    <q-td :props="props">
                        <q-btn label="Delete" flat dense color="red" @click="() => $parent.$emit('del', props.row.id)" />
                    </q-td>
                ''')
                def do_del(e):
                    delete_session(e.args, USER_ID)
                    ui.notify('Chat deleted. Refreshing...', type='info')
                    ui.navigate.to('/settings')
                t.on('del', do_del)

        with ui.card().classes('w-full bg-gray-800 p-6 rounded-xl border border-red-900'):
            ui.label('Danger Zone').classes('text-red-400 font-semibold mb-2 text-lg')
            ui.label('Permanently delete all chats and uploaded documents. This cannot be undone.').classes('text-gray-400 text-sm mb-4')
            
            async def clear_all():
                sessions = get_all_sessions_for_user(USER_ID)
                for sid, _, _ in sessions:
                    delete_session(sid, USER_ID)
                try:
                    from multi_agent import chroma_client
                    chroma_client.delete_collection(name=f"docs_{USER_ID}")
                except Exception:
                    pass
                ui.notify('All data cleared.', type='positive')
                ui.navigate.to('/')

            ui.button('Delete All My Data', on_click=clear_all).props('rounded color=negative')

ui.run(title='Nexus', storage_secret=storage_secret, dark=True, favicon='✦',
       reload=False, host='127.0.0.1', port=int(os.getenv('PORT', '8080')))