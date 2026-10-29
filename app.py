from nicegui import ui, app
from multi_agent import ask_agent_v2_stream, generate_chat_title, ingest_files
from database import (ensure_dbs_exist, create_user, authenticate_user,
                      create_session, get_sessions, load_messages, save_message,
                      delete_session, get_user_email, update_user_password,
                      get_user_role, get_all_sessions_for_user,
                      delete_last_message, check_rate_limit, get_rate_limit_status,
                      get_user_joined_date)
import base64
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
storage_secret = os.getenv("NICEGUI_STORAGE_SECRET")
if not storage_secret or len(storage_secret) < 32:
    raise RuntimeError("NICEGUI_STORAGE_SECRET must be set to a random secret of at least 32 characters")

ensure_dbs_exist()
current_task = None

GLOBAL_CSS = """
<style>
    body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
    .modern-input:focus-within { box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5); }
</style>
"""

def is_logged_in():
    return 'user_id' in app.storage.user

@ui.page('/login')
def login_page():
    if is_logged_in():
        ui.navigate.to('/')
        return
    ui.add_head_html(GLOBAL_CSS, shared=True)
    with ui.column().classes('w-full h-screen items-center justify-center bg-gray-900'):
        with ui.card().classes('w-96 p-8 rounded-2xl bg-gray-800 border border-gray-700'):
            ui.label('Nexus AI').classes('text-2xl text-center mb-8 font-bold text-white')
            email = ui.input('Email').props('outlined dense rounded dark').classes('w-full mb-4')
            password = ui.input('Password', password=True, password_toggle_button=True).props('outlined dense rounded dark').classes('w-full mb-6')

            async def handle_login():
                uid = authenticate_user(email.value, password.value)
                if uid:
                    app.storage.user['user_id'] = uid
                    ui.navigate.to('/')
                else:
                    ui.notify('Invalid credentials', type='negative')

            async def handle_register():
                if len(password.value) < 12:
                    ui.notify('Password must be at least 12 characters', type='warning')
                    return
                uid = create_user(email.value, password.value)
                if uid:
                    app.storage.user['user_id'] = uid
                    ui.navigate.to('/')
                else:
                    ui.notify('Email already exists', type='warning')

            with ui.row().classes('w-full gap-2'):
                ui.button('Login', on_click=handle_login).props('rounded').classes('flex-1')
                ui.button('Register', on_click=handle_register).props('flat').classes('flex-1')


@ui.page('/')
def main_page():
    global current_task
    if not is_logged_in():
        ui.navigate.to('/login')
        return
    ui.add_head_html(GLOBAL_CSS, shared=True)

    user_id = app.storage.user['user_id']
    if 'current_session' not in app.storage.user:
        sessions = get_sessions(user_id)
        app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")

    ui.dark_mode().enable()

    # ---------------- header ----------------
    with ui.header().classes('bg-gray-900 border-b border-gray-800 px-4 py-2 items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.label('✦').classes('text-blue-400 text-lg')
            ui.label('Nexus AI').classes('text-lg font-bold text-white')
        with ui.row().classes('items-center gap-3'):
            remaining = get_rate_limit_status(user_id)
            color = 'text-green-400' if remaining > 5 else 'text-yellow-400' if remaining > 0 else 'text-red-400'
            ui.label(f'⚡ {remaining}').classes(f'text-sm {color}')
            def logout():
                app.storage.user.pop('user_id', None)
                app.storage.user.pop('current_session', None)
                ui.navigate.to('/login')
            ui.button('Logout', on_click=logout).props('flat color=white size=sm')
            ui.button('Settings', on_click=lambda: ui.navigate.to('/settings')).props('flat color=white size=sm')

    # ---------------- layout ----------------
    with ui.row().classes('w-full no-wrap h-[calc(100vh-56px)] bg-gray-900'):
        # Sidebar
        with ui.column().classes('w-72 bg-gray-950 h-full overflow-y-auto border-r border-gray-800 p-3'):
            ui.button('+ New Chat', on_click=lambda: new_chat()).props('flat color=blue-400 align=left').classes('w-full mb-4')

            chat_list = ui.column().classes('w-full gap-1')

            def render_sidebar():
                chat_list.clear()
                for sid, title in get_sessions(user_id):
                    active = sid == app.storage.user['current_session']
                    cls = 'bg-gray-800 text-white' if active else 'text-gray-300 hover:bg-gray-800'
                    with chat_list:
                        with ui.row().classes(f'w-full items-center justify-between rounded p-2 cursor-pointer {cls}'):
                            ui.label(title).classes('truncate text-sm flex-grow').on('click', lambda s=sid: switch_session(s))
                            ui.button('×', on_click=lambda s=sid: remove_chat(s)).props('flat dense size=sm color=grey-5')

            def new_chat():
                app.storage.user['current_session'] = create_session(user_id, "New Chat")
                render_sidebar()
                render_chat()

            def switch_session(sid):
                if sid == app.storage.user['current_session']:
                    return
                app.storage.user['current_session'] = sid
                render_sidebar()
                render_chat()

            def remove_chat(sid):
                delete_session(sid, user_id)
                if app.storage.user['current_session'] == sid:
                    sessions = get_sessions(user_id)
                    app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")
                render_sidebar()
                render_chat()

            render_sidebar()

        # Main area
        with ui.column().classes('flex-grow h-full flex flex-col relative'):
            chat_box = ui.column().classes('w-full max-w-3xl mx-auto gap-4 overflow-y-auto flex-grow p-4 pb-40')

            def render_chat():
                chat_box.clear()
                msgs = load_messages(app.storage.user['current_session'], user_id)
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
                                with ui.card().classes('bg-blue-600 text-white rounded-2xl max-w-[80%] px-4 py-2 shadow-lg border-none'):
                                    ui.markdown(m['content'])
                    else:
                        with chat_box:
                            with ui.row().classes('w-full justify-start items-start gap-3'):
                                with ui.avatar().classes('bg-blue-500'):
                                    ui.label('✦').classes('text-white')
                                with ui.column().classes('flex-grow max-w-[85%] prose prose-invert prose-sm'):
                                    if "[SOURCES]" in m['content']:
                                        head, tail = m['content'].split("[SOURCES]", 1)
                                        ui.markdown(head).classes('text-gray-200')
                                        ui.label('Sources').classes('text-xs font-bold text-gray-500 mt-2')
                                        for url in tail.split("|"):
                                            if url.strip().startswith("http"):
                                                ui.link(url.strip(), url.strip()).classes('text-xs text-blue-400 block truncate')
                                    else:
                                        ui.markdown(m['content']).classes('text-gray-200')
                                    if i == len(msgs) - 1:
                                        ui.button('Regenerate', on_click=lambda: regenerate()).props('flat dense size=sm color=grey-5').classes('mt-2')
                ui.run_javascript('const e=document.querySelector(".q-page-container"); if(e) e.scrollTop=e.scrollHeight;')

            render_chat()

            async def edit_prompt(idx, content):
                # Delete the user message and the assistant reply that followed it
                msgs = load_messages(app.storage.user['current_session'], user_id)
                # Remove all messages from idx onward, then repopulate
                # Simplest safe approach: delete last message until idx remains
                while len(load_messages(app.storage.user['current_session'], user_id)) > idx:
                    delete_last_message(app.storage.user['current_session'], user_id)
                render_chat()
                input_box.value = content
                ui.run_javascript('document.querySelector("textarea").focus();')

            async def regenerate():
                msgs = load_messages(app.storage.user['current_session'], user_id)
                if not msgs or msgs[-1]['role'] != 'assistant':
                    return
                delete_last_message(app.storage.user['current_session'], user_id)
                prompt = msgs[-2]['content'] if len(msgs) >= 2 else ""
                render_chat()
                if prompt:
                    await run_agent(prompt)

            # ---------------- input area ----------------
            with ui.row().classes('w-full max-w-3xl mx-auto px-4 pb-6 absolute bottom-0 left-0 right-0'):
                with ui.card().classes('w-full bg-gray-800 shadow-2xl rounded-2xl flex-col p-2 border border-gray-700 modern-input'):
                    # file input (hidden)
                    file_input = ui.element('input')\
                        .props('type=file multiple accept=".pdf,.docx,.doc,.txt,.md,.csv,.xlsx,.xls,.pptx" style="display: none;"')\
                        .on('change', lambda e: ui.run_javascript('''
                            const f=document.querySelector("input[type=file]").files;
                            const files=Array.from(f).slice(0,3);
                            let total=0, out=[], pending=files.length;
                            if(!pending) return;
                            files.forEach(file=>{
                                if(file.size>10*1024*1024){emitEvent("upload_error",{msg:"File exceeds 10 MB"}); pending--; return;}
                                total+=file.size;
                                if(total>25*1024*1024){emitEvent("upload_error",{msg:"Total exceeds 25 MB"}); pending--; return;}
                                const r=new FileReader();
                                r.onload=e=>{
                                    out.push({name:file.name,data:e.target.result.split(",")[1]});
                                    pending--;
                                    if(pending===0&&out.length) emitEvent("files_ready",{files:out});
                                };
                                r.readAsDataURL(file);
                            });
                        '''))

                    async def on_files(e):
                        files = e.args.get('files', [])
                        res = await ui.run.io_bound(ingest_files, files, user_id)
                        ui.notify(res, type='info')

                    def on_err(e):
                        ui.notify(e.args.get('msg', 'Upload failed'), type='negative')

                    ui.on('files_ready', on_files)
                    ui.on('upload_error', on_err)

                    # textarea (multi-line)
                    input_box = ui.textarea(placeholder='Ask anything — Enter to send, Shift+Enter for new line')\
                        .props('outlined dense autogrow input-style="color:#fff"')\
                        .classes('w-full bg-transparent text-white')

                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('gap-2'):
                            ui.button('📎 Attach', on_click=lambda: ui.run_javascript('document.querySelector("input[type=file]").click()')).props('flat dense color=grey-5')
                            ui.label('').classes('text-xs text-gray-500').bind_text_from(input_box, 'value', lambda v: f'{len(v or "")} chars')
                        with ui.row().classes('gap-2'):
                            stop_btn = ui.button('Stop', on_click=lambda: stop_generation()).props('flat dense color=red-400').classes('hidden')
                            send_btn = ui.button('Send', on_click=lambda: send()).props('flat dense color=blue-400')

                    # Enter sends, Shift+Enter newline
                    input_box.on('keydown', lambda e: _handle_key(e, send))
                    def _handle_key(e, send_fn):
                        if e.args.get('key') == 'Enter' and not e.args.get('shiftKey'):
                            asyncio.create_task(send_fn())

                    async def send():
                        prompt = (input_box.value or '').strip()
                        if not prompt:
                            return
                        allowed, remaining = check_rate_limit(user_id)
                        if not allowed:
                            ui.notify('Rate limit exceeded. Wait 60s.', type='negative')
                            return
                        if remaining <= 3:
                            ui.notify(f'{remaining} messages left this minute', type='warning')
                        input_box.value = ''
                        save_message(app.storage.user['current_session'], 'user', prompt, user_id)
                        render_chat()
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
                                    status = ui.label('Thinking...').classes('text-gray-400 italic text-sm')
                                    holder = ui.column().classes('w-full')
                        final = ""
                        try:
                            current_task = asyncio.current_task()
                            async for kind, content in ask_agent_v2_stream(prompt, user_id):
                                if kind == "step":
                                    status.text = content
                                elif kind == "answer":
                                    final = content
                                    status.visible = False
                                    holder.clear()
                                    with holder:
                                        ui.markdown(final).classes('text-gray-200')
                                    ui.run_javascript('const e=document.querySelector(".q-page-container"); if(e) e.scrollTop=e.scrollHeight;')
                        except asyncio.CancelledError:
                            status.text = 'Stopped.'
                            if not final:
                                final = 'Generation stopped.'
                        finally:
                            current_task = None
                            stop_btn.classes(add='hidden')
                            send_btn.classes(remove='hidden')
                        save_message(app.storage.user['current_session'], 'assistant', final, user_id)
                        msgs = load_messages(app.storage.user['current_session'], user_id)
                        if len(msgs) == 2:
                            from database import DB_NAME
                            import sqlite3
                            title = generate_chat_title(prompt)
                            c = sqlite3.connect(DB_NAME); cur = c.cursor()
                            cur.execute("UPDATE sessions SET title=? WHERE id=? AND user_id=?", (title, app.storage.user['current_session'], user_id))
                            c.commit(); c.close()
                        render_sidebar()
                        render_chat()

                    def stop_generation():
                        global current_task
                        if current_task:
                            current_task.cancel()
                            ui.notify('Stop requested', type='info')


@ui.page('/settings')
def settings_page():
    if not is_logged_in():
        ui.navigate.to('/login')
        return
    ui.add_head_html(GLOBAL_CSS, shared=True)
    user_id = app.storage.user['user_id']
    email = get_user_email(user_id)
    role = get_user_role(user_id)
    joined = get_user_joined_date(user_id)

    with ui.header().classes('bg-gray-900 px-4 py-3 border-b border-gray-800 items-center justify-between'):
        ui.label('Settings').classes('text-lg font-bold text-white')
        ui.button('Back', on_click=lambda: ui.navigate.to('/')).props('flat color=white')

    with ui.column().classes('w-full max-w-3xl mx-auto p-6 gap-6 bg-gray-900 min-h-screen'):
        with ui.card().classes('w-full bg-gray-800 p-6 rounded-xl'):
            with ui.row().classes('items-center gap-4'):
                with ui.avatar((email[:2] if email else '??').upper()).classes('bg-blue-500 text-white'):
                    pass
                with ui.column().classes('gap-0'):
                    ui.label(email).classes('text-white font-semibold')
                    ui.label(f'Joined {joined}').classes('text-gray-400 text-xs')
            ui.separator()
            with ui.row().classes('items-center gap-2'):
                ui.label('Role:').classes('text-gray-400')
                ui.badge(role.upper()).classes('bg-blue-600 text-xs')

        with ui.card().classes('w-full bg-gray-800 p-6 rounded-xl'):
            ui.label('Change password').classes('text-white font-semibold mb-2')
            with ui.row().classes('gap-2 items-end w-full'):
                new_pw = ui.input('New Password', password=True, password_toggle_button=True).props('outlined dense').classes('flex-grow')
                async def update_pw():
                    if len(new_pw.value or '') < 12:
                        ui.notify('Minimum 12 characters', type='warning'); return
                    ok = update_user_password(user_id, new_pw.value)
                    ui.notify('Updated' if ok else 'Failed', type='positive' if ok else 'negative')
                    if ok: new_pw.value = ''
                ui.button('Update', on_click=update_pw).props('rounded')

        with ui.card().classes('w-full bg-gray-800 p-6 rounded-xl'):
            ui.label('My Chat History').classes('text-white font-semibold mb-2')
            sessions = get_all_sessions_for_user(user_id)
            if not sessions:
                ui.label('Empty').classes('text-gray-500 text-sm')
            else:
                columns = [
                    {'name': 'title', 'label': 'Title', 'field': 'title', 'align': 'left'},
                    {'name': 'date', 'label': 'Created', 'field': 'date'},
                    {'name': 'actions', 'label': '', 'field': 'id'},
                ]
                rows = [{'id': s[0], 'title': s[1], 'date': s[2]} for s in sessions]
                t = ui.table(columns=columns, rows=rows, row_key='id').classes('w-full')
                t.add_slot('body-cell-actions', '<q-td :props="props"><q-btn label="Delete" flat dense color="red" @click="() => $parent.$emit(\'del\', props.row.id)" /></q-td>')
                def do_del(e):
                    delete_session(e.args, user_id)
                    ui.notify('Deleted', type='info')
                    ui.navigate.to('/settings')
                t.on('del', do_del)


ui.run(title='Nexus AI', storage_secret=storage_secret, dark=True, favicon='✦', reload=False, host='127.0.0.1', port=int(os.getenv('PORT', '8080')))