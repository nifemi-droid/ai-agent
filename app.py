from nicegui import ui, app
from multi_agent import ask_agent_v2_stream, ingest_file, generate_chat_title
from database import (ensure_dbs_exist, create_user, authenticate_user, 
                      create_session, get_sessions, load_messages, save_message, 
                      delete_session, get_analytics, get_user_email, update_user_password, 
                      is_admin, get_user_role, get_all_sessions_for_user, get_analytics_by_route, 
                      delete_last_message, check_rate_limit, get_rate_limit_status)
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

# --- GLOBAL STYLES (safe, no user data) ---
GLOBAL_CSS = """
<style>
    /* Modern typography */
    * { font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
    
    /* Chat bubble animations */
    .chat-bubble-user { animation: slideInRight 0.2s ease-out; }
    .chat-bubble-ai { animation: slideInLeft 0.2s ease-out; }
    @keyframes slideInRight { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes slideInLeft { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
    
    /* Input glow on focus */
    .modern-input:focus-within { box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5); }
    
    /* Sidebar hover transitions */
    .chat-item { transition: background-color 0.15s ease; }
</style>
"""

def is_logged_in():
    return 'user_id' in app.storage.user

@ui.page('/login')
def login_page():
    if is_logged_in():
        ui.navigate.to('/')
        return
    ui.add_head_html(GLOBAL_CSS)
    with ui.column().classes('w-full h-screen items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800'):
        with ui.card().classes('w-96 p-8 shadow-2xl rounded-2xl bg-gray-800 border border-gray-700'):
            ui.label('🌌 Nexus AI').classes('text-2xl text-center mb-8 font-bold text-white')
            
            email = ui.input('Email').props('outlined dense rounded dark color=blue-400').classes('w-full mb-4')
            password = ui.input('Password', password=True, password_toggle_button=True).props('outlined dense rounded dark color=blue-400').classes('w-full mb-6')
            
            async def handle_login():
                user_id = authenticate_user(email.value, password.value)
                if user_id:
                    app.storage.user['user_id'] = user_id
                    ui.navigate.to('/')
                else:
                    ui.notify('Invalid email or password', type='negative', position='top')
                    
            async def handle_register():
                if len(password.value) < 12:
                    ui.notify('Password must be at least 12 characters', type='warning', position='top')
                    return
                user_id = create_user(email.value, password.value)
                if user_id:
                    app.storage.user['user_id'] = user_id
                    ui.notify('Account created!', type='positive', position='top')
                    ui.navigate.to('/')
                else:
                    ui.notify('Invalid registration details or email already exists', type='warning', position='top')
                    
            with ui.row().classes('w-full justify-between gap-2'):
                ui.button('Login', on_click=handle_login).props('rounded color=primary').classes('w-1/2')
                ui.button('Register', on_click=handle_register).props('flat color=primary').classes('w-1/2')

@ui.page('/')
def main_page():
    global current_task
    if not is_logged_in():
        ui.navigate.to('/login')
        return
    ui.add_head_html(GLOBAL_CSS)
    
    user_id = app.storage.user['user_id']
    if 'current_session' not in app.storage.user:
        sessions = get_sessions(user_id)
        app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")

    dark = ui.dark_mode()
    dark.enable()

    # --- SAFE STATIC JS (no user data) ---
    ui.add_head_html('''
        <script>
            document.addEventListener('click', function(e) {
                if (e.target && e.target.classList.contains('copy-code-btn')) {
                    const pre = e.target.closest('pre');
                    if (!pre) return;
                    const codeBlock = pre.querySelector('code');
                    if (!codeBlock) return;
                    navigator.clipboard.writeText(codeBlock.innerText).then(() => {
                        const originalText = e.target.innerText;
                        e.target.innerText = 'Copied!';
                        setTimeout(() => e.target.innerText = originalText, 2000);
                    });
                }
            });
            function addCopyButtons() {
                document.querySelectorAll('pre').forEach(pre => {
                    if (!pre.querySelector('.copy-code-btn')) {
                        const btn = document.createElement('button');
                        btn.className = 'copy-code-btn absolute top-2 right-2 bg-gray-700 hover:bg-gray-600 text-white text-xs px-2 py-1 rounded';
                        btn.innerText = 'Copy';
                        pre.style.position = 'relative';
                        pre.appendChild(btn);
                    }
                });
            }
            setInterval(addCopyButtons, 1000);
            function scrollToBottom() {
                const el = document.querySelector('.q-page-container');
                if (el) el.scrollTop = el.scrollHeight;
            }
        </script>
    ''')

    # --- HEADER ---
    with ui.header().classes('bg-gray-900 text-white border-b border-gray-800 px-4 py-2 items-center justify-between'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('auto_awesome', size='sm').classes('text-blue-400')
            ui.label('Nexus AI').classes('text-lg font-bold tracking-tight')
            
        with ui.row().classes('items-center gap-3'):
            remaining = get_rate_limit_status(user_id)
            color = 'text-green-400' if remaining > 5 else 'text-yellow-400' if remaining > 0 else 'text-red-400'
            ui.label(f'⚡ {remaining}').classes(f'text-sm font-semibold {color}').tooltip('Messages remaining this minute')
            
            def logout():
                del app.storage.user['user_id']
                del app.storage.user['current_session']
                ui.navigate.to('/login')
            ui.button(icon='logout', on_click=logout).props('flat round dense color=white size=sm').tooltip('Logout')
            ui.button(icon='settings', on_click=lambda: ui.navigate.to('/settings')).props('flat round dense color=white size=sm').tooltip('Settings')

    # --- MAIN LAYOUT ---
    with ui.row().classes('w-full no-wrap h-[calc(100vh-60px)] bg-gray-900'):
        
        # --- SIDEBAR ---
        with ui.column().classes('w-72 bg-gray-950 h-full overflow-y-auto border-r border-gray-800 p-3'):
            ui.button('New Chat', icon='add', on_click=lambda: create_new_chat()).props('rounded flat color=blue-400 align=left').classes('w-full mb-4 text-left')
            
            ui.label('Recent Chats').classes('text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 px-2')
            
            chat_list_container = ui.column().classes('w-full gap-1')
            
            def refresh_sidebar():
                chat_list_container.clear()
                for sid, title in get_sessions(user_id):
                    is_active = (sid == app.storage.user['current_session'])
                    bg = 'bg-gray-800 text-white' if is_active else 'text-gray-300 hover:bg-gray-800'
                    with ui.row().classes(f'chat-item w-full items-center justify-between rounded-lg p-2 cursor-pointer {bg}'):
                        ui.label(title).classes('truncate text-sm flex-grow').on('click', lambda s=sid: load_chat(s))
                        ui.button(icon='delete', on_click=lambda s=sid: remove_chat(s)).props('flat dense round size=sm color=grey-5').classes('opacity-0 hover:opacity-100 transition-opacity')

            def create_new_chat():
                new_id = create_session(user_id, "New Chat")
                app.storage.user['current_session'] = new_id
                refresh_sidebar()
                refresh_chat()
                
            def load_chat(sid):
                app.storage.user['current_session'] = sid
                refresh_sidebar()
                refresh_chat()
                
            def remove_chat(sid):
                delete_session(sid, user_id)
                if app.storage.user['current_session'] == sid:
                    sessions = get_sessions(user_id)
                    app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")
                refresh_sidebar()
                refresh_chat()

            refresh_sidebar()

        # --- MAIN CHAT ---
        with ui.column().classes('flex-grow h-full flex flex-col relative'):
            
            chat_container = ui.column().classes('w-full max-w-3xl mx-auto gap-6 overflow-y-auto flex-grow p-4 pb-32')
            
            def render_message_content(container, content, msg_idx, total_msgs):
                if "[PLOT_GENERATED]" in content:
                    clean = content.replace("[PLOT_GENERATED]", "")
                    with container: ui.markdown(clean).classes('w-full text-gray-200')
                    if os.path.exists("output_plot.png"):
                        with container: ui.image("output_plot.png").classes('w-full rounded-lg mt-2 border border-gray-700')
                        os.remove("output_plot.png")
                elif "[IMAGE_GENERATED]" in content:
                    parts = content.split("[IMAGE_GENERATED]")
                    with container: ui.markdown(parts[0]).classes('w-full text-gray-200')
                    if len(parts) > 1:
                        img_url = parts[1].strip()
                        if img_url.startswith("https://image.pollinations.ai/"):
                            with container: ui.image(img_url).classes('w-full rounded-lg mt-2 border border-gray-700')
                elif "[SOURCES]" in content:
                    parts = content.split("[SOURCES]")
                    with container: ui.markdown(parts[0]).classes('w-full text-gray-200')
                    urls = parts[1].split("|")
                    with container: ui.label("🌐 Sources").classes('text-xs font-bold text-gray-500 mt-3 mb-1')
                    for url in urls:
                        if url.strip().startswith("http"):
                            with container: ui.link(url, url).classes('text-xs text-blue-400 hover:underline block truncate')
                else:
                    with container: ui.markdown(content).classes('w-full text-gray-200')

                if msg_idx == total_msgs - 1:
                    with container:
                        ui.button('Regenerate', icon='refresh', on_click=lambda: regenerate_response()).props('flat dense size=sm color=grey-5').classes('mt-2')

            def refresh_chat():
                chat_container.clear()
                messages = load_messages(app.storage.user['current_session'], user_id)
                if not messages:
                    with chat_container:
                        with ui.column().classes('w-full items-center justify-center mt-32 opacity-50'):
                            ui.icon('auto_awesome', size='4rem').classes('text-blue-400 mb-4')
                            ui.label('How can I help you today?').classes('text-2xl font-light text-gray-400')
                for i, msg in enumerate(messages):
                    sent = msg['role'] == 'user'
                    if sent:
                        with chat_container:
                            with ui.row().classes('w-full justify-end chat-bubble-user'):
                                with ui.card().classes('bg-blue-600 text-white rounded-2xl rounded-tr-sm max-w-[80%] px-4 py-2 shadow-lg border-none'):
                                    ui.markdown(msg['content'])
                    else:
                        with chat_container:
                            with ui.row().classes('w-full justify-start items-start gap-3 chat-bubble-ai'):
                                with ui.avatar().classes('bg-gradient-to-br from-blue-500 to-purple-600'):
                                    ui.icon('auto_awesome', color='white')
                                with ui.column().classes('flex-grow max-w-[85%] prose prose-invert prose-sm'):
                                    render_message_content(ui.column().classes('w-full'), msg['content'], i, len(messages))
                ui.run_javascript('scrollToBottom()')

            refresh_chat()

            async def regenerate_response():
                messages = load_messages(app.storage.user['current_session'], user_id)
                if not messages or messages[-1]['role'] != 'assistant':
                    return
                delete_last_message(app.storage.user['current_session'], user_id)
                user_prompt = messages[-2]['content'] if len(messages) >= 2 else ""
                refresh_chat()
                if user_prompt:
                    await run_agent(user_prompt)

            # --- FLOATING INPUT BAR ---
            with ui.row().classes('w-full max-w-3xl mx-auto px-4 pb-6 absolute bottom-0 left-0 right-0'):
                with ui.card().classes('w-full bg-gray-800 shadow-2xl rounded-2xl flex-row items-center px-3 py-2 border border-gray-700 modern-input'):
                    
                    file_input = ui.element('input').props('type=file accept=".pdf,.csv,.xlsx,.xls" style="display: none;"')
                    file_input.on('change', lambda e: ui.run_javascript('''
                        const file = document.querySelector('input[type=file]').files[0];
                        if (file) {
                            const reader = new FileReader();
                            reader.onload = (event) => {
                                const base64Data = event.target.result.split(',')[1];
                                emitEvent('file_uploaded', {name: file.name, data: base64Data});
                            };
                            reader.readAsDataURL(file);
                        }
                    '''))

                    async def process_file_event(e):
                        file_name = e.args['name']
                        file_bytes = base64.b64decode(e.args['data'])
                        await handle_upload_logic(file_name, file_bytes)

                    ui.on('file_uploaded', process_file_event)

                    ui.button(icon='attach_file', on_click=lambda: ui.run_javascript('document.querySelector("input[type=file]").click()')).props('flat round dense color=grey-5 size=sm')
                    
                    user_input = ui.input(placeholder='Ask anything...').props('borderless dense').classes('flex-grow text-white bg-transparent text-base')
                    
                    stop_btn = ui.button('Stop', icon='stop', on_click=lambda: stop_generation()).props('flat dense color=red-400 size=sm').classes('mr-1 hidden')
                    
                    async def run_agent(prompt):
                        global current_task
                        stop_btn.classes(remove='hidden')
                        send_btn.classes(add='hidden')
                        
                        with chat_container:
                            with ui.row().classes('w-full justify-start items-start gap-3') as ai_row:
                                with ui.avatar().classes('bg-gradient-to-br from-blue-500 to-purple-600 animate-pulse'):
                                    ui.icon('auto_awesome', color='white')
                                with ui.column().classes('flex-grow max-w-[85%] prose prose-invert prose-sm'):
                                    status_label = ui.label('Thinking...').classes('text-gray-400 italic text-sm not-prose')
                                    answer_container = ui.column().classes('w-full')
                        
                        ui.run_javascript('scrollToBottom()')
                        final_answer = ""
                        try:
                            current_task = asyncio.current_task()
                            async for msg_type, content in ask_agent_v2_stream(prompt, user_id):
                                if msg_type == "step":
                                    status_label.text = content
                                elif msg_type == "answer":
                                    final_answer = content
                                    status_label.visible = False
                                    answer_container.clear()
                                    render_message_content(answer_container, final_answer, 999, 999)
                                    ai_row.find('icon').classes('animate-pulse', remove=True)
                                    ui.run_javascript('scrollToBottom()')
                        except asyncio.CancelledError:
                            status_label.text = "⏹️ Generation stopped by user."
                            if not final_answer:
                                final_answer = "Generation stopped."
                            ui.notify('Generation stopped', type='warning')
                        finally:
                            current_task = None
                            stop_btn.classes(add='hidden')
                            send_btn.classes(remove='hidden')
                        
                        save_message(app.storage.user['current_session'], 'assistant', final_answer, user_id)
                        messages = load_messages(app.storage.user['current_session'], user_id)
                        if len(messages) == 2:
                            new_title = generate_chat_title(prompt)
                            import sqlite3
                            from database import DB_NAME
                            conn = sqlite3.connect(DB_NAME)
                            cursor = conn.cursor()
                            cursor.execute("UPDATE sessions SET title = ? WHERE id = ? AND user_id = ?", (new_title, app.storage.user['current_session'], user_id))
                            conn.commit()
                            conn.close()
                        refresh_sidebar()
                        refresh_chat()

                    async def send_message():
                        if not user_input.value: return
                        allowed, remaining = check_rate_limit(user_id)
                        if not allowed:
                            ui.notify('Rate limit exceeded. Please wait 60 seconds.', type='negative', position='top')
                            return
                        if remaining <= 3:
                            ui.notify(f'Warning: You only have {remaining} messages left this minute.', type='warning', position='top')
                        prompt = user_input.value
                        user_input.value = ''
                        save_message(app.storage.user['current_session'], 'user', prompt, user_id)
                        
                        with chat_container:
                            with ui.row().classes('w-full justify-end chat-bubble-user'):
                                with ui.card().classes('bg-blue-600 text-white rounded-2xl rounded-tr-sm max-w-[80%] px-4 py-2 shadow-lg border-none'):
                                    ui.markdown(prompt)
                        await run_agent(prompt)
                        ui.navigate.to('/')

                    def stop_generation():
                        global current_task
                        if current_task:
                            current_task.cancel()

                    send_btn = ui.button(icon='send', on_click=send_message).props('flat round dense color=blue-400 size=sm')
                    user_input.on('keydown.enter', send_message)

            async def handle_upload_logic(file_name, file_bytes):
                ui.notify(f"Processing {file_name}...", type='info', position='top')
                res = await ui.run.io_bound(ingest_file, file_bytes, file_name, user_id)
                with chat_container:
                    with ui.row().classes('w-full justify-center'):
                        ui.badge(f"📁 {res}", color='gray-700').classes('text-gray-300 text-xs')

@ui.page('/settings')
def settings_page():
    if not is_logged_in():
        ui.navigate.to('/login')
        return
    ui.add_head_html(GLOBAL_CSS)
    
    user_id = app.storage.user['user_id']
    user_email = get_user_email(user_id)
    user_role = get_user_role(user_id)
    
    with ui.header().classes('items-center justify-between bg-gray-900 text-white px-4 py-3 border-b border-gray-800'):
        ui.label('⚙️ Settings').classes('text-lg font-bold')
        ui.button('Back to Chat', on_click=lambda: ui.navigate.to('/')).props('flat color=white size=sm')
        
    with ui.column().classes('w-full max-w-4xl mx-auto p-6 bg-gray-900 min-h-screen'):
        
        # Profile Card
        ui.label('Profile').classes('text-xl font-bold text-white mb-4')
        with ui.card().classes('w-full bg-gray-800 border border-gray-700 p-6 mb-6 flex-row items-center justify-between rounded-xl'):
            with ui.column():
                ui.label(f"Email: {user_email}").classes('text-white text-base')
                badge_color = 'bg-blue-600' if user_role == 'admin' else 'bg-gray-600'
                ui.badge(f"Role: {user_role.upper()}", color=badge_color).classes('mt-2')
            with ui.column().classes('gap-2'):
                new_password = ui.input('New Password', password=True, password_toggle_button=True).props('outlined dense rounded dark color=blue-400').classes('w-64')
                async def update_password():
                    if len(new_password.value) < 12:
                        ui.notify('Password must be at least 12 characters', type='warning')
                        return
                    update_user_password(user_id, new_password.value)
                    ui.notify('Password updated successfully!', type='positive')
                    new_password.value = ''
                ui.button('Update Password', on_click=update_password).props('rounded color=primary size=sm')

        # Chat History Table
        ui.label('My Chat History').classes('text-xl font-bold text-white mb-4')
        sessions = get_all_sessions_for_user(user_id)
        if not sessions:
            ui.label('No chat history yet.').classes('text-gray-500 italic')
        else:
            columns = [
                {'name': 'title', 'label': 'Chat Title', 'field': 'title', 'align': 'left'},
                {'name': 'date', 'label': 'Created', 'field': 'date'},
                {'name': 'actions', 'label': '', 'field': 'id'},
            ]
            rows = [{'id': s[0], 'title': s[1], 'date': s[2]} for s in sessions]
            table = ui.table(columns=columns, rows=rows, row_key='id').classes('w-full bg-gray-800 text-white rounded-xl').props('flat bordered dark')
            table.add_slot('body-cell-actions', '''
                <q-td :props="props">
                    <q-btn icon="delete" flat dense round color="red" @click="() => $parent.$emit('delete', props.row.id)" />
                </q-td>
            ''')
            def handle_delete(e):
                delete_session(e.args, user_id)
                ui.notify('Chat deleted. Refreshing...', type='info')
                ui.navigate.to('/settings')
            table.on('delete', handle_delete)

        ui.separator().classes('bg-gray-700 my-6')

        # Admin Analytics
        if is_admin(user_id):
            ui.label('Admin Analytics').classes('text-xl font-bold text-white mb-4')
            route_data = get_analytics_by_route()
            if not route_data:
                ui.label('No analytics data yet.').classes('text-gray-500')
            else:
                with ui.row().classes('w-full gap-4 mb-6'):
                    with ui.card().classes('w-1/2 bg-gray-800 border border-gray-700 p-4 rounded-xl'):
                        ui.label('Agent Usage').classes('text-white font-bold mb-2')
                        chart_data = [{'value': v, 'name': k} for k, v in route_data.items()]
                        ui.echart({
                            'tooltip': {'trigger': 'item'},
                            'series': [{'type': 'pie', 'radius': ['40%', '70%'], 'data': chart_data, 'label': {'color': '#fff'}}]
                        }).classes('w-full h-64')
                    with ui.card().classes('w-1/2 bg-gray-800 border border-gray-700 p-4 rounded-xl'):
                        ui.label('Recent Latency').classes('text-white font-bold mb-2')
                        recent = get_analytics()[:10]
                        latencies = [round(r[2], 2) for r in reversed(recent)]
                        labels = [f"Q{i+1}" for i in range(len(latencies))]
                        ui.echart({
                            'xAxis': {'type': 'category', 'data': labels, 'axisLabel': {'color': '#aaa'}},
                            'yAxis': {'type': 'value', 'axisLabel': {'color': '#aaa'}},
                            'series': [{'type': 'bar', 'data': latencies, 'itemStyle': {'color': '#3b82f6'}}]
                        }).classes('w-full h-64')
        else:
            ui.label('🔒 Admin dashboard is only visible to administrators.').classes('text-gray-500 italic')

ui.run(title='Nexus AI', storage_secret=storage_secret, dark=True, favicon='🌌', reload=False, host='127.0.0.1', port=int(os.getenv('PORT', '8080')))