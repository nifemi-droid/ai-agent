from nicegui import ui, app
from multi_agent import ask_agent_v2, ingest_file
from database import (ensure_dbs_exist, create_user, authenticate_user, 
                      create_session, get_sessions, load_messages, save_message, 
                      delete_session, get_analytics)
import base64

ensure_dbs_exist()

# --- AUTHENTICATION GUARD ---
def is_logged_in():
    return 'user_id' in app.storage.user

@ui.page('/login')
def login_page():
    if is_logged_in():
        ui.navigate.to('/')
        return

    with ui.column().classes('w-full h-screen items-center justify-center bg-gray-100'):
        with ui.card().classes('w-96 p-8'):
            ui.label('🌌 Nexus AI').classes('text-h4 text-center mb-6 font-bold text-blue-900')
            
            email = ui.input('Email').classes('w-full mb-4')
            password = ui.input('Password', password=True, password_toggle_button=True).classes('w-full mb-4')
            
            async def handle_login():
                user_id = authenticate_user(email.value, password.value)
                if user_id:
                    app.storage.user['user_id'] = user_id
                    ui.navigate.to('/')
                else:
                    ui.notify('Invalid email or password', type='negative')

            async def handle_register():
                user_id = create_user(email.value, password.value)
                if user_id:
                    app.storage.user['user_id'] = user_id
                    ui.notify('Account created!', type='positive')
                    ui.navigate.to('/')
                else:
                    ui.notify('Email already exists', type='warning')

            with ui.row().classes('w-full justify-between'):
                ui.button('Login', on_click=handle_login).classes('bg-blue-600 text-white')
                ui.button('Register', on_click=handle_register).props('flat color=primary')

@ui.page('/')
def main_page():
    if not is_logged_in():
        ui.navigate.to('/login')
        return

    user_id = app.storage.user['user_id']

    if 'current_session' not in app.storage.user:
        sessions = get_sessions(user_id)
        app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")

    dark = ui.dark_mode()

    with ui.header().classes('items-center justify-between bg-blue-900 text-white p-4'):
        ui.label('🌌 Nexus AI').classes('text-h5 font-bold')
        with ui.row().classes('items-center gap-4'):
            ui.switch('Dark Mode', on_change=lambda e: dark.enable() if e.value else dark.disable())
            
            def logout():
                del app.storage.user['user_id']
                del app.storage.user['current_session']
                ui.navigate.to('/login')
            ui.button(icon='logout', on_click=logout).props('flat round color=white').tooltip('Logout')
            
            ui.button(icon='settings', on_click=lambda: ui.navigate.to('/settings')).props('flat round color=white')

    with ui.row().classes('w-full no-wrap h-[calc(100vh-64px)]'):
        
        # --- SIDEBAR ---
        with ui.column().classes('w-64 bg-gray-100 dark:bg-gray-800 p-4 h-full overflow-y-auto border-r'):
            ui.button('➕ New Chat', on_click=lambda: create_new_chat()).classes('w-full bg-blue-600 text-white mb-4')
            ui.label('Recent Chats').classes('text-sm font-bold text-gray-500 mt-4 mb-2')
            
            chat_list_container = ui.column().classes('w-full gap-1')
            
            def refresh_sidebar():
                chat_list_container.clear()
                for sid, title in get_sessions(user_id):
                    is_active = (sid == app.storage.user['current_session'])
                    with ui.row().classes('w-full items-center justify-between hover:bg-gray-200 dark:hover:bg-gray-700 rounded p-1 cursor-pointer'):
                        ui.label(title).classes('truncate text-sm flex-grow' + (' font-bold text-blue-600' if is_active else '')).on('click', lambda s=sid: load_chat(s))
                        ui.button(icon='delete', on_click=lambda s=sid: remove_chat(s)).props('flat dense color=grey size=sm')

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
                delete_session(sid)
                if app.storage.user['current_session'] == sid:
                    sessions = get_sessions(user_id)
                    app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")
                refresh_sidebar()
                refresh_chat()

            refresh_sidebar()

        # --- MAIN CHAT AREA ---
        with ui.column().classes('flex-grow p-4 h-full flex flex-col'):
            chat_container = ui.column().classes('w-full max-w-4xl mx-auto gap-4 overflow-y-auto flex-grow p-4')
            
            def refresh_chat():
                chat_container.clear()
                for msg in load_messages(app.storage.user['current_session']):
                    sent = msg['role'] == 'user'
                    with chat_container:
                        ui.chat_message(msg['content'], name='You' if sent else 'Nexus', sent=sent).classes('w-full')

            refresh_chat()

            # --- CLEAN INPUT AREA ---
            with ui.row().classes('w-full max-w-4xl mx-auto items-center mt-4 gap-4 px-4 pb-4'):
                # Hidden File Input
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

                # Paperclip Button
                ui.button(icon='attach_file', on_click=lambda: ui.run_javascript('document.querySelector("input[type=file]").click()')).props('round flat color=gray').classes('shadow-none text-gray-500')
                
                # Text Input
                user_input = ui.input(placeholder='Ask anything...').props('rounded outlined dense').classes('flex-grow text-lg shadow-sm')
                
                # Send Button
                async def send_message():
                    if not user_input.value: return
                    prompt = user_input.value
                    user_input.value = ''
                    
                    save_message(app.storage.user['current_session'], 'user', prompt)
                    with chat_container:
                        ui.chat_message(prompt, name='You', sent=True).classes('w-full')
                    
                    with chat_container:
                        with ui.chat_message(name='Nexus', sent=False).classes('w-full') as thinking_msg:
                            with ui.row().classes('items-center'):
                                ui.spinner(size='sm', color='primary')
                                ui.label('Routing agents...').classes('text-gray-500 italic')
                    
                    steps, answer = await ui.run.io_bound(ask_agent_v2, prompt)
                    
                    chat_container.remove(thinking_msg)
                    
                    save_message(app.storage.user['current_session'], 'assistant', answer)
                    with chat_container:
                        with ui.chat_message(name='Nexus', sent=False).classes('w-full'):
                            ui.markdown(answer)
                    
                    refresh_sidebar()

                ui.button(icon='send', on_click=send_message).props('round color=primary').classes('h-12 w-12 shadow-md')
                user_input.on('keydown.enter', send_message)

            async def handle_upload_logic(file_name, file_bytes):
                with chat_container:
                    ui.notify(f"Processing {file_name}...", type='info', position='top')
                
                res = await ui.run.io_bound(ingest_file, file_bytes, file_name)
                
                with chat_container:
                    ui.chat_message(f"📁 *System:* {res}", name='System', sent=False).classes('w-full text-gray-500 text-sm')

@ui.page('/settings')
def settings_page():
    if not is_logged_in():
        ui.navigate.to('/login')
        return

    with ui.header().classes('items-center justify-between bg-blue-900 text-white p-4'):
        ui.label('⚙️ Settings & Admin').classes('text-h5 font-bold')
        ui.button('Back to Chat', on_click=lambda: ui.navigate.to('/')).props('flat color=white')
        
    with ui.column().classes('w-full max-w-4xl mx-auto p-8'):
        ui.label('Analytics Dashboard').classes('text-h4 mb-4')
        
        data = get_analytics()
        if not data:
            ui.label('No analytics data yet.').classes('text-gray-500')
        else:
            with ui.row().classes('gap-4 mb-6 w-full'):
                with ui.card().classes('w-1/3'):
                    ui.label('Total Queries').classes('text-gray-500')
                    ui.label(str(len(data))).classes('text-h3')
                    
                with ui.card().classes('w-1/3'):
                    ui.label('Avg Latency').classes('text-gray-500')
                    avg_lat = sum([r[2] for r in data]) / len(data)
                    ui.label(f"{avg_lat:.2f}s").classes('text-h3')
                    
                with ui.card().classes('w-1/3'):
                    ui.label('Tools Used').classes('text-gray-500')
                    ui.label(str(len(set([r[1] for r in data])))).classes('text-h3')
            
            ui.label('Recent Activity').classes('text-h5 mb-2')
            columns = [
                {'name': 'query', 'label': 'Query', 'field': 'query', 'align': 'left'},
                {'name': 'tool', 'label': 'Tool', 'field': 'tool'},
                {'name': 'latency', 'label': 'Latency', 'field': 'latency'},
                {'name': 'time', 'label': 'Time', 'field': 'time'},
            ]
            rows = [{'query': r[0], 'tool': r[1], 'latency': f"{r[2]:.2f}s", 'time': r[3]} for r in data]
            ui.table(columns=columns, rows=rows).classes('w-full')

ui.run(title='Nexus AI', storage_secret='super_secret_nexus_key_123', dark=True, favicon='🌌', reload=True)