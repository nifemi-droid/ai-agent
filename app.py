from nicegui import ui, app
from multi_agent import ask_agent_v2_stream, ingest_file
from database import (ensure_dbs_exist, create_user, authenticate_user, 
                      create_session, get_sessions, load_messages, save_message, 
                      delete_session, get_analytics)
import base64

ensure_dbs_exist()

def is_logged_in():
    return 'user_id' in app.storage.user

@ui.page('/login')
def login_page():
    if is_logged_in():
        ui.navigate.to('/')
        return

    with ui.column().classes('w-full h-screen items-center justify-center bg-gray-50'):
        with ui.card().classes('w-96 p-8 shadow-xl rounded-xl'):
            ui.label('🌌 Nexus AI').classes('text-h4 text-center mb-6 font-bold text-gray-800')
            
            email = ui.input('Email').props('outlined rounded').classes('w-full mb-4')
            password = ui.input('Password', password=True, password_toggle_button=True).props('outlined rounded').classes('w-full mb-6')
            
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

            with ui.row().classes('w-full justify-between gap-2'):
                ui.button('Login', on_click=handle_login).props('rounded color=primary').classes('w-1/2')
                ui.button('Register', on_click=handle_register).props('flat color=primary').classes('w-1/2')

@ui.page('/')
def main_page():
    if not is_logged_in():
        ui.navigate.to('/login')
        return

    user_id = app.storage.user['user_id']

    if 'current_session' not in app.storage.user:
        sessions = get_sessions(user_id)
        app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")

    # Set a custom dark theme palette
    dark = ui.dark_mode()
    dark.enable()

    with ui.header().classes('bg-gray-900 text-white border-b border-gray-800 px-4 py-2 flex items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('auto_awesome', size='sm').classes('text-blue-400')
            ui.label('Nexus AI').classes('text-h6 font-semibold')
            
        with ui.row().classes('items-center gap-4'):
            ui.switch('Dark Mode', value=True, on_change=lambda e: dark.enable() if e.value else dark.disable()).props('color=blue-400')
            
            def logout():
                del app.storage.user['user_id']
                del app.storage.user['current_session']
                ui.navigate.to('/login')
            ui.button(icon='logout', on_click=logout).props('flat round dense color=white').tooltip('Logout')
            ui.button(icon='settings', on_click=lambda: ui.navigate.to('/settings')).props('flat round dense color=white')

    with ui.row().classes('w-full no-wrap h-[calc(100vh-60px)] bg-gray-900'):
        
        # --- SIDEBAR ---
        with ui.column().classes('w-64 bg-gray-950 h-full overflow-y-auto border-r border-gray-800 p-3'):
            ui.button('New Chat', icon='add', on_click=lambda: create_new_chat()).props('rounded flat color=blue-400 align=left').classes('w-full mb-4')
            
            ui.label('Recent Chats').classes('text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 px-2')
            
            chat_list_container = ui.column().classes('w-full gap-1')
            
            def refresh_sidebar():
                chat_list_container.clear()
                for sid, title in get_sessions(user_id):
                    is_active = (sid == app.storage.user['current_session'])
                    bg_class = 'bg-gray-800 text-white' if is_active else 'text-gray-300 hover:bg-gray-800'
                    
                    with ui.row().classes(f'w-full items-center justify-between rounded p-2 cursor-pointer transition-colors {bg_class}'):
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
                delete_session(sid)
                if app.storage.user['current_session'] == sid:
                    sessions = get_sessions(user_id)
                    app.storage.user['current_session'] = sessions[0][0] if sessions else create_session(user_id, "New Chat")
                refresh_sidebar()
                refresh_chat()

            refresh_sidebar()

        # --- MAIN CHAT AREA ---
        with ui.column().classes('flex-grow h-full flex flex-col relative'):
            
            # Chat Container
            chat_container = ui.column().classes('w-full max-w-3xl mx-auto gap-6 overflow-y-auto flex-grow p-4 pb-24')
            
            def refresh_chat():
                chat_container.clear()
                for msg in load_messages(app.storage.user['current_session']):
                    sent = msg['role'] == 'user'
                    
                    if sent:
                        # User Message
                        with chat_container:
                            with ui.row().classes('w-full justify-end'):
                                with ui.card().classes('bg-gray-800 text-white rounded-2xl max-w-[80%] px-4 py-2 shadow-none'):
                                    ui.markdown(msg['content'])
                    else:
                        # AI Message (Native Markdown with Prose styling)
                        with chat_container:
                            with ui.row().classes('w-full justify-start items-start gap-4'):
                                ui.icon('auto_awesome', size='md').classes('text-blue-400 mt-1')
                                with ui.column().classes('flex-grow max-w-[85%] prose prose-invert prose-sm'):
                                    ui.markdown(msg['content']).classes('w-full text-gray-200')

            refresh_chat()

            # --- FLOATING INPUT BAR ---
            with ui.row().classes('w-full max-w-3xl mx-auto px-4 pb-6 absolute bottom-0 left-0 right-0'):
                with ui.card().classes('w-full bg-gray-800 shadow-xl rounded-2xl flex-row items-center px-2 py-1 border border-gray-700'):
                    
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

                    ui.button(icon='attach_file', on_click=lambda: ui.run_javascript('document.querySelector("input[type=file]").click()')).props('flat round dense color=grey-5').classes('ml-2')
                    
                    user_input = ui.input(placeholder='Ask anything...').props('borderless dense').classes('flex-grow text-white bg-transparent')
                    
                    async def send_message():
                        if not user_input.value: return
                        prompt = user_input.value
                        user_input.value = ''
                        
                        save_message(app.storage.user['current_session'], 'user', prompt)
                        
                        # Display User Message Immediately
                        with chat_container:
                            with ui.row().classes('w-full justify-end'):
                                with ui.card().classes('bg-gray-800 text-white rounded-2xl max-w-[80%] px-4 py-2 shadow-none'):
                                    ui.markdown(prompt)
                        
                        # Display AI Thinking Bubble
                        with chat_container:
                            with ui.row().classes('w-full justify-start items-start gap-4') as ai_row:
                                ui.icon('auto_awesome', size='md').classes('text-blue-400 mt-1 animate-pulse')
                                with ui.column().classes('flex-grow max-w-[85%] prose prose-invert prose-sm'):
                                    status_label = ui.label('Thinking...').classes('text-gray-400 italic text-sm not-prose')
                                    answer_label = ui.markdown('').classes('w-full text-gray-200')
                        
                        final_answer = ""
                        async for msg_type, content in ask_agent_v2_stream(prompt):
                            if msg_type == "step":
                                status_label.text = content
                            elif msg_type == "answer":
                                final_answer = content
                                status_label.visible = False
                                # Native NiceGUI Markdown handles tables perfectly
                                answer_label.content = final_answer
                                ai_row.find('icon').classes('animate-pulse', remove=True)
                        
                        save_message(app.storage.user['current_session'], 'assistant', final_answer)
                        refresh_sidebar()

                    ui.button(icon='send', on_click=send_message).props('flat round dense color=blue-400').classes('mr-2')
                    user_input.on('keydown.enter', send_message)

            async def handle_upload_logic(file_name, file_bytes):
                ui.notify(f"Processing {file_name}...", type='info', position='top')
                res = await ui.run.io_bound(ingest_file, file_bytes, file_name)
                
                with chat_container:
                    with ui.row().classes('w-full justify-center'):
                        ui.badge(f"📁 System: {res}", color='gray-700').classes('text-gray-300 text-xs')

@ui.page('/settings')
def settings_page():
    if not is_logged_in():
        ui.navigate.to('/login')
        return

    with ui.header().classes('items-center justify-between bg-gray-900 text-white p-4 border-b border-gray-800'):
        ui.label('⚙️ Settings & Admin').classes('text-h5 font-bold')
        ui.button('Back to Chat', on_click=lambda: ui.navigate.to('/')).props('flat color=white')
        
    with ui.column().classes('w-full max-w-4xl mx-auto p-8 bg-gray-900 min-h-screen'):
        ui.label('Analytics Dashboard').classes('text-h4 mb-4 text-white')
        
        data = get_analytics()
        if not data:
            ui.label('No analytics data yet.').classes('text-gray-500')
        else:
            with ui.row().classes('gap-4 mb-6 w-full'):
                with ui.card().classes('w-1/3 bg-gray-800 text-white border border-gray-700'):
                    ui.label('Total Queries').classes('text-gray-400')
                    ui.label(str(len(data))).classes('text-h3 text-white')
                    
                with ui.card().classes('w-1/3 bg-gray-800 text-white border border-gray-700'):
                    ui.label('Avg Latency').classes('text-gray-400')
                    avg_lat = sum([r[2] for r in data]) / len(data)
                    ui.label(f"{avg_lat:.2f}s").classes('text-h3 text-white')
                    
                with ui.card().classes('w-1/3 bg-gray-800 text-white border border-gray-700'):
                    ui.label('Tools Used').classes('text-gray-400')
                    ui.label(str(len(set([r[1] for r in data])))).classes('text-h3 text-white')
            
            ui.label('Recent Activity').classes('text-h5 mb-2 text-white')
            columns = [
                {'name': 'query', 'label': 'Query', 'field': 'query', 'align': 'left'},
                {'name': 'tool', 'label': 'Tool', 'field': 'tool'},
                {'name': 'latency', 'label': 'Latency', 'field': 'latency'},
                {'name': 'time', 'label': 'Time', 'field': 'time'},
            ]
            rows = [{'query': r[0], 'tool': r[1], 'latency': f"{r[2]:.2f}s", 'time': r[3]} for r in data]
            ui.table(columns=columns, rows=rows).classes('w-full bg-gray-800 text-white')

ui.run(title='Nexus AI', storage_secret='super_secret_nexus_key_123', dark=True, favicon='🌌', reload=True)