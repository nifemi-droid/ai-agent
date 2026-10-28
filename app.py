from nicegui import ui, app
from multi_agent import ask_agent_v2, ingest_file
from database import ensure_dbs_exist, create_session, get_sessions, load_messages, save_message, delete_session, get_analytics
import os

ensure_dbs_exist()

@ui.page('/')
def main_page():
    # 1. Initialize session for THIS specific user inside the page
    if 'current_session' not in app.storage.user:
        sessions = get_sessions()
        app.storage.user['current_session'] = sessions[0][0] if sessions else create_session("New Chat")

    # 2. Set up the Dark Mode controller
    dark = ui.dark_mode()

    # 3. Build the Header
    with ui.header().classes('items-center justify-between bg-blue-900 text-white p-4'):
        ui.label('🌌 Nexus AI').classes('text-h5 font-bold')
        with ui.row().classes('items-center gap-4'):
            # Dark mode switch
            ui.switch('Dark Mode', on_change=lambda e: dark.enable() if e.value else dark.disable())
            # Link to settings page
            ui.button(icon='settings', on_click=lambda: ui.navigate.to('/settings')).props('flat round color=white')

    # 4. Build the Main Layout
    with ui.row().classes('w-full no-wrap h-[calc(100vh-64px)]'):
        
        # --- SIDEBAR ---
        with ui.column().classes('w-64 bg-gray-100 dark:bg-gray-800 p-4 h-full overflow-y-auto border-r'):
            ui.button('➕ New Chat', on_click=lambda: create_new_chat()).classes('w-full bg-blue-600 text-white mb-4')
            
            ui.label('Recent Chats').classes('text-sm font-bold text-gray-500 mt-4 mb-2')
            
            # Container for the chat list
            chat_list_container = ui.column().classes('w-full gap-1')
            
            def refresh_sidebar():
                chat_list_container.clear()
                for sid, title in get_sessions():
                    is_active = (sid == app.storage.user['current_session'])
                    with ui.row().classes('w-full items-center justify-between hover:bg-gray-200 dark:hover:bg-gray-700 rounded p-1 cursor-pointer'):
                        # Clicking the title loads the chat
                        ui.label(title).classes('truncate text-sm flex-grow' + (' font-bold text-blue-600' if is_active else '')).on('click', lambda s=sid: load_chat(s))
                        # Delete button
                        ui.button(icon='delete', on_click=lambda s=sid: remove_chat(s)).props('flat dense color=grey size=sm')

            def create_new_chat():
                new_id = create_session("New Chat")
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
                    sessions = get_sessions()
                    app.storage.user['current_session'] = sessions[0][0] if sessions else create_session("New Chat")
                refresh_sidebar()
                refresh_chat()

            refresh_sidebar()

        # --- MAIN CHAT AREA ---
        with ui.column().classes('flex-grow p-4 h-full flex flex-col'):
            
            # Chat Message History
            chat_container = ui.column().classes('w-full max-w-4xl mx-auto gap-4 overflow-y-auto flex-grow p-4')
            
            def refresh_chat():
                chat_container.clear()
                for msg in load_messages(app.storage.user['current_session']):
                    sent = msg['role'] == 'user'
                    with chat_container:
                        ui.chat_message(msg['content'], name='You' if sent else 'Nexus', sent=sent).classes('w-full')

            refresh_chat()

            # Input Area (Fixed at the bottom)
            with ui.row().classes('w-full max-w-4xl mx-auto items-center mt-4 gap-2'):
                # File Uploader
                uploader = ui.upload(on_upload=lambda e: handle_upload(e), auto_upload=True).props('flat dense').classes('w-12 h-12')
                uploader.tooltip('Upload PDF, CSV, or Excel')
                
                # Text Input
                user_input = ui.input(placeholder='Ask anything...').props('rounded outlined dense').classes('flex-grow text-lg')
                
                # Send Button
                async def send_message():
                    if not user_input.value: return
                    prompt = user_input.value
                    user_input.value = ''
                    
                    # Save & display user message
                    save_message(app.storage.user['current_session'], 'user', prompt)
                    with chat_container:
                        ui.chat_message(prompt, name='You', sent=True).classes('w-full')
                    
                    # Create a temporary "Thinking" bubble
                    with chat_container:
                        with ui.chat_message(name='Nexus', sent=False).classes('w-full') as thinking_msg:
                            with ui.row().classes('items-center'):
                                ui.spinner(size='sm', color='primary')
                                ui.label('Routing agents...').classes('text-gray-500 italic')
                    
                    # Call the AI
                    steps, answer = await ui.run.io_bound(ask_agent_v2, prompt)
                    
                    # Remove the thinking bubble
                    chat_container.remove(thinking_msg)
                    
                    # Save & display AI response
                    save_message(app.storage.user['current_session'], 'assistant', answer)
                    with chat_container:
                        with ui.chat_message(name='Nexus', sent=False).classes('w-full'):
                            ui.markdown(answer)
                    
                    refresh_sidebar()

                ui.button(icon='send', on_click=send_message).props('round color=primary').classes('h-12 w-12')

                # Allow pressing Enter to send
                user_input.on('keydown.enter', send_message)

            # Handle file upload
            async def handle_upload(e):
                file_name = e.name
                file_bytes = e.content.read()
                
                with chat_container:
                    ui.notify(f"Processing {file_name}...", type='info', position='top')
                
                res = await ui.run.io_bound(ingest_file, file_bytes, file_name)
                
                with chat_container:
                    ui.chat_message(f"📁 *System:* {res}", name='System', sent=False).classes('w-full text-gray-500 text-sm')

@ui.page('/settings')
def settings_page():
    # Simple Admin/Settings Page
    with ui.header().classes('items-center justify-between bg-blue-900 text-white p-4'):
        ui.label('⚙️ Settings & Admin').classes('text-h5 font-bold')
        ui.button('Back to Chat', on_click=lambda: ui.navigate.to('/')).props('flat color=white')
        
    with ui.column().classes('w-full max-w-4xl mx-auto p-8'):
        ui.label('Analytics Dashboard').classes('text-h4 mb-4')
        
        data = get_analytics()
        if not data:
            ui.label('No analytics data yet.').classes('text-gray-500')
        else:
            # Summary Metrics
            with ui.row().classes('gap-4 mb-6'):
                ui.card().classes('w-1/3'):
                    ui.label('Total Queries').classes('text-gray-500')
                    ui.label(str(len(data))).classes('text-h3')
                    
                ui.card().classes('w-1/3'):
                    ui.label('Avg Latency').classes('text-gray-500')
                    avg_lat = sum([r[2] for r in data]) / len(data)
                    ui.label(f"{avg_lat:.2f}s").classes('text-h3')
                    
                ui.card().classes('w-1/3'):
                    ui.label('Tools Used').classes('text-gray-500')
                    ui.label(str(len(set([r[1] for r in data])))).classes('text-h3')
            
            # Table of recent activity
            ui.label('Recent Activity').classes('text-h5 mb-2')
            columns = [
                {'name': 'query', 'label': 'Query', 'field': 'query', 'align': 'left'},
                {'name': 'tool', 'label': 'Tool', 'field': 'tool'},
                {'name': 'latency', 'label': 'Latency', 'field': 'latency'},
                {'name': 'time', 'label': 'Time', 'field': 'time'},
            ]
            rows = [{'query': r[0], 'tool': r[1], 'latency': f"{r[2]:.2f}s", 'time': r[3]} for r in data]
            ui.table(columns=columns, rows=rows).classes('w-full')

# Start the app with a storage secret to handle cookies
ui.run(title='Nexus AI', storage_secret='super_secret_nexus_key_123', dark=True, favicon='🌌')