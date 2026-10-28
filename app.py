from nicegui import ui, app
from multi_agent import ask_agent_v2, ingest_file
from database import ensure_dbs_exist, create_session, get_sessions, load_messages, save_message

ensure_dbs_exist()

# Simple session state for the current chat
if 'current_session' not in app.storage.user:
    sessions = get_sessions()
    app.storage.user['current_session'] = sessions[0][0] if sessions else create_session("New Chat")

@ui.page('/')
def main_page():
    # Dark mode toggle
    dark = ui.dark_mode()
    with ui.header().classes('items-center justify-between'):
        ui.label('🌌 Nexus AI').classes('text-h6')
        with ui.row():
            ui.switch('Dark Mode', on_change=lambda e: dark.enable() if e.value else dark.disable())
            ui.button(icon='settings', on_click=lambda: ui.navigate.to('/settings')).props('flat round')

    # Chat history display
    with ui.column().classes('w-full max-w-4xl mx-auto p-4'):
        chat_container = ui.column().classes('w-full gap-2')
        
        def refresh_chat():
            chat_container.clear()
            for msg in load_messages(app.storage.user['current_session']):
                sent = msg['role'] == 'user'
                ui.chat_message(msg['content'], name='You' if sent else 'Nexus', sent=sent).classes('w-full')

        refresh_chat()

        # Input area
        with ui.row().classes('w-full items-center'):
            user_input = ui.input(placeholder='Ask anything...').props('rounded outlined').classes('flex-grow')
            async def send():
                if not user_input.value: return
                prompt = user_input.value
                user_input.value = ''
                save_message(app.storage.user['current_session'], 'user', prompt)
                refresh_chat()
                
                with ui.chat_message(name='Nexus', sent=False).classes('w-full'):
                    with ui.row().classes('items-center'):
                        ui.spinner(size='sm')
                        ui.label('Thinking...')
                    
                    steps, answer = await ui.run.io_bound(ask_agent_v2, prompt)
                    # Clear the thinking bubble and show the answer
                    # (In a real app, you'd update the bubble in place)
                    refresh_chat() 
                    # For simplicity, we just refresh. A more advanced version would stream.
                
            ui.button('Send', on_click=send).props('rounded')

@ui.page('/settings')
def settings_page():
    ui.label('Settings & Admin').classes('text-h4')
    ui.button('Back to Chat', on_click=lambda: ui.navigate.to('/'))
    # Admin dashboard and other settings go here
    # Example: ui.table with analytics data

ui.run(title='Nexus AI', storage_secret='a-very-secret-key-change-me')