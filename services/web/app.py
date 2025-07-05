"""
RAG Chatbot Web Interface
Flask-based chat UI with content groups
"""

import os
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
API_URL = os.environ.get('API_URL', 'http://localhost:5001')


@app.route('/')
def index():
    """Main chat interface"""
    # Get or create session ID
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
    
    return render_template('index.html')


@app.route('/groups')
def groups():
    """Content groups management page"""
    return render_template('groups.html')


@app.route('/api/proxy/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    """Proxy API requests to backend"""
    url = f"{API_URL}/api/{path}"
    
    if request.method == 'GET':
        response = requests.get(url, params=request.args)
    else:
        response = requests.request(
            method=request.method,
            url=url,
            json=request.get_json(),
            headers={key: value for key, value in request.headers if key != 'Host'}
        )
    
    return response.json(), response.status_code


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('connected', {'data': 'Connected to chat server'})


@socketio.on('join_session')
def handle_join_session(data):
    """Join a chat session room"""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('joined_session', {'session_id': session_id}, room=session_id)


@socketio.on('leave_session')
def handle_leave_session(data):
    """Leave a chat session room"""
    session_id = data.get('session_id')
    if session_id:
        leave_room(session_id)
        emit('left_session', {'session_id': session_id}, room=session_id)


@socketio.on('send_message')
def handle_message(data):
    """Handle incoming chat message via WebSocket"""
    session_id = data.get('session_id')
    message = data.get('message')
    group_id = data.get('group_id', 'general')
    
    # Send to API
    response = requests.post(
        f"{API_URL}/api/chat",
        json={
            'message': message,
            'session_id': session_id,
            'group_id': group_id
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        # Emit response to all clients in the session room
        emit('message_response', result, room=session_id)
    else:
        emit('error', {'error': 'Failed to process message'}, room=session_id)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=os.environ.get('DEBUG', 'False').lower() == 'true')