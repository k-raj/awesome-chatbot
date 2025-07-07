"""
Flask Web Server for RAG Application UI
Provides a user-friendly interface for managing groups, files, and chat sessions
"""

import os
import json
import time
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Configuration
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:5000')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'temp_uploads')
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def format_datetime(dt_string):
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return dt_string

# Template filters
app.jinja_env.filters['file_size'] = format_file_size
app.jinja_env.filters['datetime'] = format_datetime

@app.route('/')
def index():
    """Main page showing all groups"""
    try:
        # Get all groups from API
        response = requests.get(f"{API_BASE_URL}/api/groups", timeout=10)
        
        if response.status_code == 200:
            groups_data = response.json()
            groups = groups_data.get('groups', [])
            
            # Sort groups - General Chat first, then by creation date
            groups.sort(key=lambda x: (x.get('name') != 'General Chat', x.get('created_at', '')))
            
            return render_template('index.html', groups=groups)
        else:
            flash('Error loading groups. Please try again.', 'error')
            return render_template('index.html', groups=[])
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to API: {e}")
        flash('Unable to connect to the server. Please try again later.', 'error')
        return render_template('index.html', groups=[])

@app.route('/create_group', methods=['POST'])
def create_group():
    """Create a new content group"""
    try:
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        
        if not name:
            flash('Group name is required.', 'error')
            return redirect(url_for('index'))
        
        # Create group via API
        response = requests.post(
            f"{API_BASE_URL}/api/groups",
            json={'name': name, 'description': description},
            timeout=10
        )
        
        if response.status_code == 201:
            group_data = response.json()
            flash(f'Group "{name}" created successfully!', 'success')
            return redirect(url_for('group_detail', group_id=group_data['_id']))
        elif response.status_code == 409:
            flash('A group with this name already exists.', 'error')
        else:
            flash('Error creating group. Please try again.', 'error')
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating group: {e}")
        flash('Unable to connect to the server. Please try again later.', 'error')
    
    return redirect(url_for('index'))

@app.route('/group/<group_id>')
def group_detail(group_id):
    """Group detail page with file management and chat interface"""
    try:
        # Get group information (for non-general groups)
        group_info = {'_id': group_id, 'name': 'General Chat', 'description': 'General conversation without document context'}
        
        if group_id != 'general':
            groups_response = requests.get(f"{API_BASE_URL}/api/groups", timeout=10)
            if groups_response.status_code == 200:
                groups = groups_response.json().get('groups', [])
                group_info = next((g for g in groups if g['_id'] == group_id), None)
                if not group_info:
                    flash('Group not found.', 'error')
                    return redirect(url_for('index'))
        
        # Get files for this group
        files_response = requests.get(f"{API_BASE_URL}/api/groups/{group_id}/files", timeout=10)
        files = []
        if files_response.status_code == 200:
            files = files_response.json().get('files', [])
        
        # Get sessions for this group
        sessions_response = requests.get(f"{API_BASE_URL}/api/groups/{group_id}/sessions", timeout=10)
        sessions = []
        if sessions_response.status_code == 200:
            sessions = sessions_response.json().get('sessions', [])
        
        return render_template('group_detail.html', 
                             group=group_info, 
                             files=files, 
                             sessions=sessions,
                             is_general=group_id == 'general')
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading group detail: {e}")
        flash('Unable to connect to the server. Please try again later.', 'error')
        return redirect(url_for('index'))

@app.route('/group/<group_id>/upload', methods=['POST'])
def upload_file(group_id):
    """Upload a file to a group"""
    try:
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('group_detail', group_id=group_id))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('group_detail', group_id=group_id))
        
        if not allowed_file(file.filename):
            flash(f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(url_for('group_detail', group_id=group_id))
        
        # Upload file via API
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(
            f"{API_BASE_URL}/api/groups/{group_id}/files",
            files=files,
            timeout=60  # Longer timeout for file uploads
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                if result.get('status') == 'skipped':
                    flash(f'File "{file.filename}" uploaded but processing was skipped (General Chat).', 'info')
                else:
                    flash(f'File "{file.filename}" uploaded successfully and is being processed.', 'success')
                    # Store file_id in session for progress tracking
                    session[f'uploading_{result["file_id"]}'] = True
            else:
                flash('File upload failed.', 'error')
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            error_msg = error_data.get('error', 'Unknown error occurred')
            flash(f'Upload failed: {error_msg}', 'error')
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error uploading file: {e}")
        flash('Unable to connect to the server. Please try again later.', 'error')
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        flash('An unexpected error occurred during upload.', 'error')
    
    return redirect(url_for('group_detail', group_id=group_id))

@app.route('/api/file/<file_id>/status')
def get_file_status(file_id):
    """Get file processing status (AJAX endpoint)"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/files/{file_id}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': 'File not found'}, 404
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting file status: {e}")
        return {'error': 'Unable to connect to server'}, 500

@app.route('/chat/<group_id>')
@app.route('/chat/<group_id>/<session_id>')
def chat_interface(group_id, session_id=None):
    """Chat interface for a specific group and session"""
    try:
        # Get group information
        group_info = {'_id': group_id, 'name': 'General Chat', 'description': 'General conversation without document context'}
        
        if group_id != 'general':
            groups_response = requests.get(f"{API_BASE_URL}/api/groups", timeout=10)
            if groups_response.status_code == 200:
                groups = groups_response.json().get('groups', [])
                group_info = next((g for g in groups if g['_id'] == group_id), None)
                if not group_info:
                    flash('Group not found.', 'error')
                    return redirect(url_for('index'))
        
        # Get session details if session_id provided
        session_data = None
        messages = []
        
        if session_id:
            session_response = requests.get(f"{API_BASE_URL}/api/sessions/{session_id}", timeout=10)
            if session_response.status_code == 200:
                session_data = session_response.json()
                messages = session_data.get('messages', [])
        
        # Get all sessions for sidebar
        sessions_response = requests.get(f"{API_BASE_URL}/api/groups/{group_id}/sessions", timeout=10)
        all_sessions = []
        if sessions_response.status_code == 200:
            all_sessions = sessions_response.json().get('sessions', [])
        
        return render_template('chat.html',
                             group=group_info,
                             session_data=session_data,
                             messages=messages,
                             all_sessions=all_sessions,
                             current_session_id=session_id)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading chat interface: {e}")
        flash('Unable to connect to the server. Please try again later.', 'error')
        return redirect(url_for('group_detail', group_id=group_id))

@app.route('/api/chat', methods=['POST'])
def send_message():
    """Send a chat message (AJAX endpoint)"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return {'error': 'Message is required'}, 400
        
        # Send message to API
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json=data,
            timeout=60  # Longer timeout for chat responses
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            return {'error': error_data.get('error', 'Chat request failed')}, response.status_code
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message: {e}")
        return {'error': 'Unable to connect to server'}, 500
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        return {'error': 'An unexpected error occurred'}, 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check API server health
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        api_healthy = response.status_code == 200
        
        return {
            'status': 'healthy' if api_healthy else 'unhealthy',
            'webserver': 'running',
            'api_server': 'connected' if api_healthy else 'disconnected',
            'timestamp': datetime.utcnow().isoformat()
        }
    except:
        return {
            'status': 'unhealthy',
            'webserver': 'running',
            'api_server': 'disconnected',
            'timestamp': datetime.utcnow().isoformat()
        }

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

if __name__ == '__main__':
    port = int(os.environ.get('WEB_PORT', 8080))
    debug = os.environ.get('WEB_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting RAG Web UI on port {port}")
    logger.info(f"API Base URL: {API_BASE_URL}")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)