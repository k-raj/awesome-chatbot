"""
Streamlined RAG Chatbot API Server with Streaming Support
Handles REST API requests with content groups, file uploads, and streaming chat functionality
Works seamlessly with inference and embedding servers
"""

import os
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Generator
from pathlib import Path

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import redis
from redis.exceptions import ConnectionError
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import hashlib

app = Flask(__name__)
CORS(app)

# Configuration
REDIS_URL = f"redis://:{os.environ.get('REDIS_PASSWORD', 'password')}@{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}/{os.environ.get('REDIS_DB', '0')}"
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/app_data/file_uploads')

# MongoDB configuration
MONGO_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGO_PORT = os.environ.get('MONGO_INITDB_PORT', '27017')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME', 'admin')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD', 'password')
MONGO_DATABASE = os.environ.get('MONGO_INITDB_DATABASE', 'rag_system')

MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"

# App configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Create upload folder
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Initialize Redis
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("Redis connected successfully")
except ConnectionError:
    print("Warning: Redis connection failed. Some features may be unavailable.")
    redis_client = None

# Initialize MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DATABASE]

def get_collections():
    """Get MongoDB collections"""
    return {
        'content_groups': db.content_groups,
        'chat_sessions': db.chat_sessions,
        'messages': db.messages,
        'file_uploads': db.file_uploads,
    }

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def wait_for_result(key: str, timeout: int = 30) -> Optional[Dict]:
    """Wait for async result from Redis"""
    if not redis_client:
        return None
    
    for _ in range(timeout * 10):  # Check every 100ms
        result = redis_client.get(key)
        if result:
            redis_client.delete(key)  # Clean up
            return json.loads(result)
        time.sleep(0.1)
    
    return None

def stream_inference_response(task_id: str, timeout: int = 120) -> Generator[str, None, None]:
    """Stream inference response tokens as they arrive"""
    if not redis_client:
        yield "data: {\"error\": \"Redis not available\"}\n\n"
        return
    
    result_key = f"inference_result:{task_id}"
    complete_key = f"inference_complete:{task_id}"
    start_time = time.time()
    full_response = ""
    token_count = 0
    
    while time.time() - start_time < timeout:
        try:
            # Check for tokens with non-blocking pop
            token_data = redis_client.brpop(result_key, timeout=0.1)
            
            if token_data:
                _, token_content = token_data
                
                # Check for stop signal
                if token_content == "<|STOP|>":
                    # Get the complete result
                    result_json = redis_client.get(complete_key)
                    
                    if result_json:
                        result = json.loads(result_json)
                        # Send final metadata
                        yield f"data: {json.dumps({'type': 'metadata', 'generation_time': result.get('generation_time', 0), 'model_used': result.get('model_used', 'unknown'), 'context_used': result.get('context_used', 0), 'timed_out': result.get('timed_out', False)})}\n\n"
                    
                    # Send done signal
                    yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"
                    
                    # Clean up
                    redis_client.delete(result_key)
                    if result_json:
                        redis_client.delete(complete_key)
                    return
                
                # Stream the token
                full_response += token_content
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'content': token_content, 'token_count': token_count})}\n\n"
        
        except redis.exceptions.ResponseError as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return
    
    # Timeout occurred
    yield f"data: {json.dumps({'type': 'error', 'error': 'Response timeout', 'timed_out': True})}\n\n"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": "connected" if redis_client and redis_client.ping() else "disconnected",
            "mongodb": check_mongodb_health()
        }
    }
    return jsonify(status)

def check_mongodb_health():
    """Check MongoDB service health"""
    try:
        mongo_client.admin.command('ping')
        return "connected"
    except:
        return "disconnected"

@app.route('/api/groups', methods=['GET'])
def get_groups():
    """Get all content groups"""
    collections = get_collections()
    
    # Get all groups including default "General Chat"
    groups = list(collections['content_groups'].find(
        {},
        {'name': 1, 'description': 1, 'created_at': 1, 'file_count': 1}
    ).sort('created_at', -1))
    
    # Add General Chat as first group if not exists
    general_chat_exists = any(group.get('name') == 'General Chat' for group in groups)
    if not general_chat_exists:
        # Create General Chat group
        general_group = {
            '_id': 'general',
            'name': 'General Chat',
            'description': 'General conversation without document context',
            'created_at': datetime.utcnow(),
            'file_count': 0,
            'is_default': True
        }
        collections['content_groups'].update_one(
            {'_id': 'general'},
            {'$setOnInsert': general_group},
            upsert=True
        )
        groups.insert(0, general_group)
    
    # Convert datetime to string
    for group in groups:
        if group.get('created_at'):
            group['created_at'] = group['created_at'].isoformat()
    
    return jsonify({"groups": groups})

@app.route('/api/groups', methods=['POST'])
def create_group():
    """Create new content group"""
    data = request.json
    collections = get_collections()
    
    if not data or 'name' not in data:
        return jsonify({"error": "Group name is required"}), 400
    
    # Check if group name already exists
    existing_group = collections['content_groups'].find_one({'name': data['name']})
    if existing_group:
        return jsonify({"error": "Group name already exists"}), 409
    
    try:
        group_id = str(uuid.uuid4())
        group_data = {
            '_id': group_id,
            'name': data['name'],
            'description': data.get('description', ''),
            'file_count': 0,
            'message_count': 0,
            'settings': {
                'max_history': 50,
                'auto_archive': False,
                'embedding_model': EMBEDDING_MODEL
            },
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        collections['content_groups'].insert_one(group_data)
        
        return jsonify({
            "_id": group_id,
            "name": data['name'],
            "description": data.get('description', ''),
            "file_count": 0,
            "created_at": datetime.utcnow().isoformat()
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/groups/<group_id>/files', methods=['POST'])
def upload_file_to_group(group_id: str):
    """Upload and process a file to a specific group"""
    collections = get_collections()
    
    # Verify group exists (except for general)
    if group_id != 'general':
        group = collections['content_groups'].find_one({'_id': group_id})
        if not group:
            return jsonify({"error": "Group not found"}), 404
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Maximum size is 50MB"}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    file_ext = filename.rsplit('.', 1)[1].lower()
    saved_filename = f"{group_id}_{file_id}.{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, saved_filename)
    
    file.save(file_path)
    
    # Calculate checksum
    with open(file_path, 'rb') as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    try:
        file_data = {
            '_id': file_id,
            'filename': filename,
            'original_filename': filename,
            'file_path': file_path,
            'file_type': file_ext,
            'file_size': file_size,
            'group_id': group_id,
            'upload_status': 'processing',
            'processing_progress': 0,
            'chunks_count': 0,
            'checksum': checksum,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        collections['file_uploads'].insert_one(file_data)
        
        # Process file asynchronously only if not general chat
        if redis_client and group_id != 'general':
            task = {
                "id": str(uuid.uuid4()),
                "type": "process_file",
                "file_id": file_id,
                "filename": filename,
                "file_path": file_path,
                "file_type": file_ext,
                "group_id": group_id
            }
            redis_client.lpush("file_processing_tasks", json.dumps(task))
            
            # Update group file count
            collections['content_groups'].update_one(
                {'_id': group_id},
                {'$inc': {'file_count': 1}}
            )
        else:
            # For general chat, mark as completed but don't process
            collections['file_uploads'].update_one(
                {'_id': file_id},
                {'$set': {'upload_status': 'skipped', 'processing_progress': 100}}
            )
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "group_id": group_id,
            "status": "processing" if group_id != 'general' else "skipped"
        })
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up file
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/<file_id>/status', methods=['GET'])
def get_file_status(file_id: str):
    """Get status of file upload/processing"""
    collections = get_collections()
    
    file_doc = collections['file_uploads'].find_one({'_id': file_id})
    
    if not file_doc:
        return jsonify({"error": "File not found"}), 404
    
    # Convert datetime objects to ISO format
    if 'created_at' in file_doc:
        file_doc['created_at'] = file_doc['created_at'].isoformat()
    if 'processed_at' in file_doc:
        file_doc['processed_at'] = file_doc['processed_at'].isoformat()
    if 'updated_at' in file_doc:
        file_doc['updated_at'] = file_doc['updated_at'].isoformat()
    
    return jsonify(file_doc)

@app.route('/api/groups/<group_id>/files', methods=['GET'])
def get_group_files(group_id: str):
    """Get all files in a group"""
    collections = get_collections()
    
    # Verify group exists (except for general)
    if group_id != 'general':
        group = collections['content_groups'].find_one({'_id': group_id})
        if not group:
            return jsonify({"error": "Group not found"}), 404
    
    files = list(collections['file_uploads'].find(
        {'group_id': group_id},
        {'filename': 1, 'file_size': 1, 'upload_status': 1, 'chunks_count': 1, 'created_at': 1}
    ).sort('created_at', -1))
    
    # Convert datetime to string
    for file_doc in files:
        if file_doc.get('created_at'):
            file_doc['created_at'] = file_doc['created_at'].isoformat()
    
    return jsonify({
        "group_id": group_id,
        "files": files,
        "total_files": len(files)
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with streaming support"""
    data = request.json
    collections = get_collections()
    
    # Validate input
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data['message']
    group_id = data.get('group_id', 'general')
    session_id = data.get('session_id', str(uuid.uuid4()))
    stream = data.get('stream', False)  # New parameter to enable streaming
    
    # Verify group exists (except for general)
    if group_id != 'general':
        group = collections['content_groups'].find_one({'_id': group_id})
        if not group:
            return jsonify({"error": "Group not found"}), 404
    
    try:
        # Ensure session exists
        session_data = {
            '_id': session_id,
            'group_id': group_id,
            'title': data.get('title', message[:50] + '...' if len(message) > 50 else message),
            'status': 'active',
            'message_count': 0,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        collections['chat_sessions'].update_one(
            {'_id': session_id},
            {'$setOnInsert': session_data, '$set': {'last_message_at': datetime.utcnow()}},
            upsert=True
        )
        
        # Create message record
        message_id = str(uuid.uuid4())
        message_data = {
            '_id': message_id,
            'session_id': session_id,
            'group_id': group_id,
            'message_type': 'user',
            'content': message,
            'query': message,
            'created_at': datetime.utcnow()
        }
        
        collections['messages'].insert_one(message_data)
        
        # Update session message count
        collections['chat_sessions'].update_one(
            {'_id': session_id},
            {'$inc': {'message_count': 1}}
        )
        
        # Update group message count
        if group_id != 'general':
            collections['content_groups'].update_one(
                {'_id': group_id},
                {'$inc': {'message_count': 1}}
            )
        
    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({"error": "Database error"}), 500
    
    # Create task for async processing
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": "chat",
        "session_id": session_id,
        "message_id": message_id,
        "group_id": group_id,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not redis_client:
        return jsonify({
            "session_id": session_id,
            "response": "I'm currently unable to process your request. Please try again later.",
            "task_id": task_id,
            "error": "Service unavailable"
        }), 503
    
    # Get conversation history
    history = list(collections['messages'].find(
        {'session_id': session_id},
        {'_id': 0, 'message_type': 1, 'content': 1}
    ).sort('created_at', -1).limit(10))
    
    task["context"] = list(reversed(history))
    
    # Function to handle the complete flow and stream response
    def generate_streaming_response():
        response_id = str(uuid.uuid4())
        retrieved_docs = []
        
        try:
            # Send initial event
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id, 'message_id': response_id, 'task_id': task_id})}\n\n"
            
            if group_id == 'general':
                # For General Chat, skip RAG and go directly to inference
                task["retrieved_context"] = []
                redis_client.lpush("inference_tasks", json.dumps(task))
                
                # Stream the response
                full_response = ""
                generation_time = 0
                model_used = "unknown"
                timed_out = False
                
                for event in stream_inference_response(task_id):
                    yield event
                    
                    # Parse the event to extract data
                    if event.startswith("data: "):
                        try:
                            event_data = json.loads(event[6:].strip())
                            if event_data.get('type') == 'token':
                                full_response += event_data.get('content', '')
                            elif event_data.get('type') == 'metadata':
                                generation_time = event_data.get('generation_time', 0)
                                model_used = event_data.get('model_used', 'unknown')
                                timed_out = event_data.get('timed_out', False)
                        except:
                            pass
                
            else:
                # For content groups, do RAG retrieval first
                redis_client.lpush("embedding_tasks", json.dumps(task))
                
                # Send status update
                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents...'})}\n\n"
                
                # Wait for retrieval results
                retrieval_result = wait_for_result(f"retrieval_result:{task_id}", timeout=10)
                
                if retrieval_result:
                    retrieved_docs = retrieval_result.get("documents", [])
                    
                    # Send retrieved context info
                    yield f"data: {json.dumps({'type': 'context', 'documents': len(retrieved_docs)})}\n\n"
                    
                    # Add retrieved context to task
                    task["retrieved_context"] = retrieved_docs
                
                # Send to inference service
                redis_client.lpush("inference_tasks", json.dumps(task))
                
                # Stream the response
                full_response = ""
                generation_time = 0
                model_used = "unknown"
                timed_out = False
                
                for event in stream_inference_response(task_id):
                    yield event
                    
                    # Parse the event to extract data
                    if event.startswith("data: "):
                        try:
                            event_data = json.loads(event[6:].strip())
                            if event_data.get('type') == 'token':
                                full_response += event_data.get('content', '')
                            elif event_data.get('type') == 'metadata':
                                generation_time = event_data.get('generation_time', 0)
                                model_used = event_data.get('model_used', 'unknown')
                                timed_out = event_data.get('timed_out', False)
                        except:
                            pass
            
            # Store the complete response in database
            if full_response:
                response_data = {
                    '_id': response_id,
                    'session_id': session_id,
                    'group_id': group_id,
                    'message_type': 'assistant',
                    'content': full_response,
                    'response': full_response,
                    'context_used': retrieved_docs,
                    'model_used': model_used,
                    'generation_time': generation_time,
                    'timed_out': timed_out,
                    'created_at': datetime.utcnow()
                }
                
                collections['messages'].insert_one(response_data)
                
                # Update session
                collections['chat_sessions'].update_one(
                    {'_id': session_id},
                    {
                        '$inc': {'message_count': 1},
                        '$set': {'last_message_at': datetime.utcnow()}
                    }
                )
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    # If streaming is requested, return SSE response
    if stream:
        return Response(
            stream_with_context(generate_streaming_response()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering
                "Connection": "keep-alive",
            }
        )
    
    # Otherwise, use the original non-streaming approach
    else:
        # Handle based on group type
        if group_id == 'general':
            # For General Chat, skip RAG and go directly to inference
            task["retrieved_context"] = []
            redis_client.lpush("inference_tasks", json.dumps(task))
            
            # Wait for inference result
            inference_result = wait_for_result(f"inference_complete:{task_id}", timeout=30)
            
            if inference_result:
                response_text = inference_result.get("response", "Sorry, I couldn't generate a response.")
                
                # Store assistant response
                response_id = str(uuid.uuid4())
                response_data = {
                    '_id': response_id,
                    'session_id': session_id,
                    'group_id': group_id,
                    'message_type': 'assistant',
                    'content': response_text,
                    'response': response_text,
                    'context_used': [],
                    'model_used': inference_result.get("model_used", "unknown"),
                    'generation_time': inference_result.get("generation_time", 0),
                    'created_at': datetime.utcnow()
                }
                
                collections['messages'].insert_one(response_data)
                
                # Update session
                collections['chat_sessions'].update_one(
                    {'_id': session_id},
                    {
                        '$inc': {'message_count': 1},
                        '$set': {'last_message_at': datetime.utcnow()}
                    }
                )
                
                return jsonify({
                    "session_id": session_id,
                    "message_id": response_id,
                    "response": response_text,
                    "retrieved_context": [],
                    "task_id": task_id,
                    "group_type": "general"
                })
        else:
            # For content groups, do RAG retrieval first
            redis_client.lpush("embedding_tasks", json.dumps(task))
            
            # Wait for retrieval results
            retrieval_result = wait_for_result(f"retrieval_result:{task_id}", timeout=10)
            
            if retrieval_result:
                # Store retrieved documents
                retrieved_docs = retrieval_result.get("documents", [])
                
                # Add retrieved context to task
                task["retrieved_context"] = retrieved_docs
                
                # Send to inference service
                redis_client.lpush("inference_tasks", json.dumps(task))
                
                # Wait for inference result
                inference_result = wait_for_result(f"inference_complete:{task_id}", timeout=60)
                
                if inference_result:
                    response_text = inference_result.get("response", "Sorry, I couldn't generate a response.")
                    
                    # Store assistant response
                    response_id = str(uuid.uuid4())
                    response_data = {
                        '_id': response_id,
                        'session_id': session_id,
                        'group_id': group_id,
                        'message_type': 'assistant',
                        'content': response_text,
                        'response': response_text,
                        'context_used': retrieved_docs,
                        'model_used': inference_result.get("model_used", "unknown"),
                        'generation_time': inference_result.get("generation_time", 0),
                        'created_at': datetime.utcnow()
                    }
                    
                    collections['messages'].insert_one(response_data)
                    
                    # Update session
                    collections['chat_sessions'].update_one(
                        {'_id': session_id},
                        {
                            '$inc': {'message_count': 1},
                            '$set': {'last_message_at': datetime.utcnow()}
                        }
                    )
                    
                    return jsonify({
                        "session_id": session_id,
                        "message_id": response_id,
                        "response": response_text,
                        "retrieved_context": retrieved_docs,
                        "task_id": task_id,
                        "group_type": "rag"
                    })
        
        # Fallback response
        return jsonify({
            "session_id": session_id,
            "response": "I'm currently unable to process your request. Please try again later.",
            "task_id": task_id,
            "error": "Service unavailable"
        }), 503

@app.route('/api/groups/<group_id>/sessions', methods=['GET'])
def get_group_sessions(group_id: str):
    """Get chat sessions for a specific group"""
    collections = get_collections()
    
    # Verify group exists (except for general)
    if group_id != 'general':
        group = collections['content_groups'].find_one({'_id': group_id})
        if not group:
            return jsonify({"error": "Group not found"}), 404
    
    # Get pagination parameters
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    
    # Get sessions with message count
    sessions = list(collections['chat_sessions'].aggregate([
        {'$match': {'group_id': group_id}},
        {
            '$lookup': {
                'from': 'messages',
                'localField': '_id',
                'foreignField': 'session_id',
                'as': 'messages'
            }
        },
        {
            '$addFields': {
                'actual_message_count': {'$size': '$messages'},
                'preview': {
                    '$substr': [
                        {'$arrayElemAt': ['$messages.content', 0]}, 0, 100
                    ]
                }
            }
        },
        {
            '$project': {
                'session_id': '$_id',
                'title': 1,
                'message_count': '$actual_message_count',
                'preview': 1,
                'last_message_at': 1,
                'created_at': 1,
                'status': 1
            }
        },
        {'$sort': {'last_message_at': -1}},
        {'$skip': (page - 1) * limit},
        {'$limit': limit}
    ]))
    
    # Get total count
    total = collections['chat_sessions'].count_documents({'group_id': group_id})
    
    # Convert datetime to string
    for session in sessions:
        if session.get('created_at'):
            session['created_at'] = session['created_at'].isoformat()
        if session.get('last_message_at'):
            session['last_message_at'] = session['last_message_at'].isoformat()
    
    return jsonify({
        "group_id": group_id,
        "sessions": sessions,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    })

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Get specific session details with messages"""
    collections = get_collections()
    
    # Get session
    session = collections['chat_sessions'].find_one({'_id': session_id})
    
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    # Get messages
    messages = list(collections['messages'].find(
        {'session_id': session_id},
        {'content': 1, 'message_type': 1, 'created_at': 1, 'context_used': 1, 'model_used': 1}
    ).sort('created_at', 1))
    
    # Format response
    if session.get('created_at'):
        session['created_at'] = session['created_at'].isoformat()
    if session.get('last_message_at'):
        session['last_message_at'] = session['last_message_at'].isoformat()
    
    for message in messages:
        if message.get('created_at'):
            message['created_at'] = message['created_at'].isoformat()
    
    session['messages'] = messages
    
    return jsonify(session)

@app.route('/api/groups/<group_id>', methods=['DELETE'])
def delete_group(group_id: str):
    """Delete a content group and all associated data"""
    if group_id == 'general':
        return jsonify({"error": "Cannot delete General Chat group"}), 400
    
    collections = get_collections()
    
    # Check if group exists
    group = collections['content_groups'].find_one({'_id': group_id})
    if not group:
        return jsonify({"error": "Group not found"}), 404
    
    try:
        # Delete all files associated with the group
        files = list(collections['file_uploads'].find({'group_id': group_id}))
        for file_doc in files:
            file_path = file_doc.get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        # Delete file records
        collections['file_uploads'].delete_many({'group_id': group_id})
        
        # Delete all messages in this group
        collections['messages'].delete_many({'group_id': group_id})
        
        # Delete all sessions in this group
        collections['chat_sessions'].delete_many({'group_id': group_id})
        
        # Finally, delete the group
        collections['content_groups'].delete_one({'_id': group_id})
        
        return jsonify({"success": True, "message": "Group deleted successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """Delete a chat session and all its messages"""
    collections = get_collections()
    
    # Check if session exists
    session = collections['chat_sessions'].find_one({'_id': session_id})
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    try:
        # Delete all messages in this session
        collections['messages'].delete_many({'session_id': session_id})
        
        # Delete the session
        collections['chat_sessions'].delete_one({'_id': session_id})
        
        return jsonify({"success": True, "message": "Session deleted successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get overall system statistics"""
    collections = get_collections()
    
    try:
        stats = {
            "total_groups": collections['content_groups'].count_documents({}),
            "total_files": collections['file_uploads'].count_documents({}),
            "total_sessions": collections['chat_sessions'].count_documents({}),
            "total_messages": collections['messages'].count_documents({}),
            "groups": []
        }
        
        # Get per-group statistics
        groups = list(collections['content_groups'].find({}, {'name': 1}))
        for group in groups:
            group_stats = {
                "group_id": group['_id'],
                "group_name": group['name'],
                "files": collections['file_uploads'].count_documents({'group_id': group['_id']}),
                "sessions": collections['chat_sessions'].count_documents({'group_id': group['_id']}),
                "messages": collections['messages'].count_documents({'group_id': group['_id']})
            }
            stats["groups"].append(group_stats)
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('API_SERVICE_PORT', 5000))
    debug = os.environ.get('API_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting RAG API Server with Streaming Support on port {port}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.1f}MB")
    print(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
    print(f"Streaming chat: Enabled via 'stream' parameter")
    
    app.run(host='0.0.0.0', port=port, debug=debug)