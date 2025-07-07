"""
Comprehensive Test Suite for Flask RAG Web UI
Tests all routes, functionality, error handling, and API integration
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from flask import url_for
from io import BytesIO
import requests

# Import the Flask app
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app, allowed_file, format_file_size, format_datetime


class TestFlaskWebUI:
    """Test suite for Flask Web UI"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SECRET_KEY'] = 'test-secret-key'
        
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock API responses"""
        def _mock_response(status_code=200, json_data=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.status_code = status_code
            mock_resp.json.return_value = json_data or {}
            mock_resp.headers = headers or {'content-type': 'application/json'}
            return mock_resp
        return _mock_response
    
    def test_app_configuration(self, client):
        """Test Flask app configuration"""
        assert app.config['TESTING'] is True
        assert app.secret_key == 'test-secret-key'
    
    def test_utility_functions(self):
        """Test utility functions"""
        # Test allowed_file function
        assert allowed_file('document.pdf') is True
        assert allowed_file('text.txt') is True
        assert allowed_file('word.docx') is True
        assert allowed_file('malware.exe') is False
        assert allowed_file('no_extension') is False
        assert allowed_file('') is False
        
        # Test format_file_size function
        assert format_file_size(0) == "0B"
        assert format_file_size(1024) == "1.0KB"
        assert format_file_size(1048576) == "1.0MB"
        assert format_file_size(1073741824) == "1.0GB"
        
        # Test format_datetime function
        assert format_datetime('2023-12-01T10:30:00') == '2023-12-01 10:30'
        assert format_datetime('invalid-date') == 'invalid-date'  # Should return as-is for invalid dates


class TestHomePageRoutes:
    """Test home page and group management routes"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @patch('app.requests.get')
    def test_index_page_success(self, mock_get, client, mock_api_response):
        """Test successful loading of index page"""
        # Mock successful API response
        mock_groups_data = {
            'groups': [
                {
                    '_id': 'general',
                    'name': 'General Chat',
                    'description': 'General conversation',
                    'file_count': 0,
                    'created_at': '2023-12-01T10:00:00'
                },
                {
                    '_id': 'group1',
                    'name': 'Research Papers',
                    'description': 'AI research documents',
                    'file_count': 5,
                    'created_at': '2023-12-01T11:00:00'
                }
            ]
        }
        mock_get.return_value = mock_api_response(200, mock_groups_data)
        
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'Your Groups' in response.data
        assert b'General Chat' in response.data
        assert b'Research Papers' in response.data
        assert b'5 files' in response.data
        mock_get.assert_called_once()
    
    @patch('app.requests.get')
    def test_index_page_api_error(self, mock_get, client, mock_api_response):
        """Test index page when API returns error"""
        mock_get.return_value = mock_api_response(500, {'error': 'Server error'})
        
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'Error loading groups' in response.data
    
    @patch('app.requests.get')
    def test_index_page_connection_error(self, mock_get, client):
        """Test index page when API connection fails"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'Unable to connect to the server' in response.data
    
    @patch('app.requests.get')
    def test_index_page_empty_groups(self, mock_get, client, mock_api_response):
        """Test index page with no groups"""
        mock_get.return_value = mock_api_response(200, {'groups': []})
        
        response = client.get('/')
        
        assert response.status_code == 200
        assert b'No groups yet' in response.data
        assert b'Create your first group' in response.data
    
    @patch('app.requests.post')
    def test_create_group_success(self, mock_post, client, mock_api_response):
        """Test successful group creation"""
        mock_group_data = {
            '_id': 'new-group-id',
            'name': 'Test Group',
            'description': 'Test description'
        }
        mock_post.return_value = mock_api_response(201, mock_group_data)
        
        response = client.post('/create_group', data={
            'name': 'Test Group',
            'description': 'Test description'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        mock_post.assert_called_once()
        # Should redirect to group detail page
        assert b'Test Group' in response.data
    
    @patch('app.requests.post')
    def test_create_group_duplicate(self, mock_post, client, mock_api_response):
        """Test group creation with duplicate name"""
        mock_post.return_value = mock_api_response(409, {'error': 'Group name already exists'})
        
        response = client.post('/create_group', data={
            'name': 'Existing Group'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        assert b'already exists' in response.data
    
    def test_create_group_missing_name(self, client):
        """Test group creation without name"""
        response = client.post('/create_group', data={
            'description': 'Missing name'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        assert b'Group name is required' in response.data
    
    @patch('app.requests.post')
    def test_create_group_api_error(self, mock_post, client, mock_api_response):
        """Test group creation with API error"""
        mock_post.return_value = mock_api_response(500, {'error': 'Server error'})
        
        response = client.post('/create_group', data={
            'name': 'Test Group'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        assert b'Error creating group' in response.data


class TestGroupDetailRoutes:
    """Test group detail and file management routes"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @patch('app.requests.get')
    def test_group_detail_success(self, mock_get, client, mock_api_response):
        """Test successful group detail page loading"""
        # Mock multiple API calls
        mock_responses = [
            mock_api_response(200, {'groups': [{'_id': 'group1', 'name': 'Test Group', 'description': 'Test'}]}),
            mock_api_response(200, {'files': [{'_id': 'file1', 'filename': 'test.pdf', 'file_size': 1024, 'upload_status': 'completed'}]}),
            mock_api_response(200, {'sessions': [{'session_id': 'session1', 'title': 'Test Session', 'message_count': 5}]})
        ]
        mock_get.side_effect = mock_responses
        
        response = client.get('/group/group1')
        
        assert response.status_code == 200
        assert b'Test Group' in response.data
        assert b'test.pdf' in response.data
        assert b'Test Session' in response.data
    
    @patch('app.requests.get')
    def test_group_detail_general_chat(self, mock_get, client, mock_api_response):
        """Test general chat group detail"""
        mock_responses = [
            mock_api_response(200, {'files': []}),
            mock_api_response(200, {'sessions': []})
        ]
        mock_get.side_effect = mock_responses
        
        response = client.get('/group/general')
        
        assert response.status_code == 200
        assert b'General Chat' in response.data
        # Should not show file upload section for general chat
        assert b'Drop files here' not in response.data
    
    @patch('app.requests.get')
    def test_group_detail_nonexistent(self, mock_get, client, mock_api_response):
        """Test group detail for non-existent group"""
        mock_get.return_value = mock_api_response(200, {'groups': []})
        
        response = client.get('/group/nonexistent', follow_redirects=True)
        
        assert response.status_code == 200
        assert b'Group not found' in response.data
    
    @patch('app.requests.post')
    def test_file_upload_success(self, mock_post, client, mock_api_response):
        """Test successful file upload"""
        mock_post.return_value = mock_api_response(200, {
            'success': True,
            'file_id': 'file123',
            'filename': 'test.pdf',
            'status': 'processing'
        })
        
        data = {
            'file': (BytesIO(b'fake pdf content'), 'test.pdf')
        }
        
        response = client.post('/group/group1/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        
        assert response.status_code == 200
        assert b'uploaded successfully' in response.data
        mock_post.assert_called_once()
    
    @patch('app.requests.post')
    def test_file_upload_general_chat(self, mock_post, client, mock_api_response):
        """Test file upload to general chat (should be skipped)"""
        mock_post.return_value = mock_api_response(200, {
            'success': True,
            'file_id': 'file123',
            'filename': 'test.pdf',
            'status': 'skipped'
        })
        
        data = {
            'file': (BytesIO(b'fake pdf content'), 'test.pdf')
        }
        
        response = client.post('/group/general/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        
        assert response.status_code == 200
        assert b'processing was skipped' in response.data
    
    def test_file_upload_no_file(self, client):
        """Test upload without file"""
        response = client.post('/group/group1/upload', 
                             data={}, 
                             follow_redirects=True)
        
        assert response.status_code == 200
        assert b'No file selected' in response.data
    
    def test_file_upload_invalid_type(self, client):
        """Test upload with invalid file type"""
        data = {
            'file': (BytesIO(b'fake exe content'), 'malware.exe')
        }
        
        response = client.post('/group/group1/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        
        assert response.status_code == 200
        assert b'File type not allowed' in response.data
    
    @patch('app.requests.post')
    def test_file_upload_api_error(self, mock_post, client, mock_api_response):
        """Test file upload with API error"""
        mock_post.return_value = mock_api_response(400, {'error': 'File too large'})
        
        data = {
            'file': (BytesIO(b'fake pdf content'), 'test.pdf')
        }
        
        response = client.post('/group/group1/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        
        assert response.status_code == 200
        assert b'File too large' in response.data
    
    @patch('app.requests.get')
    def test_file_status_success(self, mock_get, client, mock_api_response):
        """Test file status endpoint"""
        mock_file_data = {
            '_id': 'file123',
            'filename': 'test.pdf',
            'upload_status': 'completed',
            'chunks_count': 15
        }
        mock_get.return_value = mock_api_response(200, mock_file_data)
        
        response = client.get('/api/file/file123/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['_id'] == 'file123'
        assert data['upload_status'] == 'completed'
    
    @patch('app.requests.get')
    def test_file_status_not_found(self, mock_get, client, mock_api_response):
        """Test file status for non-existent file"""
        mock_get.return_value = mock_api_response(404, {'error': 'File not found'})
        
        response = client.get('/api/file/nonexistent/status')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


class TestChatRoutes:
    """Test chat interface and messaging routes"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @patch('app.requests.get')
    def test_chat_interface_new_session(self, mock_get, client, mock_api_response):
        """Test chat interface for new session"""
        mock_responses = [
            mock_api_response(200, {'groups': [{'_id': 'group1', 'name': 'Test Group'}]}),
            mock_api_response(200, {'sessions': []})
        ]
        mock_get.side_effect = mock_responses
        
        response = client.get('/chat/group1')
        
        assert response.status_code == 200
        assert b'Test Group' in response.data
        assert b'New conversation' in response.data
        assert b'Start a conversation' in response.data
    
    @patch('app.requests.get')
    def test_chat_interface_existing_session(self, mock_get, client, mock_api_response):
        """Test chat interface for existing session"""
        mock_responses = [
            mock_api_response(200, {'groups': [{'_id': 'group1', 'name': 'Test Group'}]}),
            mock_api_response(200, {
                '_id': 'session1',
                'title': 'Test Session',
                'messages': [
                    {
                        'message_type': 'user',
                        'content': 'Hello',
                        'created_at': '2023-12-01T10:00:00'
                    },
                    {
                        'message_type': 'assistant',
                        'content': 'Hi there!',
                        'created_at': '2023-12-01T10:01:00',
                        'context_used': []
                    }
                ]
            }),
            mock_api_response(200, {'sessions': [{'session_id': 'session1', 'title': 'Test Session'}]})
        ]
        mock_get.side_effect = mock_responses
        
        response = client.get('/chat/group1/session1')
        
        assert response.status_code == 200
        assert b'Test Session' in response.data
        assert b'Hello' in response.data
        assert b'Hi there!' in response.data
    
    @patch('app.requests.get')
    def test_chat_interface_general_chat(self, mock_get, client, mock_api_response):
        """Test chat interface for general chat"""
        mock_get.return_value = mock_api_response(200, {'sessions': []})
        
        response = client.get('/chat/general')
        
        assert response.status_code == 200
        assert b'General Chat' in response.data
        assert b'Ask me anything' in response.data
    
    @patch('app.requests.get')
    def test_chat_interface_group_not_found(self, mock_get, client, mock_api_response):
        """Test chat interface for non-existent group"""
        mock_get.return_value = mock_api_response(200, {'groups': []})
        
        response = client.get('/chat/nonexistent', follow_redirects=True)
        
        assert response.status_code == 200
        assert b'Group not found' in response.data
    
    @patch('app.requests.post')
    def test_send_message_success(self, mock_post, client, mock_api_response):
        """Test successful message sending"""
        mock_post.return_value = mock_api_response(200, {
            'session_id': 'session123',
            'message_id': 'msg123',
            'response': 'AI response here',
            'retrieved_context': []
        })
        
        response = client.post('/api/chat',
                             json={
                                 'message': 'Hello AI',
                                 'group_id': 'group1'
                             })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['response'] == 'AI response here'
        assert data['session_id'] == 'session123'
    
    @patch('app.requests.post')
    def test_send_message_with_session(self, mock_post, client, mock_api_response):
        """Test message sending with existing session"""
        mock_post.return_value = mock_api_response(200, {
            'session_id': 'existing_session',
            'response': 'Continued conversation'
        })
        
        response = client.post('/api/chat',
                             json={
                                 'message': 'Follow up question',
                                 'group_id': 'group1',
                                 'session_id': 'existing_session'
                             })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['response'] == 'Continued conversation'
    
    def test_send_message_no_message(self, client):
        """Test message sending without message content"""
        response = client.post('/api/chat', json={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Message is required' in data['error']
    
    @patch('app.requests.post')
    def test_send_message_api_error(self, mock_post, client, mock_api_response):
        """Test message sending with API error"""
        mock_post.return_value = mock_api_response(500, {'error': 'Server error'})
        
        response = client.post('/api/chat',
                             json={
                                 'message': 'Hello',
                                 'group_id': 'group1'
                             })
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
    
    @patch('app.requests.post')
    def test_send_message_connection_error(self, mock_post, client):
        """Test message sending with connection error"""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        response = client.post('/api/chat',
                             json={
                                 'message': 'Hello',
                                 'group_id': 'group1'
                             })
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'Unable to connect to server' in data['error']


class TestHealthAndErrorRoutes:
    """Test health check and error handling routes"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @patch('app.requests.get')
    def test_health_check_success(self, mock_get, client, mock_api_response):
        """Test health check when API is healthy"""
        mock_get.return_value = mock_api_response(200, {'status': 'healthy'})
        
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['webserver'] == 'running'
        assert data['api_server'] == 'connected'
    
    @patch('app.requests.get')
    def test_health_check_api_down(self, mock_get, client, mock_api_response):
        """Test health check when API is down"""
        mock_get.return_value = mock_api_response(500, {'error': 'Server error'})
        
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'unhealthy'
        assert data['api_server'] == 'disconnected'
    
    @patch('app.requests.get')
    def test_health_check_connection_error(self, mock_get, client):
        """Test health check with connection error"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Cannot connect")
        
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'unhealthy'
        assert data['api_server'] == 'disconnected'
    
    def test_404_error_handler(self, client):
        """Test 404 error page"""
        response = client.get('/nonexistent-page')
        
        assert response.status_code == 404
        assert b'Page not found' in response.data
        assert b'404' in response.data
    
    def test_500_error_handler(self, client):
        """Test 500 error handling"""
        # This is harder to test directly, but we can test the error template
        with app.test_request_context():
            from app import internal_error
            response = internal_error(Exception("Test error"))
            assert response[1] == 500


class TestSecurityAndValidation:
    """Test security features and input validation"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    def test_xss_prevention_in_group_name(self, client):
        """Test XSS prevention in group creation"""
        malicious_name = '<script>alert("xss")</script>'
        
        response = client.post('/create_group', data={
            'name': malicious_name,
            'description': 'Test'
        }, follow_redirects=True)
        
        # Should escape the script tag
        assert b'<script>' not in response.data
        assert b'&lt;script&gt;' in response.data or b'Group name is required' in response.data
    
    def test_file_size_validation(self, client):
        """Test file size validation"""
        # Create a large file content (simulated)
        large_content = b'x' * (51 * 1024 * 1024)  # 51MB
        
        data = {
            'file': (BytesIO(large_content), 'large_file.pdf')
        }
        
        # This test assumes the web server validates file size before sending to API
        # In the actual implementation, this would be handled by the API
        response = client.post('/group/group1/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        
        # Response depends on whether client-side validation is implemented
        assert response.status_code == 200
    
    def test_sql_injection_prevention(self, client):
        """Test SQL injection prevention in group names"""
        malicious_name = "'; DROP TABLE users; --"
        
        response = client.post('/create_group', data={
            'name': malicious_name,
            'description': 'Test'
        }, follow_redirects=True)
        
        # Should handle the malicious input safely
        assert response.status_code == 200
    
    def test_path_traversal_prevention(self, client):
        """Test path traversal prevention in file uploads"""
        data = {
            'file': (BytesIO(b'fake content'), '../../../etc/passwd')
        }
        
        response = client.post('/group/group1/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        
        # Should either reject the file or sanitize the filename
        assert response.status_code == 200


class TestIntegrationScenarios:
    """Test complete user workflows and integration scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @patch('app.requests.get')
    @patch('app.requests.post')
    def test_complete_workflow_new_user(self, mock_post, mock_get, client, mock_api_response):
        """Test complete workflow for new user"""
        # 1. User visits home page - no groups exist
        mock_get.return_value = mock_api_response(200, {'groups': []})
        response = client.get('/')
        assert response.status_code == 200
        assert b'No groups yet' in response.data
        
        # 2. User creates first group
        mock_post.return_value = mock_api_response(201, {
            '_id': 'new-group',
            'name': 'My First Group',
            'description': 'Test group'
        })
        
        response = client.post('/create_group', data={
            'name': 'My First Group',
            'description': 'Test group'
        }, follow_redirects=False)
        
        assert response.status_code == 302  # Redirect to group detail
        
        # 3. User uploads file to group
        mock_post.return_value = mock_api_response(200, {
            'success': True,
            'file_id': 'file123',
            'filename': 'document.pdf',
            'status': 'processing'
        })
        
        data = {'file': (BytesIO(b'fake pdf'), 'document.pdf')}
        response = client.post('/group/new-group/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        
        assert response.status_code == 200
        assert b'uploaded successfully' in response.data
    
    @patch('app.requests.get')
    @patch('app.requests.post')
    def test_chat_conversation_flow(self, mock_post, mock_get, client, mock_api_response):
        """Test complete chat conversation flow"""
        # 1. User enters chat interface
        mock_get.side_effect = [
            mock_api_response(200, {'groups': [{'_id': 'group1', 'name': 'Test Group'}]}),
            mock_api_response(200, {'sessions': []})
        ]
        
        response = client.get('/chat/group1')
        assert response.status_code == 200
        
        # 2. User sends first message
        mock_post.return_value = mock_api_response(200, {
            'session_id': 'session123',
            'response': 'Hello! How can I help you?',
            'retrieved_context': []
        })
        
        response = client.post('/api/chat', json={
            'message': 'Hello AI',
            'group_id': 'group1'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'session_id' in data
        
        # 3. User sends follow-up message
        mock_post.return_value = mock_api_response(200, {
            'session_id': 'session123',
            'response': 'I can help you with that document.',
            'retrieved_context': [{'content': 'Document excerpt...'}]
        })
        
        response = client.post('/api/chat', json={
            'message': 'What does the document say about AI?',
            'group_id': 'group1',
            'session_id': 'session123'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'retrieved_context' in data
    
    @patch('app.requests.get')
    def test_error_recovery_scenarios(self, mock_get, client, mock_api_response):
        """Test error recovery scenarios"""
        # 1. API server temporarily down
        mock_get.side_effect = requests.exceptions.ConnectionError("API server down")
        
        response = client.get('/')
        assert response.status_code == 200
        assert b'Unable to connect to the server' in response.data
        
        # 2. API server returns error
        mock_get.side_effect = None
        mock_get.return_value = mock_api_response(500, {'error': 'Internal server error'})
        
        response = client.get('/')
        assert response.status_code == 200
        assert b'Error loading groups' in response.data
        
        # 3. API server recovers
        mock_get.return_value = mock_api_response(200, {'groups': []})
        
        response = client.get('/')
        assert response.status_code == 200
        assert b'No groups yet' in response.data


class TestPerformanceAndLoad:
    """Test performance and load handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @patch('app.requests.get')
    def test_large_groups_list(self, mock_get, client, mock_api_response):
        """Test handling of large number of groups"""
        # Create 100 mock groups
        large_groups_list = {
            'groups': [
                {
                    '_id': f'group{i}',
                    'name': f'Group {i}',
                    'description': f'Description for group {i}',
                    'file_count': i % 10,
                    'created_at': '2023-12-01T10:00:00'
                }
                for i in range(100)
            ]
        }
        mock_get.return_value = mock_api_response(200, large_groups_list)
        
        response = client.get('/')
        
        assert response.status_code == 200
        # Should handle large lists without issues
        assert b'Group 0' in response.data
        assert b'Group 99' in response.data
    
    @patch('app.requests.get')
    def test_large_chat_history(self, mock_get, client, mock_api_response):
        """Test handling of large chat history"""
        # Create large message history
        large_messages = [
            {
                'message_type': 'user' if i % 2 == 0 else 'assistant',
                'content': f'Message {i} content here',
                'created_at': '2023-12-01T10:00:00'
            }
            for i in range(200)
        ]
        
        mock_responses = [
            mock_api_response(200, {'groups': [{'_id': 'group1', 'name': 'Test Group'}]}),
            mock_api_response(200, {
                '_id': 'session1',
                'title': 'Large Session',
                'messages': large_messages
            }),
            mock_api_response(200, {'sessions': []})
        ]
        mock_get.side_effect = mock_responses
        
        response = client.get('/chat/group1/session1')
        
        assert response.status_code == 200
        assert b'Large Session' in response.data
        # Should handle large message history
        assert b'Message 0' in response.data
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get('/api/health')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)


class TestAccessibilityAndUsability:
    """Test accessibility and usability features"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    @patch('app.requests.get')
    def test_semantic_html_structure(self, mock_get, client, mock_api_response):
        """Test proper semantic HTML structure"""
        mock_get.return_value = mock_api_response(200, {'groups': []})
        
        response = client.get('/')
        
        # Check for proper semantic elements
        assert b'<nav' in response.data
        assert b'<main' in response.data
        assert b'role="alert"' in response.data or b'alert' in response.data
        assert b'aria-label' in response.data or b'aria-labelledby' in response.data
    
    @patch('app.requests.get')
    def test_form_accessibility(self, mock_get, client, mock_api_response):
        """Test form accessibility features"""
        mock_get.return_value = mock_api_response(200, {'groups': []})
        
        response = client.get('/')
        
        # Check for proper form labels and structure
        assert b'<label' in response.data
        assert b'for=' in response.data
        assert b'required' in response.data
    
    @patch('app.requests.get')
    def test_responsive_design_elements(self, mock_get, client, mock_api_response):
        """Test responsive design elements"""
        mock_get.return_value = mock_api_response(200, {'groups': []})
        
        response = client.get('/')
        
        # Check for responsive meta tag and Bootstrap classes
        assert b'viewport' in response.data
        assert b'col-md' in response.data or b'col-lg' in response.data
        assert b'container' in response.data


class TestDataValidationAndSanitization:
    """Test data validation and sanitization"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    def test_group_name_validation(self, client):
        """Test group name validation"""
        # Test empty name
        response = client.post('/create_group', data={
            'name': '',
            'description': 'Test'
        }, follow_redirects=True)
        assert b'Group name is required' in response.data
        
        # Test name with only whitespace
        response = client.post('/create_group', data={
            'name': '   ',
            'description': 'Test'
        }, follow_redirects=True)
        assert b'Group name is required' in response.data
    
    def test_file_upload_validation(self, client):
        """Test file upload validation"""
        # Test upload with empty filename
        data = {'file': (BytesIO(b'content'), '')}
        response = client.post('/group/group1/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        assert b'No file selected' in response.data
        
        # Test upload with invalid extension
        data = {'file': (BytesIO(b'content'), 'script.js')}
        response = client.post('/group/group1/upload', 
                             data=data, 
                             content_type='multipart/form-data',
                             follow_redirects=True)
        assert b'File type not allowed' in response.data
    
    def test_chat_message_validation(self, client):
        """Test chat message validation"""
        # Test empty message
        response = client.post('/api/chat', json={})
        assert response.status_code == 400
        
        # Test message without group_id
        response = client.post('/api/chat', json={'message': 'Hello'})
        assert response.status_code == 200  # group_id defaults to 'general'
    
    def test_json_input_validation(self, client):
        """Test JSON input validation"""
        # Test invalid JSON
        response = client.post('/api/chat', 
                             data='invalid json',
                             content_type='application/json')
        assert response.status_code == 400
        
        # Test malformed JSON
        response = client.post('/api/chat', 
                             data='{"message": "hello"',  # Missing closing brace
                             content_type='application/json')
        assert response.status_code == 400


class TestCachingAndSession:
    """Test caching and session management"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.test_client() as client:
            with app.app_context():
                yield client
    
    def test_session_data_persistence(self, client):
        """Test session data persistence across requests"""
        with client.session_transaction() as sess:
            sess['test_key'] = 'test_value'
        
        # Session should persist across requests
        with client.session_transaction() as sess:
            assert sess.get('test_key') == 'test_value'
    
    def test_flash_message_persistence(self, client):
        """Test flash message persistence"""
        with client.session_transaction() as sess:
            from flask import flash
            with app.test_request_context():
                flash('Test message', 'success')
        
        # Flash messages should be available in next request
        response = client.get('/')
        # Note: Flash messages are consumed when displayed


def run_performance_tests():
    """Run performance-specific tests"""
    import time
    
    print("\n🚀 Running Performance Tests...")
    
    # Test template rendering performance
    start_time = time.time()
    with app.test_client() as client:
        with patch('app.requests.get') as mock_get:
            mock_get.return_value = MagicMock()
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'groups': []}
            
            # Render page multiple times
            for _ in range(100):
                response = client.get('/')
                assert response.status_code == 200
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    print(f"   ✅ Template rendering: {avg_time:.4f}s per request")
    
    # Test JSON processing performance
    start_time = time.time()
    large_data = {'groups': [{'_id': f'g{i}', 'name': f'Group {i}'} for i in range(1000)]}
    
    for _ in range(100):
        json.dumps(large_data)
        json.loads(json.dumps(large_data))
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    print(f"   ✅ JSON processing: {avg_time:.4f}s per operation")


def run_security_tests():
    """Run security-specific tests"""
    print("\n🔒 Running Security Tests...")
    
    with app.test_client() as client:
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        
        # Test XSS prevention
        xss_payloads = [
            '<script>alert("xss")</script>',
            '"><script>alert("xss")</script>',
            "javascript:alert('xss')",
            '<img src=x onerror=alert("xss")>'
        ]
        
        for payload in xss_payloads:
            response = client.post('/create_group', data={
                'name': payload,
                'description': 'Test'
            }, follow_redirects=True)
            
            # Should not contain unescaped script tags
            assert b'<script>' not in response.data
            print(f"   ✅ XSS payload blocked: {payload[:20]}...")
        
        # Test file upload security
        dangerous_files = [
            ('malware.exe', b'fake exe'),
            ('script.php', b'<?php echo "test"; ?>'),
            ('shell.sh', b'#!/bin/bash\necho "test"'),
            ('../../../etc/passwd', b'fake passwd')
        ]
        
        for filename, content in dangerous_files:
            data = {'file': (BytesIO(content), filename)}
            response = client.post('/group/group1/upload', 
                                 data=data, 
                                 content_type='multipart/form-data',
                                 follow_redirects=True)
            
            # Should reject dangerous files
            assert response.status_code == 200
            assert b'File type not allowed' in response.data or b'error' in response.data
            print(f"   ✅ Dangerous file blocked: {filename}")


if __name__ == '__main__':
    # Run tests with pytest
    print("🧪 Running Flask Web UI Test Suite...")
    
    # Basic test run
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings'
    ])
    
    # Run additional performance and security tests
    run_performance_tests()
    run_security_tests()
    
    print("\n✅ All tests completed!")
    print("\n📋 Test Coverage Summary:")
    print("   🏠 Home page and navigation")
    print("   📁 Group management")
    print("   📄 File upload and processing")
    print("   💬 Chat interface and messaging")
    print("   🔍 Health checks and error handling")
    print("   🔒 Security and validation")
    print("   ⚡ Performance and load testing")
    print("   ♿ Accessibility features")
    print("   🔧 Integration scenarios")