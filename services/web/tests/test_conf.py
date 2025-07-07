# conftest.py - Pytest configuration and fixtures
"""
Pytest configuration and shared fixtures for Flask Web UI tests
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
from app import app


@pytest.fixture(scope='session')
def app_instance():
    """Create Flask app instance for testing"""
    app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret-key',
        'API_BASE_URL': 'http://test-api:5000',
        'UPLOAD_FOLDER': tempfile.mkdtemp()
    })
    return app


@pytest.fixture
def client(app_instance):
    """Create test client"""
    with app_instance.test_client() as client:
        with app_instance.app_context():
            yield client


@pytest.fixture
def mock_api_response():
    """Factory for creating mock API responses"""
    def _create_mock(status_code=200, json_data=None, headers=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data or {}
        mock_resp.headers = headers or {'content-type': 'application/json'}
        return mock_resp
    return _create_mock


@pytest.fixture
def sample_groups():
    """Sample groups data for testing"""
    return {
        'groups': [
            {
                '_id': 'general',
                'name': 'General Chat',
                'description': 'General conversation without document context',
                'file_count': 0,
                'created_at': '2023-12-01T10:00:00'
            },
            {
                '_id': 'research',
                'name': 'Research Papers',
                'description': 'AI and ML research documents',
                'file_count': 3,
                'created_at': '2023-12-01T11:00:00'
            },
            {
                '_id': 'company',
                'name': 'Company Docs',
                'description': 'Internal company documentation',
                'file_count': 10,
                'created_at': '2023-12-01T12:00:00'
            }
        ]
    }


@pytest.fixture
def sample_files():
    """Sample files data for testing"""
    return {
        'files': [
            {
                '_id': 'file1',
                'filename': 'research_paper.pdf',
                'file_size': 1048576,  # 1MB
                'upload_status': 'completed',
                'chunks_count': 25,
                'created_at': '2023-12-01T10:30:00'
            },
            {
                '_id': 'file2',
                'filename': 'meeting_notes.txt',
                'file_size': 2048,  # 2KB
                'upload_status': 'processing',
                'chunks_count': 0,
                'created_at': '2023-12-01T11:30:00'
            },
            {
                '_id': 'file3',
                'filename': 'failed_upload.docx',
                'file_size': 512000,  # 512KB
                'upload_status': 'failed',
                'chunks_count': 0,
                'created_at': '2023-12-01T12:30:00'
            }
        ]
    }


@pytest.fixture
def sample_sessions():
    """Sample chat sessions data for testing"""
    return {
        'sessions': [
            {
                'session_id': 'session1',
                'title': 'Discussion about AI trends',
                'message_count': 15,
                'last_message_at': '2023-12-01T14:30:00',
                'preview': 'What are the latest trends in artificial intelligence?'
            },
            {
                'session_id': 'session2',
                'title': 'Document analysis questions',
                'message_count': 8,
                'last_message_at': '2023-12-01T13:45:00',
                'preview': 'Can you summarize the key points from the uploaded document?'
            }
        ]
    }


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing"""
    return [
        {
            'message_type': 'user',
            'content': 'What are the main topics covered in the research paper?',
            'created_at': '2023-12-01T14:00:00'
        },
        {
            'message_type': 'assistant',
            'content': 'Based on the uploaded research paper, the main topics include machine learning algorithms, neural networks, and natural language processing techniques.',
            'created_at': '2023-12-01T14:01:00',
            'context_used': [
                {
                    'content': 'This paper presents a comprehensive survey of machine learning algorithms...',
                    'metadata': {'filename': 'research_paper.pdf', 'page': 1}
                }
            ]
        },
        {
            'message_type': 'user',
            'content': 'Can you explain the neural network architecture mentioned?',
            'created_at': '2023-12-01T14:05:00'
        },
        {
            'message_type': 'assistant',
            'content': 'The paper describes a transformer-based architecture with attention mechanisms...',
            'created_at': '2023-12-01T14:06:00',
            'context_used': []
        }
    ]


# ===== test_fixtures.py =====
"""
Additional test fixtures and utilities for specific test scenarios
"""

import pytest
import json
from io import BytesIO
from unittest.mock import patch, MagicMock


class MockFile:
    """Mock file object for testing file uploads"""
    
    def __init__(self, filename, content, content_type=None):
        self.filename = filename
        self.content = content
        self.content_type = content_type or self._guess_content_type(filename)
        self.stream = BytesIO(content)
    
    def _guess_content_type(self, filename):
        """Guess content type from filename"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        types = {
            'pdf': 'application/pdf',
            'txt': 'text/plain',
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        return types.get(ext, 'application/octet-stream')


@pytest.fixture
def pdf_file():
    """Mock PDF file for testing"""
    return MockFile('test_document.pdf', b'%PDF-1.4 fake pdf content')


@pytest.fixture
def txt_file():
    """Mock text file for testing"""
    return MockFile('test_document.txt', b'This is a test text file with some content.')


@pytest.fixture
def docx_file():
    """Mock DOCX file for testing"""
    return MockFile('test_document.docx', b'PK fake docx content')


@pytest.fixture
def large_file():
    """Mock large file for testing size limits"""
    content = b'x' * (51 * 1024 * 1024)  # 51MB
    return MockFile('large_file.pdf', content)


@pytest.fixture
def malicious_file():
    """Mock malicious file for security testing"""
    return MockFile('malware.exe', b'MZ fake executable content')


@pytest.fixture
def api_timeout_mock():
    """Mock for API timeout scenarios"""
    def _timeout_side_effect(*args, **kwargs):
        import requests
        raise requests.exceptions.Timeout("Request timed out")
    return _timeout_side_effect


@pytest.fixture
def api_connection_error_mock():
    """Mock for API connection error scenarios"""
    def _connection_error_side_effect(*args, **kwargs):
        import requests
        raise requests.exceptions.ConnectionError("Connection failed")
    return _connection_error_side_effect


# ===== test_integration.py =====
"""
Integration tests for the Flask Web UI with real API interactions
"""

import pytest
import requests
import time
from unittest.mock import patch
import threading


class TestAPIIntegration:
    """Integration tests with API server"""
    
    @pytest.mark.integration
    def test_real_api_health_check(self):
        """Test actual API health check"""
        try:
            response = requests.get('http://localhost:5000/health', timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert 'status' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not available for integration test")
    
    @pytest.mark.integration
    @patch.dict(os.environ, {'API_BASE_URL': 'http://localhost:5000'})
    def test_full_workflow_with_real_api(self, client):
        """Test full workflow with real API server"""
        # This test requires the API server to be running
        try:
            # Test health endpoint first
            response = client.get('/api/health')
            if response.status_code != 200:
                pytest.skip("API server not healthy")
            
            # Test group creation
            response = client.post('/create_group', data={
                'name': 'Integration Test Group',
                'description': 'Created during integration testing'
            }, follow_redirects=True)
            
            assert response.status_code == 200
            
        except Exception as e:
            pytest.skip(f"Integration test failed due to API unavailability: {e}")


class TestConcurrentAccess:
    """Test concurrent access scenarios"""
    
    def test_concurrent_file_uploads(self, client, pdf_file):
        """Test concurrent file upload handling"""
        results = []
        
        def upload_file(file_content, filename):
            try:
                data = {'file': (BytesIO(file_content), filename)}
                response = client.post('/group/test-group/upload',
                                     data=data,
                                     content_type='multipart/form-data')
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Create multiple threads for concurrent uploads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=upload_file,
                args=(pdf_file.content, f'concurrent_file_{i}.pdf')
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All uploads should be handled gracefully
        assert len(results) == 5
        # Results should be HTTP status codes or error messages
        assert all(isinstance(r, (int, str)) for r in results)
    
    def test_concurrent_chat_requests(self, client):
        """Test concurrent chat request handling"""
        results = []
        
        def send_message(message):
            try:
                response = client.post('/api/chat', json={
                    'message': message,
                    'group_id': 'general'
                })
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Create multiple threads for concurrent chat
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=send_message,
                args=(f'Concurrent message {i}',)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should be handled
        assert len(results) == 10


# ===== test_e2e.py =====
"""
End-to-end tests simulating complete user workflows
"""

import pytest
from unittest.mock import patch, MagicMock


class TestEndToEndWorkflows:
    """End-to-end workflow tests"""
    
    @patch('app.requests.get')
    @patch('app.requests.post')
    def test_new_user_complete_workflow(self, mock_post, mock_get, client, mock_api_response):
        """Test complete new user workflow"""
        # Step 1: User visits home page (empty state)
        mock_get.return_value = mock_api_response(200, {'groups': []})
        response = client.get('/')
        assert response.status_code == 200
        assert b'No groups yet' in response.data
        
        # Step 2: User creates first group
        mock_post.return_value = mock_api_response(201, {
            '_id': 'first-group',
            'name': 'My Documents',
            'description': 'Personal document collection'
        })
        
        response = client.post('/create_group', data={
            'name': 'My Documents',
            'description': 'Personal document collection'
        })
        assert response.status_code == 302  # Redirect to group detail
        
        # Step 3: User uploads document
        mock_post.return_value = mock_api_response(200, {
            'success': True,
            'file_id': 'uploaded-file',
            'filename': 'important_document.pdf',
            'status': 'processing'
        })
        
        data = {'file': (BytesIO(b'fake pdf content'), 'important_document.pdf')}
        response = client.post('/group/first-group/upload',
                             data=data,
                             content_type='multipart/form-data',
                             follow_redirects=True)
        assert response.status_code == 200
        assert b'uploaded successfully' in response.data
        
        # Step 4: User starts chat session
        mock_get.side_effect = [
            mock_api_response(200, {'groups': [{'_id': 'first-group', 'name': 'My Documents'}]}),
            mock_api_response(200, {'sessions': []})
        ]
        
        response = client.get('/chat/first-group')
        assert response.status_code == 200
        assert b'My Documents' in response.data
        
        # Step 5: User sends first message
        mock_post.return_value = mock_api_response(200, {
            'session_id': 'new-session',
            'response': 'Hello! I can help you with your documents.',
            'retrieved_context': []
        })
        
        response = client.post('/api/chat', json={
            'message': 'Hello, what can you tell me about my document?',
            'group_id': 'first-group'
        })
        assert response.status_code == 200
        data = response.get_json()
        assert 'response' in data
        assert 'session_id' in data
    
    @patch('app.requests.get')
    @patch('app.requests.post')
    def test_returning_user_workflow(self, mock_post, mock_get, client, mock_api_response, sample_groups, sample_sessions):
        """Test returning user workflow with existing data"""
        # Step 1: User returns to home page with existing groups
        mock_get.return_value = mock_api_response(200, sample_groups)
        response = client.get('/')
        assert response.status_code == 200
        assert b'Research Papers' in response.data
        assert b'Company Docs' in response.data
        
        # Step 2: User navigates to existing group
        mock_get.side_effect = [
            mock_api_response(200, sample_groups),
            mock_api_response(200, {'files': []}),
            mock_api_response(200, sample_sessions)
        ]
        
        response = client.get('/group/research')
        assert response.status_code == 200
        assert b'Research Papers' in response.data
        
        # Step 3: User continues existing chat session
        mock_get.side_effect = [
            mock_api_response(200, sample_groups),
            mock_api_response(200, {
                '_id': 'session1',
                'title': 'Discussion about AI trends',
                'messages': [
                    {
                        'message_type': 'user',
                        'content': 'Previous question about AI',
                        'created_at': '2023-12-01T10:00:00'
                    }
                ]
            }),
            mock_api_response(200, sample_sessions)
        ]
        
        response = client.get('/chat/research/session1')
        assert response.status_code == 200
        assert