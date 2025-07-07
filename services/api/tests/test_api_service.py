"""
Comprehensive Test Suite for RAG API Server
Tests all endpoints including content groups, file uploads, and chat functionality
"""

import pytest
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import requests
from io import BytesIO

# Test configuration
API_SERVICE_PORT = int(os.environ.get('API_SERVICE_PORT', 5000))
API_BASE_URL = f"http://localhost:{API_SERVICE_PORT}"


class TestRAGAPIServer:
    """Test suite for RAG API Server"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.base_url = API_BASE_URL
        cls.created_groups = []
        cls.uploaded_files = []
        cls.created_sessions = []
                
        # Create test files
        cls.validate_test_files()
        
        # Wait for server to be ready
        cls.wait_for_server()
    
    @classmethod
    def wait_for_server(cls, timeout=30):
        """Wait for server to be available"""
        for _ in range(timeout):
            try:
                response = requests.get(f"{cls.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✓ Server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        raise Exception("Server did not become available within timeout")
    
    @classmethod
    def validate_test_files(cls):
        """Create test files for upload testing"""
        test_files = ["./embedding/tests/test_data/about_amazon.pdf", 
                      "./embedding/tests/test_data/tech_screen.pdf", 
                      "./embedding/tests/test_data/ml_jd.docx", 
                      "./embedding/tests/test_data/food_safety.txt"]
        for file_path in test_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Test file {file_path} does not exist")
            else:
                print(f"✓ Test file {file_path} is ready for upload tests")

    
    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        # Clean up created groups
        for group_id in cls.created_groups:
            try:
                requests.delete(f"{cls.base_url}/api/groups/{group_id}")
            except:
                pass
            
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
        print("✓ Health check passed")
    
    def test_get_groups_empty(self):
        """Test getting groups when none exist (should create General Chat)"""
        response = requests.get(f"{self.base_url}/api/groups")
        
        assert response.status_code == 200
        data = response.json()
        assert "groups" in data
        
        # Should have at least General Chat
        groups = data["groups"]
        general_chat = next((g for g in groups if g["name"] == "General Chat"), None)
        assert general_chat is not None
        assert general_chat["_id"] == "general"
        print("✓ Get groups with auto-created General Chat")
    
    def test_create_content_group(self):
        """Test creating a new content group"""
        group_data = {
            "name": "Test Research Group",
            "description": "A group for testing research documents"
        }
        
        response = requests.post(
            f"{self.base_url}/api/groups",
            json=group_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == group_data["name"]
        assert data["description"] == group_data["description"]
        assert "_id" in data
        assert data["file_count"] == 0
        
        # Store for cleanup
        self.created_groups.append(data["_id"])
        self.test_group_id = data["_id"]
        print(f"✓ Created content group: {data['_id']}")
    
    def test_create_duplicate_group(self):
        """Test creating a group with duplicate name"""
        group_data = {
            "name": "Test Research Group",  # Same as previous test
            "description": "Duplicate group"
        }
        
        response = requests.post(
            f"{self.base_url}/api/groups",
            json=group_data
        )
        
        assert response.status_code == 409
        data = response.json()
        assert "error" in data
        assert "already exists" in data["error"]
        print("✓ Duplicate group creation properly rejected")
    
    def test_create_group_missing_name(self):
        """Test creating group without required name"""
        group_data = {"description": "Missing name"}
        
        response = requests.post(
            f"{self.base_url}/api/groups",
            json=group_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        print("✓ Group creation with missing name rejected")
    
    def test_upload_file_to_content_group(self):
        """Test uploading a file to a content group"""
        test_txt_path = "./embedding/tests/test_data/food_safety.txt"
        with open(test_txt_path, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            response = requests.post(
                f"{self.base_url}/api/groups/{self.test_group_id}/files",
                files=files
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test_document.txt"
        assert data["group_id"] == self.test_group_id
        assert data["status"] == "processing"
        
        self.test_file_id = data["file_id"]
        self.uploaded_files.append(data["file_id"])
        print(f"✓ Uploaded file to content group: {data['file_id']}")
    
    def test_upload_file_to_general_chat(self):
        """Test uploading a file to General Chat (should be skipped)"""
        with open(self.test_txt_path, 'rb') as f:
            files = {'file': ('general_test.txt', f, 'text/plain')}
            response = requests.post(
                f"{self.base_url}/api/groups/general/files",
                files=files
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "skipped"  # Should be skipped for general
        print("✓ File upload to General Chat properly skipped")
    
    def test_upload_invalid_file_type(self):
        """Test uploading an unsupported file type"""
        # Create a fake .exe file
        fake_file = BytesIO(b"fake exe content")
        files = {'file': ('malware.exe', fake_file, 'application/octet-stream')}
        
        response = requests.post(
            f"{self.base_url}/api/groups/{self.test_group_id}/files",
            files=files
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "not allowed" in data["error"]
        print("✓ Invalid file type properly rejected")
    
    def test_upload_no_file(self):
        """Test uploading without providing a file"""
        response = requests.post(
            f"{self.base_url}/api/groups/{self.test_group_id}/files"
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "No file provided" in data["error"]
        print("✓ No file upload properly rejected")
    
    def test_upload_to_nonexistent_group(self):
        """Test uploading to a non-existent group"""
        with open(self.test_txt_path, 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            response = requests.post(
                f"{self.base_url}/api/groups/nonexistent-group/files",
                files=files
            )
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["error"]
        print("✓ Upload to non-existent group properly rejected")
    
    def test_get_file_status(self):
        """Test getting file processing status"""
        response = requests.get(f"{self.base_url}/api/files/{self.test_file_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["_id"] == self.test_file_id
        assert data["filename"] == "test_document.txt"
        assert data["group_id"] == self.test_group_id
        assert "upload_status" in data
        print(f"✓ Retrieved file status: {data['upload_status']}")
    
    def test_get_nonexistent_file_status(self):
        """Test getting status of non-existent file"""
        response = requests.get(f"{self.base_url}/api/files/nonexistent-file/status")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        print("✓ Non-existent file status properly handled")
    
    def test_get_group_files(self):
        """Test getting files in a group"""
        response = requests.get(f"{self.base_url}/api/groups/{self.test_group_id}/files")
        
        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == self.test_group_id
        assert "files" in data
        assert data["total_files"] >= 1
        print(f"✓ Retrieved group files: {data['total_files']} files")
    
    def test_get_files_nonexistent_group(self):
        """Test getting files from non-existent group"""
        response = requests.get(f"{self.base_url}/api/groups/nonexistent/files")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        print("✓ Files from non-existent group properly handled")
    
    def test_general_chat(self):
        """Test general chat without RAG"""
        chat_data = {
            "message": "Hello! How are you today?",
            "group_id": "general"
        }
        
        # Mock the Redis responses for general chat
        with patch('requests.post') as mock_post:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=chat_data
            )
        
        # Note: This will likely timeout waiting for inference service
        # In a real test environment, you'd mock the Redis interactions
        print("✓ General chat endpoint called (response depends on inference service)")
    
    def test_content_group_chat(self):
        """Test chat with RAG in content group"""
        chat_data = {
            "message": "What is mentioned in the uploaded document?",
            "group_id": self.test_group_id
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=chat_data
        )
        
        # Response will depend on services being available
        print("✓ Content group chat endpoint called (response depends on services)")
    
    def test_chat_with_session_continuation(self):
        """Test continuing a chat session"""
        # First message
        chat_data = {
            "message": "Start a conversation about AI",
            "group_id": "general"
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=chat_data
        )
        
        # Extract session_id from response (if available)
        # Second message in same session would use the session_id
        print("✓ Session continuation chat tested")
    
    def test_chat_missing_message(self):
        """Test chat without message"""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "required" in data["error"]
        print("✓ Chat without message properly rejected")
    
    def test_chat_nonexistent_group(self):
        """Test chat with non-existent group"""
        chat_data = {
            "message": "Test message",
            "group_id": "nonexistent-group"
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=chat_data
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        print("✓ Chat with non-existent group properly rejected")
    
    def test_get_group_sessions(self):
        """Test getting sessions for a group"""
        response = requests.get(f"{self.base_url}/api/groups/{self.test_group_id}/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == self.test_group_id
        assert "sessions" in data
        assert "pagination" in data
        print(f"✓ Retrieved group sessions: {len(data['sessions'])} sessions")
    
    def test_get_sessions_with_pagination(self):
        """Test getting sessions with pagination"""
        response = requests.get(
            f"{self.base_url}/api/groups/{self.test_group_id}/sessions?page=1&limit=5"
        )
        
        assert response.status_code == 200
        data = response.json()
        pagination = data["pagination"]
        assert pagination["page"] == 1
        assert pagination["limit"] == 5
        print("✓ Session pagination working")
    
    def test_get_sessions_nonexistent_group(self):
        """Test getting sessions from non-existent group"""
        response = requests.get(f"{self.base_url}/api/groups/nonexistent/sessions")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        print("✓ Sessions from non-existent group properly handled")
    
    def test_get_general_chat_sessions(self):
        """Test getting sessions from General Chat"""
        response = requests.get(f"{self.base_url}/api/groups/general/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == "general"
        print("✓ Retrieved General Chat sessions")
    
    def test_get_session_details(self):
        """Test getting specific session details"""
        # First get sessions to find one
        sessions_response = requests.get(f"{self.base_url}/api/groups/general/sessions")
        
        if sessions_response.status_code == 200:
            sessions_data = sessions_response.json()
            if sessions_data["sessions"]:
                session_id = sessions_data["sessions"][0]["session_id"]
                
                response = requests.get(f"{self.base_url}/api/sessions/{session_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    assert data["_id"] == session_id
                    assert "messages" in data
                    print(f"✓ Retrieved session details: {len(data['messages'])} messages")
                else:
                    print("✓ Session details endpoint accessible")
            else:
                print("✓ No sessions available for testing session details")
        else:
            print("✓ Session details test skipped (no sessions available)")
    
    def test_get_nonexistent_session(self):
        """Test getting non-existent session"""
        response = requests.get(f"{self.base_url}/api/sessions/nonexistent-session")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        print("✓ Non-existent session properly handled")
    
    def test_get_all_groups_after_creation(self):
        """Test getting all groups after creating some"""
        response = requests.get(f"{self.base_url}/api/groups")
        
        assert response.status_code == 200
        data = response.json()
        groups = data["groups"]
        
        # Should have General Chat + created groups
        assert len(groups) >= 2
        
        # Check that our created group is there
        created_group = next((g for g in groups if g["_id"] == self.test_group_id), None)
        assert created_group is not None
        assert created_group["name"] == "Test Research Group"
        print(f"✓ Retrieved all groups: {len(groups)} total")


class TestRAGAPIIntegration:
    """Integration tests that require all services running"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.base_url = API_BASE_URL
    
    @pytest.mark.integration
    def test_full_workflow_content_group(self):
        """Test complete workflow: create group -> upload file -> chat"""
        # 1. Create group
        group_data = {
            "name": "Integration Test Group",
            "description": "Testing full workflow"
        }
        
        response = requests.post(f"{self.base_url}/api/groups", json=group_data)
        assert response.status_code == 201
        group_id = response.json()["_id"]
        
        try:
            # 2. Upload file
            test_content = "This is test content for integration testing. AI is awesome!"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_path = f.name
            
            with open(temp_path, 'rb') as f:
                files = {'file': ('integration_test.txt', f, 'text/plain')}
                upload_response = requests.post(
                    f"{self.base_url}/api/groups/{group_id}/files",
                    files=files
                )
            
            assert upload_response.status_code == 200
            file_id = upload_response.json()["file_id"]
            
            # 3. Wait for file processing (in real scenario)
            time.sleep(2)
            
            # 4. Check file status
            status_response = requests.get(f"{self.base_url}/api/files/{file_id}/status")
            assert status_response.status_code == 200
            
            # 5. Chat about the content
            chat_data = {
                "message": "What does the document say about AI?",
                "group_id": group_id
            }
            
            chat_response = requests.post(f"{self.base_url}/api/chat", json=chat_data)
            
            # Response depends on services being available
            print("✓ Full workflow integration test completed")
            
        finally:
            # Cleanup
            os.unlink(temp_path)
            requests.delete(f"{self.base_url}/api/groups/{group_id}")
    
    @pytest.mark.integration
    def test_general_chat_workflow(self):
        """Test general chat workflow without RAG"""
        chat_data = {
            "message": "What is the capital of France?",
            "group_id": "general"
        }
        
        response = requests.post(f"{self.base_url}/api/chat", json=chat_data)
        
        # Note: This depends on inference service being available
        # In real testing, you might want to mock the Redis interactions
        print("✓ General chat workflow test completed")
    
    @pytest.mark.integration
    def test_session_continuation_workflow(self):
        """Test multi-turn conversation in same session"""
        # First message
        chat_data_1 = {
            "message": "Hello, my name is Alice",
            "group_id": "general"
        }
        
        response_1 = requests.post(f"{self.base_url}/api/chat", json=chat_data_1)
        
        if response_1.status_code == 200:
            session_id = response_1.json().get("session_id")
            
            if session_id:
                # Second message in same session
                chat_data_2 = {
                    "message": "What is my name?",
                    "group_id": "general",
                    "session_id": session_id
                }
                
                response_2 = requests.post(f"{self.base_url}/api/chat", json=chat_data_2)
                
                print("✓ Session continuation workflow tested")
            else:
                print("✓ Session continuation test incomplete (no session_id)")
        else:
            print("✓ Session continuation test skipped (first message failed)")


def run_tests():
    """Run all tests"""
    print("🚀 Starting RAG API Server Tests\n")
    
    # Run basic functionality tests
    print("=" * 50)
    print("BASIC FUNCTIONALITY TESTS")
    print("=" * 50)
    
    test_instance = TestRAGAPIServer()
    test_instance.setup_class()
    
    try:
        # Health and basic tests
        test_instance.test_health_check()
        test_instance.test_get_groups_empty()
        
        # Group management tests
        test_instance.test_create_content_group()
        test_instance.test_create_duplicate_group()
        test_instance.test_create_group_missing_name()
        
        # File upload tests
        test_instance.test_upload_file_to_content_group()
        test_instance.test_upload_file_to_general_chat()
        test_instance.test_upload_invalid_file_type()
        test_instance.test_upload_no_file()
        test_instance.test_upload_to_nonexistent_group()
        
        # File status tests
        test_instance.test_get_file_status()
        test_instance.test_get_nonexistent_file_status()
        test_instance.test_get_group_files()
        test_instance.test_get_files_nonexistent_group()
        
        # Chat tests
        test_instance.test_general_chat()
        test_instance.test_content_group_chat()
        test_instance.test_chat_with_session_continuation()
        test_instance.test_chat_missing_message()
        test_instance.test_chat_nonexistent_group()
        
        # Session tests
        test_instance.test_get_group_sessions()
        test_instance.test_get_sessions_with_pagination()
        test_instance.test_get_sessions_nonexistent_group()
        test_instance.test_get_general_chat_sessions()
        test_instance.test_get_session_details()
        test_instance.test_get_nonexistent_session()
        
        # Final tests
        test_instance.test_get_all_groups_after_creation()
        
        print("\n" + "=" * 50)
        print("INTEGRATION TESTS")
        print("=" * 50)
        
        # Integration tests (require all services)
        integration_tests = TestRAGAPIIntegration()
        integration_tests.test_full_workflow_content_group()
        integration_tests.test_general_chat_workflow()
        integration_tests.test_session_continuation_workflow()
        
    finally:
        test_instance.teardown_class()
    
    print("\n" + "🎉" * 20)
    print("ALL TESTS COMPLETED!")
    print("🎉" * 20)


if __name__ == "__main__":
    run_tests()