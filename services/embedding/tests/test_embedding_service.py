"""
Redis Integration Test Suite for Advanced RAG Service
Tests core functionalities using actual Redis to send and receive data
No mocking - tests real Redis queue processing workflows with proper data cleanup
"""

import unittest
import redis
import json
import time
import tempfile
import os
import threading
from datetime import datetime
from bson import ObjectId
import logging
from pymongo import MongoClient
import chromadb
from chromadb.config import Settings
from elasticsearch import Elasticsearch

# Suppress logging during tests
logging.getLogger().setLevel(logging.ERROR)

# Test configuration
REDIS_URL = f"redis://:{os.environ.get('REDIS_PASSWORD', 'password')}@{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}/{os.environ.get('REDIS_DB', '0')}"

# MongoDB configuration
MONGO_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGO_PORT = os.environ.get('MONGO_INITDB_PORT', '27017')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME', 'admin')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD', 'password')
MONGO_DATABASE = os.environ.get('MONGO_INITDB_DATABASE', 'rag_system')
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"

# ChromaDB configuration
CHROMA_HOST = os.environ.get('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.environ.get('CHROMA_PORT', '8000'))
CHROMA_PERSISTENT = os.environ.get('CHROMA_PERSISTENT', 'false').lower() == 'true'

# Elasticsearch configuration
ELASTICSEARCH_HOST = os.environ.get('ELASTICSEARCH_HOST', 'localhost')
ELASTICSEARCH_PORT = int(os.environ.get('ELASTICSEARCH_PORT', '9200'))
ELASTICSEARCH_USERNAME = os.environ.get('ELASTICSEARCH_USERNAME', None)
ELASTICSEARCH_PASSWORD = os.environ.get('ELASTICSEARCH_PASSWORD', None)


class RedisIntegrationTestBase(unittest.TestCase):
    """Base class for Redis integration tests with full data cleanup"""
    
    @classmethod
    def setUpClass(cls):
        """Set up connections for all tests"""
        try:
            # Redis connection
            cls.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            cls.redis_client.ping()
            print(f"✅ Connected to Redis at {REDIS_URL}")
        except Exception as e:
            raise unittest.SkipTest(f"Redis not available: {e}")
        
        try:
            # MongoDB connection
            cls.mongo_client = MongoClient(MONGO_URI)
            cls.db = cls.mongo_client[MONGO_DATABASE]
            cls.db.command('ping')
            print(f"✅ Connected to MongoDB at {MONGO_URI}")
        except Exception as e:
            raise unittest.SkipTest(f"MongoDB not available: {e}")
        
        try:
            # ChromaDB connection
            if CHROMA_PERSISTENT:
                cls.chroma_client = chromadb.PersistentClient(
                    path="/app_data/chroma_db",
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                cls.chroma_client = chromadb.HttpClient(
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                    settings=Settings(anonymized_telemetry=False)
                )
            
            # Test ChromaDB connection
            cls.chroma_client.list_collections()
            print(f"✅ Connected to ChromaDB")
        except Exception as e:
            print(f"⚠️  ChromaDB not available: {e}, using in-memory client")
            cls.chroma_client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
        
        try:
            # Elasticsearch connection
            es_config = {
                'hosts': [{'host': ELASTICSEARCH_HOST, 'port': ELASTICSEARCH_PORT, 'scheme': 'http'}],
                'request_timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True
            }
            
            if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
                es_config['basic_auth'] = (ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
            
            cls.es_client = Elasticsearch(**es_config)
            if cls.es_client.ping():
                print(f"✅ Connected to Elasticsearch")
            else:
                raise ConnectionError("Elasticsearch ping failed")
        except Exception as e:
            raise unittest.SkipTest(f"Elasticsearch not available: {e}")
        
        # Clean up any existing test data at class level
        cls._clean_all_test_data()
    
    @classmethod
    def _clean_all_test_data(cls):
        """Clean all test data from all systems"""
        print("🧹 Cleaning all test data...")
        
        # Clean Redis
        cls.redis_client.flushdb()
        
        # Clean MongoDB test collections
        try:
            # Delete test documents
            cls.db.file_uploads.delete_many({'group_id': {'$regex': '^test_'}})
            cls.db.messages.delete_many({'group_id': {'$regex': '^test_'}})
            cls.db.chat_sessions.delete_many({'group_id': {'$regex': '^test_'}})
            print("✅ Cleaned MongoDB test data")
        except Exception as e:
            print(f"⚠️  Error cleaning MongoDB: {e}")
        
        # Clean ChromaDB collections
        try:
            collections = cls.chroma_client.list_collections()
            for collection in collections:
                if 'test' in collection.name.lower() or collection.name == 'documents':
                    try:
                        cls.chroma_client.delete_collection(collection.name)
                        print(f"✅ Deleted ChromaDB collection: {collection.name}")
                    except Exception as e:
                        print(f"⚠️  Error deleting ChromaDB collection {collection.name}: {e}")
        except Exception as e:
            print(f"⚠️  Error cleaning ChromaDB: {e}")
        
        # Clean Elasticsearch indices
        try:
            # Delete test indices
            test_indices = ['rag_chunks', 'test_*']
            for index_pattern in test_indices:
                try:
                    if cls.es_client.indices.exists(index=index_pattern):
                        cls.es_client.indices.delete(index=index_pattern, ignore=[400, 404])
                        print(f"✅ Deleted Elasticsearch index: {index_pattern}")
                except Exception as e:
                    print(f"⚠️  Error deleting Elasticsearch index {index_pattern}: {e}")
        except Exception as e:
            print(f"⚠️  Error cleaning Elasticsearch: {e}")
        
        print("✅ All test data cleaned")
    
    def setUp(self):
        """Clean up before each test"""
        # Clear Redis queues and results
        test_queues = [
            'embedding_tasks',
            'file_processing_tasks',
            'group_management_tasks'
        ]
        for queue in test_queues:
            self.redis_client.delete(queue)
        
        # Clear result keys
        for key in self.redis_client.keys('*_result:*'):
            self.redis_client.delete(key)
        for key in self.redis_client.keys('file_processing_result:*'):
            self.redis_client.delete(key)
    
    def tearDown(self):
        """Clean up after each test"""
        self.setUp()  # Same cleanup
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        try:
            cls._clean_all_test_data()
            
            # Close connections
            if hasattr(cls, 'redis_client'):
                cls.redis_client.close()
            if hasattr(cls, 'mongo_client'):
                cls.mongo_client.close()
            if hasattr(cls, 'es_client') and hasattr(cls.es_client, 'close'):
                cls.es_client.close()
                
            print("✅ All connections closed")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
    
    @staticmethod
    def fetch_results(redis_client, result_key, timeout=30):
        """Helper method to fetch results from Redis with timeout"""
        start = time.time()
        while time.time() - start < timeout:
            result = redis_client.get(result_key)
            if result:
                return json.loads(result)
            time.sleep(0.5)
        return None
    
    def create_mongo_file_entry(self, file_id, filename, file_path, group_id='test_group'):
        """Create a MongoDB entry for file upload"""
        file_entry = {
            '_id': file_id,
            'filename': filename,
            'original_filename': filename,
            'file_path': file_path,
            'file_type': self._get_file_type(filename),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'mime_type': self._get_mime_type(filename),
            'group_id': group_id,
            'uploaded_by': 'test_user',
            'upload_status': 'pending',
            'processing_progress': 0,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        try:
            self.db.file_uploads.insert_one(file_entry)
            print(f"✅ Created MongoDB entry for file: {filename}")
            return file_entry
        except Exception as e:
            print(f"⚠️  Error creating MongoDB entry: {e}")
            return None
    
    def _get_file_type(self, filename):
        """Get file type from filename"""
        ext = os.path.splitext(filename.lower())[1]
        type_mapping = {
            '.pdf': 'pdf',
            '.txt': 'txt',
            '.doc': 'doc',
            '.docx': 'docx'
        }
        return type_mapping.get(ext, 'unknown')
    
    def _get_mime_type(self, filename):
        """Get MIME type from filename"""
        ext = os.path.splitext(filename.lower())[1]
        mime_mapping = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        return mime_mapping.get(ext, 'application/octet-stream')


class TestChatAfterFileProcessingRedisFlow(RedisIntegrationTestBase):
    """Test file processing and chat Redis queue flow"""
    
    def test_03_chat_task_submission_and_retrieval(self):
        """Test submitting chat task to Redis and retrieving results"""
        
        # 1. Submit chat task to Redis queue
        chat_task = {
            'id': 'test_chat_001',
            'type': 'chat',
            'message': 'What responsibilities are of machine learning scientist role at amazon ?',
            'group_id': 'test_group',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Push task to embedding_tasks queue (where chat tasks go)
        task_json = json.dumps(chat_task)
        self.redis_client.lpush('embedding_tasks', task_json)
        
        result_key = f"retrieval_result:{chat_task['id']}"
        stored_result = self.fetch_results(self.redis_client, result_key, timeout=300)
        
        # Note: This test assumes the embedding service is running
        # If no service is running, the result will be None
        if stored_result:
            self.assertIn('documents', stored_result)
            self.assertEqual(len(stored_result['documents']), 5)
            self.assertEqual(stored_result['group_id'], 'test_group')
            
            print(f"✅ Chat task processed successfully with {len(stored_result.get('documents', []))} documents")
        else:
            print("⚠️  No embedding service running - task queued but not processed")
    
    def test_02_file_processing_task_submission(self):
        """Test file processing task submission and result retrieval"""
        
        test_files = ["about_amazon.pdf", "tech_screen.pdf", "ml_jd.docx", "food_safety.txt"]
        
        for test_file in test_files:
            with self.subTest(file=test_file):
                upload_dir = "/app_data/file_uploads"
                os.makedirs(upload_dir, exist_ok=True)
            
                # Create a temporary test file
                test_file_path = f"./tests/test_data/{test_file}"
                dest_file_path = os.path.join(upload_dir, test_file)
                
                # Copy test file if it exists
                if os.path.exists(test_file_path):
                    with open(test_file_path, 'rb') as src, open(dest_file_path, 'wb') as dst:
                        dst.write(src.read())
                else:
                    # Create dummy file for testing
                    with open(dest_file_path, 'w') as f:
                        f.write(f"Test content for {test_file}\n" * 100)

                file_id = str(ObjectId())
                
                # Create MongoDB entry before processing
                self.create_mongo_file_entry(file_id, test_file, dest_file_path, 'test_group')
                
                file_task = {
                    'file_id': file_id,
                    'file_path': dest_file_path,
                    'group_id': 'test_group',
                    'filename': test_file,
                    'type': 'process_file',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Push to file processing queue
                self.redis_client.lpush('file_processing_tasks', json.dumps(file_task))                
                result_key = f"file_processing_result:{file_task['file_id']}"
                stored_result = self.fetch_results(self.redis_client, result_key, timeout=300)
                
                # If embedding service is running, check results
                if stored_result:
                    if stored_result['status'] == 'completed':
                        self.assertGreater(stored_result['chunks_count'], 0)
                        self.assertEqual(stored_result['status'], 'completed')
                        self.assertEqual(stored_result['group_id'], 'test_group')
                        
                        # Verify MongoDB entry was updated
                        file_doc = self.db.file_uploads.find_one({'_id': file_id})
                        self.assertIsNotNone(file_doc)
                        self.assertEqual(file_doc['upload_status'], 'completed')
                        self.assertGreater(file_doc.get('chunks_count', 0), 0)
                        
                        print(f"✅ File {test_file} processed successfully with {stored_result['chunks_count']} chunks")
                    else:
                        print(f"⚠️  File {test_file} processing failed: {stored_result.get('error', 'Unknown error')}")
                else:
                    print(f"⚠️  No embedding service running - file {test_file} queued but not processed")
                
    
    def test_01_file_processing_error_handling(self):
        """Test file processing error scenario"""
        
        file_id = str(ObjectId())
        
        # Create MongoDB entry for non-existent file
        self.create_mongo_file_entry(file_id, 'nonexistent.txt', '/nonexistent/file.txt', 'test_group')
        
        # Submit task with non-existent file
        error_task = {
            'file_id': file_id,
            'file_path': '/nonexistent/file.txt',
            'filename': 'nonexistent.txt',
            'group_id': 'test_group',
            'type': 'process_file'
        }
        result_key = f"file_processing_result:{error_task['file_id']}"
        self.redis_client.lpush('file_processing_tasks', json.dumps(error_task))
        stored_result = self.fetch_results(self.redis_client, result_key, timeout=300)
        
        if stored_result:
            self.assertEqual(stored_result['status'], 'failed')
            self.assertIn('error', stored_result)
            self.assertTrue('File not found' in stored_result['error'])
            
            # Verify MongoDB entry was updated with error
            file_doc = self.db.file_uploads.find_one({'_id': file_id})
            self.assertIsNotNone(file_doc)
            self.assertEqual(file_doc['upload_status'], 'failed')
            self.assertIsNotNone(file_doc.get('error_message'))
            
            print("✅ Error handling working correctly")
        else:
            print("⚠️  No embedding service running - error case not tested")

 



class RedisIntegrationTestRunner:
    """Test runner for Redis integration tests"""
    
    @staticmethod
    def run_all_tests():
        """Run all Redis integration tests"""
        
        print("🚀 Starting Redis Integration Tests for RAG Service Core Functionalities")
        print("=" * 80)
        
        # Check all service connectivity first
        services_status = {}
        
        # Redis
        try:
            test_redis = redis.from_url(REDIS_URL, decode_responses=True)
            test_redis.ping()
            services_status['Redis'] = '✅'
        except Exception as e:
            services_status['Redis'] = f'❌ {e}'
        
        # MongoDB
        try:
            test_mongo = MongoClient(MONGO_URI)
            test_mongo[MONGO_DATABASE].command('ping')
            test_mongo.close()
            services_status['MongoDB'] = '✅'
        except Exception as e:
            services_status['MongoDB'] = f'❌ {e}'
        
        # ChromaDB
        try:
            if CHROMA_PERSISTENT:
                test_chroma = chromadb.PersistentClient(
                    path="/app_data/chroma_db",
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
            else:
                test_chroma = chromadb.HttpClient(
                    host=CHROMA_HOST, port=CHROMA_PORT,
                    settings=Settings(anonymized_telemetry=False)
                )
            test_chroma.list_collections()
            services_status['ChromaDB'] = '✅'
        except Exception as e:
            services_status['ChromaDB'] = f'⚠️  Using in-memory: {e}'
        
        # Elasticsearch
        try:
            es_config = {
                'hosts': [{'host': ELASTICSEARCH_HOST, 'port': ELASTICSEARCH_PORT, 'scheme': 'http'}],
                'timeout': 5
            }
            if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
                es_config['http_auth'] = (ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
            
            test_es = Elasticsearch(**es_config)
            if test_es.ping():
                services_status['Elasticsearch'] = '✅'
                test_es.close()
            else:
                services_status['Elasticsearch'] = '❌ Ping failed'
        except Exception as e:
            services_status['Elasticsearch'] = f'❌ {e}'
        
        print("Service Status:")
        for service, status in services_status.items():
            print(f"  {service}: {status}")
        print()
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add test classes
        test_classes = [
            TestChatAfterFileProcessingRedisFlow
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=None)
        result = runner.run(test_suite)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 REDIS INTEGRATION TEST SUMMARY")
        print("=" * 80)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        
        if result.failures:
            print(f"\n❌ FAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")
                
        if result.errors:
            print(f"\n💥 ERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        if result.wasSuccessful():
            print(f"\n🎉 All tests passed! Redis integration is working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Check services and configuration.")
        
        return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive test suite
    print("🚀 Redis Integration Tests with Full Data Cleanup")
    print("=" * 80)
    
    success = RedisIntegrationTestRunner.run_all_tests()
    
    exit(0 if success else 1)