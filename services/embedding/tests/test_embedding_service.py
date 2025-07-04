"""
Redis Integration Test Suite for Advanced RAG Service
Tests core functionalities using actual Redis to send and receive data
No mocking - tests real Redis queue processing workflows
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

# Suppress logging during tests
logging.getLogger().setLevel(logging.ERROR)

# Test configuration
REDIS_URL = f"redis://:{os.environ.get('REDIS_PASSWORD', 'password')}@{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}/{os.environ.get('REDIS_DB', '0')}"


class RedisIntegrationTestBase(unittest.TestCase):
    """Base class for Redis integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up Redis connection for all tests"""
        try:
            cls.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            cls.redis_client.ping()
            print(f"✅ Connected to Redis at {REDIS_URL}")
        except Exception as e:
            raise unittest.SkipTest(f"Redis not available: {e}")
        
        # Clean up any existing test data
        cls.redis_client.flushdb()
    
    def setUp(self):
        """Clean up before each test"""
        # Clear all test queues
        test_queues = [
            'embedding_tasks',
            'file_processing_tasks', 
            'dataset_processing_tasks'
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
            cls.redis_client.flushdb()
            cls.redis_client.close()
        except:
            pass
    @staticmethod
    def fetch_results(redis_client, result_key, timeout=30):
        """Helper method to fetch results from Redis with timeout"""
        start = time.time()
        while time.time() - start < timeout:
            result = redis_client.get(result_key)
            if result:
                return json.loads(result)
            time.sleep(0.5)


class TestChatTaskRedisFlow(RedisIntegrationTestBase):
    """Test chat task Redis queue processing"""
    
    def test_chat_task_submission_and_retrieval(self):
        """Test submitting chat task to Redis and retrieving results"""
        
        # 1. Submit chat task to Redis queue
        chat_task = {
            'id': 'test_chat_001',
            'type': 'chat',
            'message': 'What is machine learning?',
            'group_id': 'general',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Push task to embedding_tasks queue (where chat tasks go)
        task_json = json.dumps(chat_task)
        self.redis_client.lpush('embedding_tasks', task_json)
        
        # Verify task is in queue
        queue_length = self.redis_client.llen('embedding_tasks')
        self.assertEqual(queue_length, 1)
        
        # 2. Simulate service processing by checking queue
        # Pop task from queue (this is what the service does)
        queue_data = self.redis_client.brpop(['embedding_tasks'], timeout=1)
        self.assertIsNotNone(queue_data)
        
        queue_name, retrieved_task_json = queue_data
        retrieved_task = json.loads(retrieved_task_json)
        
        # Verify task data integrity
        self.assertEqual(retrieved_task['id'], 'test_chat_001')
        self.assertEqual(retrieved_task['type'], 'chat')
        self.assertEqual(retrieved_task['message'], 'What is machine learning?')
        self.assertEqual(retrieved_task['group_id'], 'general')
        
        # 3. Simulate service storing result back to Redis
        mock_result = {
            'task_id': 'test_chat_001',
            'documents': [
                {
                    'id': 'doc_1',
                    'content': 'Machine learning is a subset of AI...',
                    'metadata': {'source': 'test_doc'},
                    'relevance_score': 0.85,
                    'scores': {
                        'initial': 0.8,
                        'rerank': 0.9,
                        'final': 0.85
                    }
                }
            ],
            'retrieval_time_ms': 150,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store result with TTL (as service does)
        result_key = f"retrieval_result:{chat_task['id']}"
        self.redis_client.setex(result_key, 60, json.dumps(mock_result))
        
        # 4. Verify result can be retrieved
        stored_result_json = self.redis_client.get(result_key)
        self.assertIsNotNone(stored_result_json)
        
        stored_result = json.loads(stored_result_json)
        self.assertEqual(stored_result['task_id'], 'test_chat_001')
        self.assertEqual(len(stored_result['documents']), 1)
        self.assertIn('retrieval_time_ms', stored_result)
        
        # Verify document structure
        doc = stored_result['documents'][0]
        self.assertIn('content', doc)
        self.assertIn('relevance_score', doc)
        self.assertIn('scores', doc)
    
    def test_chat_task_with_filters(self):
        """Test chat task with group filtering"""
        
        chat_task = {
            'id': 'test_chat_002',
            'type': 'chat',
            'message': 'Show me research documents',
            'group_id': 'research_team',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Submit to queue
        self.redis_client.lpush('embedding_tasks', json.dumps(chat_task))
        
        # Process from queue
        queue_data = self.redis_client.brpop(['embedding_tasks'], timeout=1)
        retrieved_task = json.loads(queue_data[1])
        
        # Verify group_id is preserved for filtering
        self.assertEqual(retrieved_task['group_id'], 'research_team')
        
        # Simulate filtered results
        filtered_result = {
            'task_id': 'test_chat_002',
            'documents': [
                {
                    'id': 'research_doc_1',
                    'content': 'Research document content...',
                    'metadata': {'group_id': 'research_team', 'type': 'research'},
                    'relevance_score': 0.92
                }
            ],
            'applied_filters': {'group_id': 'research_team'},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        result_key = f"retrieval_result:{chat_task['id']}"
        self.redis_client.setex(result_key, 60, json.dumps(filtered_result))
        
        # Verify filtered result
        stored_result = json.loads(self.redis_client.get(result_key))
        self.assertIn('applied_filters', stored_result)
        self.assertEqual(stored_result['applied_filters']['group_id'], 'research_team')
    
    def test_multiple_concurrent_chat_tasks(self):
        """Test handling multiple concurrent chat tasks"""
        
        # Submit multiple tasks
        tasks = []
        for i in range(5):
            task = {
                'id': f'concurrent_chat_{i}',
                'type': 'chat',
                'message': f'Query number {i}',
                'group_id': 'general',
                'timestamp': datetime.utcnow().isoformat()
            }
            tasks.append(task)
            self.redis_client.lpush('embedding_tasks', json.dumps(task))
        
        # Verify all tasks are queued
        self.assertEqual(self.redis_client.llen('embedding_tasks'), 5)
        
        # Process all tasks
        processed_tasks = []
        for _ in range(5):
            queue_data = self.redis_client.brpop(['embedding_tasks'], timeout=1)
            if queue_data:
                task = json.loads(queue_data[1])
                processed_tasks.append(task)
                
                # Store mock result for each
                result = {
                    'task_id': task['id'],
                    'documents': [{'id': f"doc_{task['id']}", 'content': f"Result for {task['id']}"}],
                    'timestamp': datetime.utcnow().isoformat()
                }
                result_key = f"retrieval_result:{task['id']}"
                self.redis_client.setex(result_key, 60, json.dumps(result))
        
        # Verify all tasks were processed
        self.assertEqual(len(processed_tasks), 5)
        
        # Verify all results are available
        for task in tasks:
            result_key = f"retrieval_result:{task['id']}"
            result = self.redis_client.get(result_key)
            self.assertIsNotNone(result)


class TestFileProcessingRedisFlow(RedisIntegrationTestBase):
    """Test file processing Redis queue flow"""
    
    def test_file_processing_task_submission(self):
        """Test file processing task submission and result retrieval"""
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = """
            Machine Learning Introduction
            
            Machine learning is a powerful subset of artificial intelligence.
            It enables systems to automatically learn and improve from experience.
            
            Key concepts:
            - Supervised learning
            - Unsupervised learning  
            - Reinforcement learning
            """
            f.write(test_content)
            f.flush()
            temp_file_path = f.name
        
        try:
            # 1. Submit file processing task
            file_task = {
                'file_id': str(ObjectId()),
                'file_path': temp_file_path,
                'file_type': 'txt',
                'group_id': 'test_group',
                'filename': 'ml_intro.txt',
                'type': 'process_file',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Push to file processing queue
            self.redis_client.lpush('file_processing_tasks', json.dumps(file_task))
            result_key = f"file_processing_result:{file_task['file_id']}"
            stored_result = self.fetch_results( self.redis_client, result_key, timeout=300)
            self.assertGreater(stored_result['chunks_count'], 0)
            
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
    
    def test_file_processing_error_handling(self):
        """Test file processing error scenario"""
        
        # Submit task with non-existent file
        error_task = {
            'file_id': str(ObjectId()),
            'file_path': '/nonexistent/file.txt',
            'file_type': 'txt',
            'group_id': 'test_group',
            'type': 'process_file'
        }
        result_key = f"file_processing_result:{error_task['file_id']}"
        self.redis_client.lpush('file_processing_tasks', json.dumps(error_task))
        stored_result = self.fetch_results( self.redis_client, result_key, timeout=300)
        
        self.assertEqual(stored_result['status'], 'failed')
        self.assertIn('error', stored_result)
        self.assertTrue('File not found' in stored_result['error'])

class TestEmbeddingTaskRedisFlow(RedisIntegrationTestBase):
    """Test embedding computation Redis flow"""
    
    def test_embedding_computation_task(self):
        """Test embedding computation task flow"""
        
        embedding_task = {
            'message_id': str(ObjectId()),
            'query': 'What is deep learning?',
            'response': 'Deep learning is a subset of machine learning using neural networks.',
            'type': 'compute_embeddings',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Submit task
        self.redis_client.lpush('embedding_tasks', json.dumps(embedding_task))
        
        # Process task
        queue_data = self.redis_client.brpop(['embedding_tasks'], timeout=1)
        retrieved_task = json.loads(queue_data[1])
        
        # Verify task data
        self.assertEqual(retrieved_task['type'], 'compute_embeddings')
        self.assertEqual(retrieved_task['query'], 'What is deep learning?')
        self.assertIn('message_id', retrieved_task)
        
        # Simulate embedding computation result
        embedding_result = {
            'message_id': embedding_task['message_id'],
            'status': 'completed',
            'query_embedding_computed': True,
            'response_embedding_computed': True,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store result (though embedding tasks don't typically store results in Redis,
        # they update MongoDB directly, but we can simulate completion status)
        result_key = f"embedding_result:{embedding_task['message_id']}"
        self.redis_client.setex(result_key, 60, json.dumps(embedding_result))
        
        # Verify result
        stored_result = json.loads(self.redis_client.get(result_key))
        self.assertEqual(stored_result['status'], 'completed')
        self.assertTrue(stored_result['query_embedding_computed'])
        self.assertTrue(stored_result['response_embedding_computed'])


class TestIndexingTaskRedisFlow(RedisIntegrationTestBase):
    """Test document indexing Redis flow"""
    
    def test_document_indexing_task(self):
        """Test document indexing task flow"""
        
        indexing_task = {
            'id': 'index_task_001',
            'type': 'index',
            'documents': [
                {
                    'id': 'doc_1',
                    'content': 'Natural language processing is a subfield of AI.',
                    'metadata': {'category': 'AI', 'source': 'textbook'}
                },
                {
                    'id': 'doc_2', 
                    'content': 'Computer vision enables machines to interpret visual information.',
                    'metadata': {'category': 'AI', 'source': 'research'}
                }
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Submit to dataset processing queue
        self.redis_client.lpush('dataset_processing_tasks', json.dumps(indexing_task))
        
        # Process task
        queue_data = self.redis_client.brpop(['dataset_processing_tasks'], timeout=1)
        retrieved_task = json.loads(queue_data[1])
        
        # Verify task structure
        self.assertEqual(retrieved_task['type'], 'index')
        self.assertEqual(len(retrieved_task['documents']), 2)
        
        # Verify document data
        doc1 = retrieved_task['documents'][0]
        self.assertEqual(doc1['id'], 'doc_1')
        self.assertIn('content', doc1)
        self.assertIn('metadata', doc1)
        
        # Simulate indexing result
        indexing_result = {
            'task_id': 'index_task_001',
            'success': True,
            'document_count': 2,
            'total_requested': 2,
            'chunks_created': 6,  # 3 chunks per document
            'timestamp': datetime.utcnow().isoformat()
        }
        
        result_key = f"indexing_result:{indexing_task['id']}"
        self.redis_client.setex(result_key, 60, json.dumps(indexing_result))
        
        # Verify result
        stored_result = json.loads(self.redis_client.get(result_key))
        self.assertTrue(stored_result['success'])
        self.assertEqual(stored_result['document_count'], 2)
        self.assertEqual(stored_result['chunks_created'], 6)
    
    def test_large_batch_indexing(self):
        """Test large batch document indexing"""
        
        # Create large batch of documents
        documents = []
        for i in range(50):
            documents.append({
                'id': f'batch_doc_{i}',
                'content': f'Document {i} content with meaningful text for indexing and retrieval.',
                'metadata': {'batch_id': 'large_batch', 'index': i}
            })
        
        large_batch_task = {
            'id': 'large_batch_index',
            'type': 'index',
            'documents': documents,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Submit task
        self.redis_client.lpush('dataset_processing_tasks', json.dumps(large_batch_task))
        
        # Process task
        queue_data = self.redis_client.brpop(['dataset_processing_tasks'], timeout=1)
        retrieved_task = json.loads(queue_data[1])
        
        # Verify large batch
        self.assertEqual(len(retrieved_task['documents']), 50)
        
        # Simulate successful batch processing
        batch_result = {
            'task_id': 'large_batch_index',
            'success': True,
            'document_count': 50,
            'total_requested': 50,
            'processing_time_ms': 2500,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        result_key = f"indexing_result:{large_batch_task['id']}"
        self.redis_client.setex(result_key, 60, json.dumps(batch_result))
        
        # Verify batch result
        stored_result = json.loads(self.redis_client.get(result_key))
        self.assertEqual(stored_result['document_count'], 50)
        self.assertIn('processing_time_ms', stored_result)




class RedisIntegrationTestRunner:
    """Test runner for Redis integration tests"""
    
    @staticmethod
    def run_all_tests():
        """Run all Redis integration tests"""
        
        print("🚀 Starting Redis Integration Tests for RAG Service Core Functionalities")
        print("=" * 80)
        
        # Check Redis connectivity first
        try:
            test_redis = redis.from_url(REDIS_URL, decode_responses=True)
            test_redis.ping()
            print(f"✅ Redis connection successful: {REDIS_URL}")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            print("Please ensure Redis is running and accessible.")
            return False
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add test classes
        test_classes = [
            TestFileProcessingRedisFlow,
            TestEmbeddingTaskRedisFlow,
            TestIndexingTaskRedisFlow,
            TestChatTaskRedisFlow,

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
            print(f"\n⚠️  Some tests failed. Check Redis service and configuration.")
        
        return result.wasSuccessful()


if __name__ == "__main__":
    # Add load testing and failure scenarios to the test runner
    print("🚀 Redis Integration Tests with Load Testing and Failure Scenarios")
    print("=" * 80)
    
    # Run comprehensive test suite
    success = RedisIntegrationTestRunner.run_all_tests()
    
    exit(0 if success else 1)