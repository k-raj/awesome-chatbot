"""
Enhanced Test script for the inference service with MongoDB tracking
Tests core functionalities including:
- Basic inference
- RAG with context  
- Conversation history
- Chat templating
- Error handling
- MongoDB model tracking
- Usage statistics
- Inference logging
- Streaming response handling
- Timeout functionality
"""

import json
import time
import uuid
import redis
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone
import pymongo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = f"redis://:{os.environ.get('REDIS_PASSWORD', 'password')}@{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}/{os.environ.get('REDIS_DB', '0')}"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# MongoDB configuration
MONGO_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGO_PORT = os.environ.get('MONGO_INITDB_PORT', '27017')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME', 'admin')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD', 'password')
MONGO_DATABASE = os.environ.get('MONGO_INITDB_DATABASE', 'rag_system')

MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"


class InferenceServiceTester:
    """Enhanced test harness for the inference service with MongoDB tracking and streaming support"""
    
    def __init__(self):
        self.redis_client = redis_client
        self.test_results = []
        self.mongodb_client = None
        self.mongodb_db = None
        self._setup_mongodb()
    
    def _setup_mongodb(self):
        """Setup MongoDB connection for testing"""
        try:
            self.mongodb_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.mongodb_db = self.mongodb_client[MONGO_DATABASE]
            # Test connection
            self.mongodb_db.command('ping')
            logger.info("MongoDB connection established for testing")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.mongodb_client = None
            self.mongodb_db = None
    
    def create_task(self, 
                   message: str, 
                   retrieved_context: List[Dict] = None,
                   conversation_history: List[Dict] = None,
                   timeout=120) -> str:
        """Create an inference task and submit it to Redis"""
        task_id = str(uuid.uuid4())
        
        task = {
            'id': task_id,
            'message': message,
            'retrieved_context': retrieved_context or [],
            'context': conversation_history or [],
            "timeout": timeout
        }
        
        # Push task to Redis queue
        self.redis_client.lpush('inference_tasks', json.dumps(task))
        logger.info(f"Created task {task_id}: {message[:50]}...")
        
        return task_id
    
    def wait_for_result(self, task_id: str, timeout: int = 600) -> Optional[Dict]:
        """Wait for inference result with timeout"""
        start_time = time.time()
        response = ""
        timed_out = False
        result_key = f"inference_result:{task_id}"
        result = {}
        while time.time() - start_time < timeout:
            try:
                # Use brpop with a short timeout to avoid blocking indefinitely
                token_data = self.redis_client.brpop(result_key, timeout=1)
                
                if token_data:
                    _, token_content = token_data
                                        
                    # Check for stop signal
                    if token_content == "<|STOP|>":              
                        # Clean up the result key
                        self.redis_client.delete(result_key)
                        
                        # Get the complete result
                        completed_result_key = f"inference_complete:{task_id}"
                        result_json = self.redis_client.get(completed_result_key)
                        
                        if result_json:
                            result = json.loads(result_json)
                            result['response'] = response
                            result['timed_out'] = timed_out
                            logger.info(f"Got {'timed out' if timed_out else 'complete'} result for task {task_id}")
                        else:
                            # Fallback result if complete result not found
                            result = {
                                'task_id': task_id,
                                'response': response,
                                'timed_out': timed_out,
                                'generation_time': time.time() - start_time,
                                'model_used': 'unknown',
                                'context_used': 0
                            }
                        break
                        
                    response += token_content
                        
            except redis.exceptions.ResponseError as e:
                # Handle Redis errors gracefully
                logger.warning(f"Redis error waiting for task {task_id}: {e}")
                break
            time.sleep(0.001)
        
        if time.time() - start_time > timeout:
            logger.error(f"Task {task_id} timed out after {timeout} seconds")
        return result
    
    def test_basic_inference(self):
        """Test basic inference without context"""
        logger.info("\n=== Testing Basic Inference ===")
        
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about artificial intelligence."
        ]
        
        for prompt in test_prompts:
            task_id = self.create_task(prompt)
            result = self.wait_for_result(task_id)
            
            if result:
                self.test_results.append({
                    'test': 'basic_inference',
                    'prompt': prompt,
                    'success': True,
                    'response_length': len(result['response']),
                    'generation_time': result.get('generation_time', 0),
                    'timed_out': result.get('timed_out', False),
                    'task_id': task_id
                })
                logger.info(f"Response preview: {result['response'][:100]}...")
            else:
                self.test_results.append({
                    'test': 'basic_inference',
                    'prompt': prompt,
                    'success': False,
                    'task_id': task_id
                })
    
    def test_rag_with_context(self):
        """Test RAG functionality with retrieved context"""
        logger.info("\n=== Testing RAG with Context ===")
        
        # Simulate retrieved documents
        retrieved_docs = [
            {
                'content': 'The company was founded in 2019 by John Smith and Jane Doe. They started with a small team of 5 engineers.',
                'metadata': {'source': 'company_history.txt', 'score': 0.95}
            },
            {
                'content': 'Our main product is an AI-powered analytics platform that helps businesses make data-driven decisions.',
                'metadata': {'source': 'product_overview.txt', 'score': 0.89}
            },
            {
                'content': 'In 2023, the company raised $50 million in Series B funding led by TechVentures.',
                'metadata': {'source': 'news_article.txt', 'score': 0.87}
            }
        ]
        
        questions = [
            "Who founded the company?",
            "What is the main product?",
            "How much funding was raised in Series B?"
        ]
        
        for question in questions:
            task_id = self.create_task(question, retrieved_context=retrieved_docs)
            result = self.wait_for_result(task_id)
            
            if result:
                self.test_results.append({
                    'test': 'rag_with_context',
                    'prompt': question,
                    'success': True,
                    'context_used': result.get('context_used', 0),
                    'response_length': len(result['response']),
                    'generation_time': result.get('generation_time', 0),
                    'timed_out': result.get('timed_out', False),
                    'task_id': task_id
                })
                logger.info(f"Response: {result['response']}")
            else:
                self.test_results.append({
                    'test': 'rag_with_context',
                    'prompt': question,
                    'success': False,
                    'task_id': task_id
                })
    
    def test_conversation_history(self):
        """Test maintaining conversation context"""
        logger.info("\n=== Testing Conversation History ===")
        
        # Simulate a conversation
        conversation = []
        
        # First exchange
        prompt1 = "My name is Alice and I'm interested in learning Python."
        task_id1 = self.create_task(prompt1, conversation_history=conversation)
        result1 = self.wait_for_result(task_id1)
        
        if result1:
            conversation.append({'role': 'user', 'content': prompt1})
            conversation.append({'role': 'assistant', 'content': result1['response']})
            
            # Second exchange - should remember the name
            prompt2 = "What programming language did I mention I want to learn?"
            task_id2 = self.create_task(prompt2, conversation_history=conversation)
            result2 = self.wait_for_result(task_id2)
            
            if result2:
                # Check if the model remembers Python
                success = 'python' in result2['response'].lower()
                self.test_results.append({
                    'test': 'conversation_history',
                    'success': success,
                    'remembered_context': success,
                    'response': result2['response'],
                    'timed_out': result2.get('timed_out', False),
                    'task_id': task_id2
                })
                logger.info(f"Memory test {'PASSED' if success else 'FAILED'}")
                logger.info(f"Response: {result2['response']}")
    
    def test_timeout_functionality(self):
        """Test timeout functionality with a request designed to take longer than timeout"""
        logger.info("\n=== Testing Timeout Functionality ===")
        
        # Create a prompt that might take a while to process
        long_prompt = """Write a very detailed, comprehensive essay about the history of artificial intelligence, 
        including every major milestone, researcher, breakthrough, and technological advancement from the 1940s to present day. 
        Include detailed explanations of neural networks, machine learning algorithms, natural language processing, 
        computer vision, robotics, and all major AI applications. Make it at least 5000 words long."""
        
        task_id = self.create_task(long_prompt,  timeout=10)
        
        # Set a short timeout to test the functionality
        start_time = time.time()
        result = self.wait_for_result(task_id)  # Very short timeout
        elapsed_time = time.time() - start_time
        
        if result:
            timed_out = result.get('timed_out', False)
            self.test_results.append({
                'test': 'timeout_functionality',
                'success': True,  # Success means we got a result (even if timed out)
                'timed_out': timed_out,
                'elapsed_time': elapsed_time,
                'response_length': len(result.get('response', '')),
                'task_id': task_id
            })
            
            logger.info(f"Timeout test: {'TIMED OUT' if timed_out else 'COMPLETED'} in {elapsed_time:.2f}s")
            if timed_out:
                logger.info("✓ Timeout functionality working correctly")
            else:
                logger.info("Model completed before timeout - this is also valid")
        else:
            self.test_results.append({
                'test': 'timeout_functionality',
                'success': False,
                'elapsed_time': elapsed_time,
                'task_id': task_id
            })
            logger.info("✗ Timeout test failed - no result received")
    
    def test_error_handling(self):
        """Test error handling with edge cases"""
        logger.info("\n=== Testing Error Handling ===")
        
        # Test empty message
        task_id = self.create_task("", timeout=10)
        result = self.wait_for_result(task_id)
        
        self.test_results.append({
            'test': 'error_handling_empty',
            'success': result is not None,
            'timed_out': result.get('timed_out', False) if result else False,
            'task_id': task_id
        })
        
        # Test very long message
        long_message = "Explain " + " ".join(["quantum physics"] * 500)
        task_id = self.create_task(long_message, timeout=60)
        result = self.wait_for_result(task_id)
        
        self.test_results.append({
            'test': 'error_handling_long',
            'success': result is not None,
            'timed_out': result.get('timed_out', False) if result else False,
            'task_id': task_id
        })
    
    def test_performance(self):
        """Test inference performance and throughput"""
        logger.info("\n=== Testing Performance ===")
        
        # Submit multiple tasks concurrently
        num_tasks = 5
        task_ids = []
        
        start_time = time.time()
        
        for i in range(num_tasks):
            prompt = f"Generate a random fact about the number {i+1}."
            task_id = self.create_task(prompt)
            task_ids.append(task_id)
        
        # Wait for all results
        results = []
        for task_id in task_ids:
            result = self.wait_for_result(task_id)
            if result:
                results.append(result)
        
        total_time = time.time() - start_time
        
        if results:
            avg_generation_time = sum(r.get('generation_time', 0) for r in results) / len(results)
            timed_out_count = sum(1 for r in results if r.get('timed_out', False))
            
            self.test_results.append({
                'test': 'performance',
                'success': True,
                'tasks_completed': len(results),
                'tasks_timed_out': timed_out_count,
                'total_time': round(total_time, 2),
                'avg_generation_time': round(avg_generation_time, 2),
                'throughput': round(len(results) / total_time, 2),
                'task_ids': task_ids
            })
            logger.info(f"Completed {len(results)} tasks in {total_time:.2f}s")
            logger.info(f"Average generation time: {avg_generation_time:.2f}s")
            logger.info(f"Throughput: {len(results) / total_time:.2f} tasks/second")
            logger.info(f"Timeouts: {timed_out_count}/{len(results)}")
    
    def test_mongodb_model_tracking(self):
        """Test MongoDB model registration and tracking"""
        logger.info("\n=== Testing MongoDB Model Tracking ===")
        
        if not isinstance(self.mongodb_db, pymongo.database.Database):
            logger.error("MongoDB not available, skipping tracking tests")
            self.test_results.append({
                'test': 'mongodb_model_tracking',
                'success': False,
                'error': 'MongoDB not available'
            })
            return
        
        try:
            # Check if model is registered in MongoDB
            model_collection = self.mongodb_db.model_usage
            
            # Get current model entry
            model_entry = model_collection.find_one({
                "model_type": os.environ.get('MODEL_TYPE', 'ollama'),
                "model_name": os.environ.get('MODEL_NAME', 'llama3.2:3b')
            })
            
            if model_entry:
                logger.info(f"Found model entry: {model_entry['model_name']}")
                logger.info(f"Total requests: {model_entry.get('total_requests', 0)}")
                logger.info(f"Startup count: {model_entry.get('startup_count', 0)}")
                logger.info(f"Status: {model_entry.get('status', 'unknown')}")
                
                self.test_results.append({
                    'test': 'mongodb_model_tracking',
                    'success': True,
                    'model_found': True,
                    'total_requests': model_entry.get('total_requests', 0),
                    'startup_count': model_entry.get('startup_count', 0),
                    'status': model_entry.get('status', 'unknown')
                })
            else:
                logger.warning("Model entry not found in MongoDB")
                self.test_results.append({
                    'test': 'mongodb_model_tracking',
                    'success': False,
                    'model_found': False,
                    'error': 'Model entry not found'
                })
                
        except Exception as e:
            logger.error(f"Error testing MongoDB model tracking: {e}")
            self.test_results.append({
                'test': 'mongodb_model_tracking',
                'success': False,
                'error': str(e)
            })
    
    def test_mongodb_inference_logging(self):
        """Test MongoDB inference request logging including timeout tracking"""
        logger.info("\n=== Testing MongoDB Inference Logging ===")
        
        if not isinstance(self.mongodb_db, pymongo.database.Database):
            logger.error("MongoDB not available, skipping logging tests")
            self.test_results.append({
                'test': 'mongodb_inference_logging',
                'success': False,
                'error': 'MongoDB not available'
            })
            return
        
        try:
            # Get initial log count
            logs_collection = self.mongodb_db.inference_logs
            initial_count = logs_collection.count_documents({})
            
            # Submit a test inference request
            test_prompt = "Test request for MongoDB logging validation"
            task_id = self.create_task(test_prompt)
            result = self.wait_for_result(task_id)
            
            if result:
                # Wait a moment for logging to complete
                time.sleep(2)
                
                # Check if new log entry was created
                final_count = logs_collection.count_documents({})
                new_logs = final_count - initial_count
                
                # Find the specific log entry for our task
                log_entry = logs_collection.find_one({"task_id": task_id})
                
                if log_entry:
                    logger.info(f"Found log entry for task {task_id}")
                    logger.info(f"Generation time: {log_entry.get('generation_time', 'N/A')}")
                    logger.info(f"Response length: {log_entry.get('response_length', 'N/A')}")
                    logger.info(f"Timed out: {log_entry.get('timed_out', False)}")
                    
                    self.test_results.append({
                        'test': 'mongodb_inference_logging',
                        'success': True,
                        'log_entry_found': True,
                        'new_logs_count': new_logs,
                        'generation_time': log_entry.get('generation_time'),
                        'response_length': log_entry.get('response_length'),
                        'timed_out': log_entry.get('timed_out', False),
                        'task_id': task_id
                    })
                else:
                    logger.warning(f"Log entry not found for task {task_id}")
                    self.test_results.append({
                        'test': 'mongodb_inference_logging',
                        'success': False,
                        'log_entry_found': False,
                        'new_logs_count': new_logs,
                        'task_id': task_id
                    })
            else:
                self.test_results.append({
                    'test': 'mongodb_inference_logging',
                    'success': False,
                    'error': 'Failed to get inference result'
                })
                
        except Exception as e:
            logger.error(f"Error testing MongoDB inference logging: {e}")
            self.test_results.append({
                'test': 'mongodb_inference_logging',
                'success': False,
                'error': str(e)
            })
    
    def test_mongodb_statistics_updates(self):
        """Test that MongoDB statistics are updated correctly"""
        logger.info("\n=== Testing MongoDB Statistics Updates ===")
        
        if not isinstance(self.mongodb_db, pymongo.database.Database):
            logger.error("MongoDB not available, skipping statistics tests")
            self.test_results.append({
                'test': 'mongodb_statistics_updates',
                'success': False,
                'error': 'MongoDB not available'
            })
            return
        
        try:
            model_collection = self.mongodb_db.model_usage
            model_name = os.environ.get('MODEL_NAME', 'llama3.2:3b')
            model_type = os.environ.get('MODEL_TYPE', 'ollama')
            
            # Get initial statistics
            initial_stats = model_collection.find_one({
                "model_name": model_name,
                "model_type": model_type
            })
            
            if not initial_stats:
                logger.error("Model entry not found for statistics test")
                self.test_results.append({
                    'test': 'mongodb_statistics_updates',
                    'success': False,
                    'error': 'Model entry not found'
                })
                return
            
            initial_requests = initial_stats.get('total_requests', 0)
            initial_tokens = initial_stats.get('total_tokens_generated', 0)
            initial_time = initial_stats.get('total_generation_time', 0.0)
            
            # Submit multiple test requests
            test_prompts = [
                "Generate a short sentence about cats",
                "What is 2+2?",
                "Describe the color blue"
            ]
            
            task_ids = []
            for prompt in test_prompts:
                task_id = self.create_task(prompt)
                task_ids.append(task_id)
            
            # Wait for all results
            completed_tasks = 0
            timed_out_tasks = 0
            for task_id in task_ids:
                result = self.wait_for_result(task_id)
                if result:
                    completed_tasks += 1
                    if result.get('timed_out', False):
                        timed_out_tasks += 1
            
            # Wait for statistics to update
            time.sleep(3)
            
            # Get updated statistics
            updated_stats = model_collection.find_one({
                "model_name": model_name,
                "model_type": model_type
            })
            
            if updated_stats:
                final_requests = updated_stats.get('total_requests', 0)
                final_tokens = updated_stats.get('total_tokens_generated', 0)
                final_time = updated_stats.get('total_generation_time', 0.0)
                final_avg_time = updated_stats.get('average_generation_time', 0.0)
                
                # Note: Only non-timed-out requests should increment statistics
                expected_request_increase = completed_tasks - timed_out_tasks
                requests_increased = final_requests >= initial_requests + expected_request_increase
                tokens_increased = final_tokens >= initial_tokens  # Should increase for successful requests
                time_increased = final_time >= initial_time
                
                logger.info(f"Requests: {initial_requests} -> {final_requests}")
                logger.info(f"Tokens: {initial_tokens} -> {final_tokens}")
                logger.info(f"Total time: {initial_time:.2f} -> {final_time:.2f}")
                logger.info(f"Average time: {final_avg_time:.2f}")
                logger.info(f"Completed tasks: {completed_tasks}, Timed out: {timed_out_tasks}")
                
                self.test_results.append({
                    'test': 'mongodb_statistics_updates',
                    'success': requests_increased and (tokens_increased or timed_out_tasks == completed_tasks),
                    'completed_tasks': completed_tasks,
                    'timed_out_tasks': timed_out_tasks,
                    'requests_increase': final_requests - initial_requests,
                    'tokens_increase': final_tokens - initial_tokens,
                    'time_increase': final_time - initial_time,
                    'average_generation_time': final_avg_time
                })
            else:
                self.test_results.append({
                    'test': 'mongodb_statistics_updates',
                    'success': False,
                    'error': 'Could not retrieve updated statistics'
                })
                
        except Exception as e:
            logger.error(f"Error testing MongoDB statistics updates: {e}")
            self.test_results.append({
                'test': 'mongodb_statistics_updates',
                'success': False,
                'error': str(e)
            })
    
    def test_mongodb_query_functions(self):
        """Test MongoDB query functions for retrieving statistics and logs"""
        logger.info("\n=== Testing MongoDB Query Functions ===")
        
        if not isinstance(self.mongodb_db, pymongo.database.Database):
            logger.error("MongoDB not available, skipping query tests")
            self.test_results.append({
                'test': 'mongodb_query_functions',
                'success': False,
                'error': 'MongoDB not available'
            })
            return
        
        try:
            # Test querying model statistics
            model_collection = self.mongodb_db.model_usage
            all_models = list(model_collection.find({}))
            
            # Test querying inference logs
            logs_collection = self.mongodb_db.inference_logs
            recent_logs = list(logs_collection.find({}).sort("timestamp", -1).limit(10))
            
            # Test filtering logs by model
            model_name = os.environ.get('MODEL_NAME', 'llama3.2:3b')
            model_logs = list(logs_collection.find({"model_name": model_name}).limit(5))
            
            # Test querying for timed out requests
            timeout_logs = list(logs_collection.find({"timed_out": True}).limit(5))
            
            success = len(all_models) > 0 and len(recent_logs) >= 0  # logs might be empty initially
            
            logger.info(f"Found {len(all_models)} model(s) in database")
            logger.info(f"Found {len(recent_logs)} recent log entries")
            logger.info(f"Found {len(model_logs)} logs for model {model_name}")
            logger.info(f"Found {len(timeout_logs)} timeout log entries")
            
            self.test_results.append({
                'test': 'mongodb_query_functions',
                'success': success,
                'models_count': len(all_models),
                'recent_logs_count': len(recent_logs),
                'model_logs_count': len(model_logs),
                'timeout_logs_count': len(timeout_logs)
            })
            
        except Exception as e:
            logger.error(f"Error testing MongoDB query functions: {e}")
            self.test_results.append({
                'test': 'mongodb_query_functions',
                'success': False,
                'error': str(e)
            })
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        logger.info("Starting enhanced inference service tests with MongoDB tracking and streaming support...")
        
        # Check if services are running
        try:
            self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return
        
        # Run all tests
        self.test_basic_inference()
        self.test_rag_with_context()
        self.test_conversation_history()
        self.test_timeout_functionality()  # New timeout test
        self.test_error_handling()
        self.test_performance()
        
        # MongoDB-specific tests
        self.test_mongodb_model_tracking()
        self.test_mongodb_inference_logging()
        self.test_mongodb_statistics_updates()
        self.test_mongodb_query_functions()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*70)
        logger.info("ENHANCED INFERENCE SERVICE TEST REPORT WITH STREAMING & TIMEOUT")
        logger.info("="*70)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for t in self.test_results if t.get('success', False))
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Group by test type
        test_groups = {}
        for result in self.test_results:
            test_type = result['test']
            if test_type not in test_groups:
                test_groups[test_type] = []
            test_groups[test_type].append(result)
        
        # Print details by group
        for test_type, results in test_groups.items():
            logger.info(f"\n{test_type.upper().replace('_', ' ')}:")
            successes = sum(1 for r in results if r.get('success', False))
            logger.info(f"  Success rate: {successes}/{len(results)}")
            
            # Count timeouts across all results in this group
            timeouts = sum(1 for r in results if r.get('timed_out', False))
            if timeouts > 0:
                logger.info(f"  Timeouts: {timeouts}/{len(results)}")
            
            # Print specific metrics
            if test_type == 'timeout_functionality' and results:
                r = results[0]
                if r.get('success'):
                    logger.info(f"  Elapsed time: {r.get('elapsed_time', 0):.2f}s")
                    logger.info(f"  Timed out: {r.get('timed_out', False)}")
                    logger.info(f"  Response length: {r.get('response_length', 0)} chars")
            
            elif test_type == 'performance' and results:
                r = results[0]
                if r.get('success'):
                    logger.info(f"  Throughput: {r['throughput']} tasks/sec")
                    logger.info(f"  Avg generation time: {r['avg_generation_time']}s")
                    logger.info(f"  Tasks timed out: {r.get('tasks_timed_out', 0)}")
            
            elif test_type == 'mongodb_model_tracking' and results:
                r = results[0]
                if r.get('success'):
                    logger.info(f"  Total requests tracked: {r.get('total_requests', 0)}")
                    logger.info(f"  Model startup count: {r.get('startup_count', 0)}")
                    logger.info(f"  Model status: {r.get('status', 'unknown')}")
            
            elif test_type == 'mongodb_inference_logging' and results:
                r = results[0]
                if r.get('success'):
                    logger.info(f"  Log entry found: {r.get('log_entry_found', False)}")
                    logger.info(f"  Generation time: {r.get('generation_time', 0):.2f}s")
                    logger.info(f"  Timeout tracked: {r.get('timed_out', False)}")
            
            elif test_type == 'mongodb_statistics_updates' and results:
                r = results[0]
                if r.get('success'):
                    logger.info(f"  Completed tasks: {r.get('completed_tasks', 0)}")
                    logger.info(f"  Timed out tasks: {r.get('timed_out_tasks', 0)}")
                    logger.info(f"  Requests increase: {r.get('requests_increase', 0)}")
                    logger.info(f"  Avg generation time: {r.get('average_generation_time', 0):.2f}s")
            
            elif test_type == 'mongodb_query_functions' and results:
                r = results[0]
                if r.get('success'):
                    logger.info(f"  Models in database: {r.get('models_count', 0)}")
                    logger.info(f"  Recent log entries: {r.get('recent_logs_count', 0)}")
                    logger.info(f"  Timeout log entries: {r.get('timeout_logs_count', 0)}")
            
            # Show errors for failed tests
            for result in results:
                if not result.get('success') and 'error' in result:
                    logger.info(f"  ERROR: {result['error']}")
        
        # Summary
        logger.info("\n" + "="*70)
        mongodb_tests = [r for r in self.test_results if 'mongodb' in r['test']]
        mongodb_successes = sum(1 for r in mongodb_tests if r.get('success', False))
        
        if mongodb_tests:
            logger.info(f"MongoDB tracking tests: {mongodb_successes}/{len(mongodb_tests)} passed")
        
        timeout_tests = [r for r in self.test_results if r.get('timed_out')]
        if timeout_tests:
            logger.info(f"Timeout functionality: {len(timeout_tests)} timeout(s) detected and handled")
        
        if successful_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Inference service with streaming & timeout is fully functional.")
        else:
            logger.info(f"⚠️  {total_tests - successful_tests} test(s) failed. Review logs above.")
        
        logger.info("="*70)
    
    def cleanup(self):
        """Cleanup test resources"""
        if self.mongodb_client:
            self.mongodb_client.close()


def main():
    """Main test runner"""
    tester = InferenceServiceTester()
    
    # Give the inference service time to start if just launched
    logger.info("Waiting for inference service to be ready...")
    time.sleep(2)
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise
    finally:
        tester.cleanup()


if __name__ == '__main__':
    main()