"""
Inference Service with MongoDB Model Usage Tracking
Handles LLM inference with various backends (Ollama,) and tracks model usage statistics
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timezone
import redis
import requests
from dataclasses import dataclass
from pymongo import MongoClient, errors
from pymongo.collection import Collection
import ollama

# Configure logging
logging.basicConfig(filename="/app_data/logs/inference_service.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGO_PORT = os.environ.get('MONGO_INITDB_PORT', '27017')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME', 'admin')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD', 'password')
MONGO_DATABASE = os.environ.get('MONGO_INITDB_DATABASE', 'rag_system')

MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"

# Configuration
REDIS_URL = f"redis://:{os.environ.get('REDIS_PASSWORD', 'password')}@{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}/{os.environ.get('REDIS_DB', '0')}"
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'ollama')
MODEL_NAME = os.environ.get('MODEL_NAME', 'llama3.2:3b')

# Model-specific configuration
MODEL_CONFIGS = {
    'llama3.2:3b': {
        'max_tokens': 4096,  # Llama 3.2 3B supports up to 128K context
        'temperature': 0.6,  # Slightly lower for more consistent RAG responses
        'top_p': 0.9,
        'repeat_penalty': 1.1,
        'stop': ['<|eot_id|>', '<|end|>']
    },
    'default': {
        'max_tokens': 2048,
        'temperature': 0.7,
        'top_p': 0.9,
        'repeat_penalty': 1.1
    }
}

# Get model config
model_config = MODEL_CONFIGS.get(MODEL_NAME, MODEL_CONFIGS['default'])
MAX_TOKENS = model_config['max_tokens']
TEMPERATURE = model_config['temperature']
TOP_P = model_config.get('top_p', 0.9)
REPEAT_PENALTY = model_config.get('repeat_penalty', 1.1)
STOP_SEQUENCES = model_config.get('stop', [])

# System prompt for RAG
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base. 
Use the provided context to answer questions accurately and concisely. 
If the context doesn't contain relevant information, clearly state that and provide the best answer you can based on your general knowledge.
Always cite which source you're using when referencing the context."""

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# MongoDB connection
mongodb_client = None
mongodb_db = None


class ModelUsageTracker:
    """Handles tracking of model usage and inference statistics in MongoDB"""
    
    def __init__(self, db):
        self.db = db
        self.models_collection = db.model_usage
        self.inference_logs_collection = db.inference_logs
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for efficient queries"""
        try:
            # Index on model_name and model_type for model_usage collection
            self.models_collection.create_index([
                ("model_name", 1),
                ("model_type", 1)
            ])
            
            # Index on timestamp for inference_logs collection
            self.inference_logs_collection.create_index([
                ("timestamp", -1)
            ])
            
            # Compound index for filtering logs by model and date
            self.inference_logs_collection.create_index([
                ("model_name", 1),
                ("timestamp", -1)
            ])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def register_model_startup(self, model_name: str, model_type: str, config: dict):
        """Register model startup and initialize/update usage statistics"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Try to find existing model entry
            existing_model = self.models_collection.find_one({
                "model_name": model_name,
                "model_type": model_type
            })
            
            if existing_model:
                # Update existing model entry
                self.models_collection.update_one(
                    {"_id": existing_model["_id"]},
                    {
                        "$set": {
                            "last_started": current_time,
                            "config": config,
                            "status": "active"
                        },
                        "$inc": {
                            "startup_count": 1
                        }
                    }
                )
                logger.info(f"Updated existing model entry for {model_name}")
            else:
                # Create new model entry
                model_doc = {
                    "model_name": model_name,
                    "model_type": model_type,
                    "config": config,
                    "first_started": current_time,
                    "last_started": current_time,
                    "startup_count": 1,
                    "total_requests": 0,
                    "total_tokens_generated": 0,
                    "total_generation_time": 0.0,
                    "average_generation_time": 0.0,
                    "status": "active"
                }
                
                result = self.models_collection.insert_one(model_doc)
                logger.info(f"Created new model entry for {model_name} with ID: {result.inserted_id}")
        
        except Exception as e:
            logger.error(f"Error registering model startup: {e}")
    
    def log_inference_request(self, model_name: str, model_type: str, 
                             prompt: str, response: str, generation_time: float,
                             context_used: int = 0, task_id: str = None,
                             timed_out: bool = False):
        """Log individual inference request"""
        try:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            response_tokens = len(response) // 4
            
            log_entry = {
                "task_id": task_id,
                "model_name": model_name,
                "model_type": model_type,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "estimated_tokens": response_tokens,
                "generation_time": generation_time,
                "context_documents": context_used,
                "timed_out": timed_out,
                "timestamp": datetime.now(timezone.utc)
            }
            
            # Insert log entry
            self.inference_logs_collection.insert_one(log_entry)
            
            # Update model usage statistics only if not timed out
            if not timed_out:
                self.models_collection.update_one(
                    {
                        "model_name": model_name,
                        "model_type": model_type
                    },
                    {
                        "$inc": {
                            "total_requests": 1,
                            "total_tokens_generated": response_tokens,
                            "total_generation_time": generation_time
                        }
                    }
                )
                
                # Update average generation time
                model_stats = self.models_collection.find_one({
                    "model_name": model_name,
                    "model_type": model_type
                })
                
                if model_stats:
                    avg_time = model_stats["total_generation_time"] / model_stats["total_requests"]
                    self.models_collection.update_one(
                        {"_id": model_stats["_id"]},
                        {"$set": {"average_generation_time": avg_time}}
                    )
            
            logger.debug(f"Logged inference request for {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging inference request: {e}")
    
    def get_model_statistics(self, model_name: str = None, model_type: str = None) -> List[Dict]:
        """Get model usage statistics"""
        try:
            query = {}
            if model_name:
                query["model_name"] = model_name
            if model_type:
                query["model_type"] = model_type
            
            stats = list(self.models_collection.find(query))
            
            # Convert ObjectId to string for JSON serialization
            for stat in stats:
                stat["_id"] = str(stat["_id"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return []
    
    def get_inference_logs(self, model_name: str = None, limit: int = 100) -> List[Dict]:
        """Get recent inference logs"""
        try:
            query = {}
            if model_name:
                query["model_name"] = model_name
            
            logs = list(
                self.inference_logs_collection
                .find(query)
                .sort("timestamp", -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string for JSON serialization
            for log in logs:
                log["_id"] = str(log["_id"])
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting inference logs: {e}")
            return []
    
    def mark_model_inactive(self, model_name: str, model_type: str):
        """Mark model as inactive (when service stops)"""
        try:
            self.models_collection.update_one(
                {
                    "model_name": model_name,
                    "model_type": model_type
                },
                {
                    "$set": {
                        "status": "inactive",
                        "last_stopped": datetime.now(timezone.utc)
                    }
                }
            )
            logger.info(f"Marked {model_name} as inactive")
            
        except Exception as e:
            logger.error(f"Error marking model as inactive: {e}")


def get_mongodb_connection(uri: str, database: str) -> Tuple[MongoClient, object]:
    """Get MongoDB connection"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client[database]
        # Test connection
        db.command('ping')
        logger.info("MongoDB connection established")
        return client, db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


# Initialize MongoDB connection
try:
    mongodb_client, mongodb_db = get_mongodb_connection(MONGO_URI, MONGO_DATABASE)
    model_tracker = ModelUsageTracker(mongodb_db)
except Exception as e:
    logger.error(f"Failed to initialize MongoDB tracking: {e}")
    model_tracker = None


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'system', 'user', 'assistant'
    content: str


class ChatTemplate:
    """Handles chat templating for different models"""
    
    @staticmethod
    def format_llama3(messages: List[ChatMessage]) -> str:
        """Format messages for Llama 3 models"""
        formatted = "<|begin_of_text|>"
        
        for msg in messages:
            if msg.role == "system":
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"
        
        # Add the start of assistant response
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted
    
    @staticmethod
    def format_messages(messages: List[ChatMessage], model_name: str) -> str:
        """Format messages based on model type"""
        if 'llama3' in model_name.lower():
            return ChatTemplate.format_llama3(messages)
        else:
            # Default format for other models
            formatted = ""
            for msg in messages:
                if msg.role == "system":
                    formatted += f"System: {msg.content}\n\n"
                elif msg.role == "user":
                    formatted += f"User: {msg.content}\n\n"
                elif msg.role == "assistant":
                    formatted += f"Assistant: {msg.content}\n\n"
            formatted += "Assistant: "
            return formatted


class LLMInference:
    """Base class for LLM inference"""
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        raise NotImplementedError


class OllamaInference(LLMInference):
    """Ollama inference backend"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self):
        """Ensure the model is downloaded and loaded"""
        try:
            # Check if model exists
            client = ollama.Client()
            models = client.list()
            model_names = [model.model for model in models.models]
            if self.model_name not in model_names:
                logger.info(f"Pulling model {self.model_name}...")
                for progress in client.pull(self.model_name, stream=True):
                    logger.info(f"Pull progress: {progress.get('status')}")
        except Exception as e:
            raise Exception(f"Error checking Ollama models: {e}")
    
    def generate(self, prompt: str, context: str = "", conversation_history: List[Dict] = None, timeout: int = 120, **kwargs):
        """Generate response using Ollama with chat template and proper timeout handling"""
        try:
            # Build messages list
            messages = []
            
            # Add system prompt
            system_content = SYSTEM_PROMPT
            if context:
                system_content += f"\n\nContext from knowledge base:\n{context}"
            messages.append(ChatMessage(role="system", content=system_content))
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 exchanges
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role in ['user', 'assistant']:
                        messages.append(ChatMessage(role=role, content=content))
            
            # Add current user prompt
            messages.append(ChatMessage(role="user", content=prompt))
            
            # Format using chat template
            formatted_prompt = ChatTemplate.format_messages(messages, self.model_name)
            client = ollama.Client(timeout=timeout)
            start_time = time.time()
            
            # Create the generator with timeout handling
            generator = client.generate(
                model=self.model_name,
                prompt=formatted_prompt,
                stream=True,
                options={   
                    "temperature": kwargs.get('temperature', TEMPERATURE),
                    "num_predict": kwargs.get('max_tokens', MAX_TOKENS),
                    "top_p": kwargs.get('top_p', TOP_P),
                    "repeat_penalty": kwargs.get('repeat_penalty', REPEAT_PENALTY),
                    "stop": kwargs.get('stop', STOP_SEQUENCES)
                }
            )
            
            # Stream with timeout checking
            for chunk in generator:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if elapsed_time > timeout:
                    logger.warning(f"Generate task:{kwargs.get('task_id', '')} timed out after: {elapsed_time:.2f}s")
                    yield "", True, True  # empty response, done=True, timed_out=True
                    return
                
                if chunk:
                    yield chunk.response, chunk.done, False  # response, done, timed_out=False
                    
                    if chunk.done:
                        return
            
        except Exception as e:
            logger.error(f"Error in Ollama inference: {e}")
            yield "Sorry, I encountered an error generating a response.", True, False


# Initialize appropriate inference backend
def create_inference_engine() -> LLMInference:
    """Factory function to create appropriate inference engine"""
    if MODEL_TYPE == 'ollama':
        logger.info(f"Initializing Ollama with model: {MODEL_NAME}")
        return OllamaInference(MODEL_NAME)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")


# Create inference engine and register model startup
inference_engine = create_inference_engine()

# Register model startup in MongoDB
if model_tracker:
    model_tracker.register_model_startup(
        model_name=MODEL_NAME,
        model_type=MODEL_TYPE,
        config=model_config
    )


def format_context(documents: List[Dict]) -> str:
    """Format retrieved documents as context for the LLM"""
    if not documents:
        return ""
    
    context_parts = []
    for i, doc in enumerate(documents[:5]):  # Limit to top 5 documents
        content = doc.get('content', '').strip()
        metadata = doc.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        
        if content:
            context_parts.append(f"[Source {i+1} - {source}]:\n{content}")
    
    return "\n\n".join(context_parts)


def process_inference_tasks():
    """Main loop to process inference tasks from Redis"""
    logger.info("Inference service started")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Max tokens: {MAX_TOKENS}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"MongoDB tracking: {'enabled' if model_tracker else 'disabled'}")
    
    while True:
        try:
            # Check for new tasks (blocking pop with 1 second timeout)
            task_data = redis_client.brpop('inference_tasks', timeout=1)
            
            if task_data:
                _, task_json = task_data
                task = json.loads(task_json)
                
                task_id = task.get('id', 'unknown')
                logger.info(f"Processing inference task: {task_id}")
                
                # Extract information
                prompt = task.get('message', '')
                retrieved_docs = task.get('retrieved_context', [])
                conversation_history = task.get('context', [])
                
                # Format context from retrieved documents
                context = format_context(retrieved_docs)
                
                # Generate response with tracking
                start_time = time.time()
                response = ""
                timed_out = False
                
                for resp_token, done, timeout_occurred in inference_engine.generate(
                        prompt=prompt,
                        context=context,
                        conversation_history=conversation_history,
                        timeout=task.get('timeout', 120),
                        task_id=task_id
                    ):
                    
                    if timeout_occurred:
                        # Handle timeout case
                        timed_out = True
                        redis_client.lpush(
                            f"inference_result:{task_id}", 
                            "<|STOP|>"
                        )
                        break
                    
                    if done:
                        redis_client.lpush(
                            f"inference_result:{task_id}", 
                            "<|STOP|>"
                        )
                        break
                    
                    if resp_token:
                        redis_client.lpush(
                            f"inference_result:{task_id}", 
                            resp_token
                        )
                        response += resp_token
                
                generation_time = time.time() - start_time
                
                # Log the inference request to MongoDB
                if model_tracker:
                    model_tracker.log_inference_request(
                        model_name=MODEL_NAME,
                        model_type=MODEL_TYPE,
                        prompt=prompt,
                        response=response,
                        generation_time=generation_time,
                        context_used=len(context.split('\n\n')) if context else 0,
                        task_id=task_id,
                        timed_out=timed_out
                    )
                
                # Store result in Redis
                result = {
                    'task_id': task_id,
                    'response': response,
                    'model_used': f"{MODEL_TYPE}:{MODEL_NAME}",
                    'context_used': len(retrieved_docs),
                    'generation_time': round(generation_time, 2),
                    'timed_out': timed_out,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Also publish to a channel for real-time updates
                redis_client.setex(
                    f"inference_complete:{task_id}",
                    300,
                    json.dumps(result)
                )
                
                status = "timed out" if timed_out else "completed"
                logger.info(f"Generated response for task {task_id} {status} in {generation_time:.2f}s")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in task: {e}")
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            time.sleep(1)  # Wait before retrying


def health_check():
    """Health check for the inference service"""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check MongoDB connection
        mongodb_status = "connected" if model_tracker else "disconnected"
        if model_tracker:
            # Try to get model stats to verify connection
            model_stats = model_tracker.get_model_statistics(MODEL_NAME, MODEL_TYPE)
            mongodb_status = f"connected ({len(model_stats)} models tracked)"
        
        # Check model availability with a quick test
        test_response = ""
        for resp_token, done, timed_out in inference_engine.generate("Hello", "", timeout=10):
            if done or timed_out:
                break
            test_response += resp_token
        
        return {
            'status': 'healthy',
            'redis': 'connected',
            'mongodb': mongodb_status,
            'model_type': MODEL_TYPE,
            'model_name': MODEL_NAME,
            'model_config': {
                'max_tokens': MAX_TOKENS,
                'temperature': TEMPERATURE,
                'top_p': TOP_P,
                'repeat_penalty': REPEAT_PENALTY
            },
            'test_response': bool(test_response)
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


def shutdown_handler():
    """Handle graceful shutdown"""
    logger.info("Shutting down inference service...")
    
    # Mark model as inactive
    if model_tracker:
        model_tracker.mark_model_inactive(MODEL_NAME, MODEL_TYPE)
    
    # Close MongoDB connection
    if mongodb_client:
        mongodb_client.close()
    
    logger.info("Shutdown complete")


if __name__ == '__main__':
    import signal
    import sys
    from pathlib import Path
    status_file = '/app_data/logs/inference_service_status.txt'
    if Path(status_file).exists():
        os.remove(status_file)
        
    # Register shutdown handler
    def signal_handler(sig, frame):
        shutdown_handler()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run health check
    health = health_check()
    logger.info(f"Health check: {json.dumps(health, indent=2)}")
    
    if health['status'] == 'healthy':
        try:
            try:
                with open(status_file, 'w') as f:
                    f.write(f"SUCCESS: Inference service healthy\n")
                logger.info(f"Success status file created at {status_file}")
            except Exception as e:
                logger.error(f"Failed to write status file: {e}")
            # Start processing tasks
            process_inference_tasks()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            shutdown_handler()
    else:
        logger.error("Health check failed, not starting service")
        exit(1)