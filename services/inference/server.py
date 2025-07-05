"""
Inference Service
Handles LLM inference with various backends (Ollama, vLLM, OpenAI)
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
import asyncio

import redis
import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'ollama')
MODEL_NAME = os.environ.get('MODEL_NAME', 'llama2')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Model configuration
MAX_TOKENS = 2048
TEMPERATURE = 0.7
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base. 
Use the provided context to answer questions accurately. If the context doesn't contain 
relevant information, say so and provide the best answer you can based on your general knowledge."""

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


class LLMInference:
    """Base class for LLM inference"""
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        raise NotImplementedError


class OllamaInference(LLMInference):
    """Ollama inference backend"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self):
        """Ensure the model is downloaded and loaded"""
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    logger.info(f"Pulling model {self.model_name}...")
                    pull_response = requests.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name}
                    )
                    if pull_response.status_code != 200:
                        logger.error(f"Failed to pull model: {pull_response.text}")
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        """Generate response using Ollama"""
        try:
            # Construct full prompt with context
            full_prompt = f"{SYSTEM_PROMPT}\n\n"
            if context:
                full_prompt += f"Context:\n{context}\n\n"
            full_prompt += f"User: {prompt}\n\nAssistant:"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', TEMPERATURE),
                        "num_predict": kwargs.get('max_tokens', MAX_TOKENS)
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                logger.error(f"Ollama API error: {response.text}")
                return "Sorry, I encountered an error generating a response."
                
        except Exception as e:
            logger.error(f"Error in Ollama inference: {e}")
            return "Sorry, I encountered an error generating a response."


class VLLMInference(LLMInference):
    """vLLM inference backend"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = "http://localhost:8000"
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        """Generate response using vLLM"""
        try:
            # Construct full prompt with context
            full_prompt = f"{SYSTEM_PROMPT}\n\n"
            if context:
                full_prompt += f"Context:\n{context}\n\n"
            full_prompt += f"User: {prompt}\n\nAssistant:"
            
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "max_tokens": kwargs.get('max_tokens', MAX_TOKENS),
                    "temperature": kwargs.get('temperature', TEMPERATURE)
                }
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['text']
            else:
                logger.error(f"vLLM API error: {response.text}")
                return "Sorry, I encountered an error generating a response."
                
        except Exception as e:
            logger.error(f"Error in vLLM inference: {e}")
            return "Sorry, I encountered an error generating a response."


class OpenAIInference(LLMInference):
    """OpenAI API inference backend"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        """Generate response using OpenAI API"""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Relevant context from knowledge base:\n{context}"
                })
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', MAX_TOKENS),
                temperature=kwargs.get('temperature', TEMPERATURE)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI inference: {e}")
            return "Sorry, I encountered an error generating a response."


# Initialize appropriate inference backend
def create_inference_engine() -> LLMInference:
    """Factory function to create appropriate inference engine"""
    if MODEL_TYPE == 'ollama':
        logger.info(f"Initializing Ollama with model: {MODEL_NAME}")
        return OllamaInference(MODEL_NAME)
    elif MODEL_TYPE == 'vllm':
        logger.info(f"Initializing vLLM with model: {MODEL_NAME}")
        return VLLMInference(MODEL_NAME)
    elif MODEL_TYPE == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for OpenAI models")
        logger.info(f"Initializing OpenAI with model: {MODEL_NAME}")
        return OpenAIInference(MODEL_NAME, OPENAI_API_KEY)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")


# Create inference engine
inference_engine = create_inference_engine()


def format_context(documents: List[Dict]) -> str:
    """Format retrieved documents as context for the LLM"""
    if not documents:
        return ""
    
    context_parts = []
    for i, doc in enumerate(documents[:5]):  # Limit to top 5 documents
        content = doc.get('content', '').strip()
        if content:
            context_parts.append(f"[Source {i+1}]:\n{content}")
    
    return "\n\n".join(context_parts)


def generate_response(prompt: str, context: str, conversation_history: List[Dict]) -> str:
    """Generate response with conversation history"""
    # Format conversation history
    history_prompt = ""
    if conversation_history:
        recent_history = conversation_history[-5:]  # Last 5 exchanges
        for msg in recent_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                history_prompt += f"\nUser: {content}"
            else:
                history_prompt += f"\nAssistant: {content}"
    
    # Combine history with current prompt
    if history_prompt:
        full_prompt = f"Previous conversation:{history_prompt}\n\nCurrent question: {prompt}"
    else:
        full_prompt = prompt
    
    # Generate response
    return inference_engine.generate(full_prompt, context)


def process_inference_tasks():
    """Main loop to process inference tasks from Redis"""
    logger.info("Inference service started")
    
    while True:
        try:
            # Check for new tasks (blocking pop with 1 second timeout)
            task_data = redis_client.brpop('inference_tasks', timeout=1)
            
            if task_data:
                _, task_json = task_data
                task = json.loads(task_json)
                
                logger.info(f"Processing inference task: {task['id']}")
                
                # Extract information
                prompt = task.get('message', '')
                retrieved_docs = task.get('retrieved_context', [])
                conversation_history = task.get('context', [])
                
                # Format context from retrieved documents
                context = format_context(retrieved_docs)
                
                # Generate response
                response = generate_response(prompt, context, conversation_history)
                
                # Store result in Redis
                result = {
                    'task_id': task['id'],
                    'response': response,
                    'model_used': f"{MODEL_TYPE}:{MODEL_NAME}",
                    'context_used': len(retrieved_docs),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                redis_client.setex(
                    f"inference_result:{task['id']}", 
                    60,  # Expire after 60 seconds
                    json.dumps(result)
                )
                
                logger.info(f"Generated response for task {task['id']}")
                
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
        
        # Check model availability
        test_response = inference_engine.generate("Hello", "")
        
        return {
            'status': 'healthy',
            'redis': 'connected',
            'model_type': MODEL_TYPE,
            'model_name': MODEL_NAME,
            'test_response': bool(test_response)
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


if __name__ == '__main__':
    # Run health check
    health = health_check()
    logger.info(f"Health check: {health}")
    
    # Start processing tasks
    process_inference_tasks()