"""
Advanced Embedding Service with Hybrid Retrieval - MongoDB Version (Fixed for ChromaDB v2)
Handles document embedding, BM25 indexing, and similarity search
"""

import os
import json
import time
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from bson import ObjectId

import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
import docx

# Import advanced RAG components
from rag_utils import (
    Document, AdvancedRAGPipeline,
    initialize_chroma_client, get_mongodb_connection,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = f"redis://:{os.environ.get('REDIS_PASSWORD', 'password')}@{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}/{os.environ.get('REDIS_DB', '0')}"
CHROMA_HOST = os.environ.get('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.environ.get('CHROMA_PORT', '8000'))
CHROMA_PERSISTENT = os.environ.get('CHROMA_PERSISTENT', 'true').lower() == 'true'
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
RERANKER_MODEL = os.environ.get('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

# MongoDB configuration
MONGO_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGO_PORT = os.environ.get('MONGO_INITDB_PORT', '27017')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME', 'admin')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD', 'password')
MONGO_DATABASE = os.environ.get('MONGO_INITDB_DATABASE', 'rag_system')

MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"

# Global variables for services
redis_client = None
embedding_model = None
chroma_client = None
mongo_client = None
db = None
rag_pipeline = None


def initialize_services():
    """Initialize all services with proper error handling"""
    global redis_client, embedding_model, chroma_client, mongo_client, db, rag_pipeline
    
    logger.info("Initializing services...")
    
    # Initialize Redis
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise
    
    # Initialize embedding model
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding model {EMBEDDING_MODEL} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise
    
    # Initialize ChromaDB
    try:
        if CHROMA_PERSISTENT:
            chroma_client = initialize_chroma_client(use_persistent=True)
        else:
            chroma_client = initialize_chroma_client(
                host=CHROMA_HOST, 
                port=CHROMA_PORT, 
                use_persistent=False
            )
        logger.info("ChromaDB initialized successfully")
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
        # Fallback to in-memory client
        chroma_client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
        logger.warning("Using in-memory ChromaDB client")
    
    # Initialize MongoDB
    try:
        mongo_client, db = get_mongodb_connection(MONGO_URI, MONGO_DATABASE)
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise
    
    # Initialize RAG pipeline
    try:
        rag_pipeline = AdvancedRAGPipeline(
            mongo_uri=MONGO_URI,
            database_name=MONGO_DATABASE,
            redis_client=redis_client,
            chroma_client=chroma_client
        )
        logger.info("Advanced RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"RAG pipeline initialization failed: {e}")
        raise
    
    logger.info("All services initialized successfully")


def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract text from different file types"""
    try:
        logger.info(f"Extracting text from {file_type} file: {file_path}")
        
        if file_type == 'pdf':
            text = ""
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Add page markers for better chunking
                            text += f"\n\n[Page {i+1}]\n{page_text}"
                logger.info(f"Extracted {len(text)} characters from PDF")
                return text
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                return ""
        
        elif file_type in ['doc', 'docx']:
            text = ""
            try:
                doc = docx.Document(file_path)
                
                # Extract from paragraphs
                for para in doc.paragraphs:
                    if para.text.strip():
                        text += para.text + "\n\n"
                
                # Extract from tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        if row_text.strip():
                            text += row_text + "\n"
                    text += "\n"
                
                logger.info(f"Extracted {len(text)} characters from DOCX")
                return text
            except Exception as e:
                logger.error(f"DOCX extraction failed: {e}")
                return ""
        
        elif file_type == 'txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                logger.info(f"Extracted {len(text)} characters from TXT")
                return text
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            text = file.read()
                        logger.info(f"Extracted {len(text)} characters from TXT using {encoding}")
                        return text
                    except:
                        continue
                logger.error("Failed to decode text file with any encoding")
                return ""
        
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return ""
    
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""


def log_retrieval_performance(query: str, num_results: int, response_time_ms: int):
    """Log retrieval performance metrics to MongoDB"""
    try:
        db.retrieval_logs.insert_one({
            'query': query,
            'num_results': num_results,
            'response_time_ms': response_time_ms,
            'created_at': datetime.utcnow()
        })
    except Exception as e:
        logger.error(f"Error logging retrieval performance: {e}")


def process_uploaded_file(task: Dict):
    """Process an uploaded file with advanced chunking and indexing"""
    file_id = task['file_id']
    file_path = task['file_path']
    file_type = task['file_type']
    group_id = task['group_id']
    
    logger.info(f"Processing file {file_id}: {file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text from file
        text = extract_text_from_file(file_path, file_type)
        
        if not text or len(text.strip()) < 10:
            raise ValueError("No meaningful text extracted from file")
        
        logger.info(f"Extracted {len(text)} characters from file {file_id}")
        
        # Create document
        doc = Document(
            id=f"upload_{file_id}",
            content=text,
            metadata={
                'file_id': file_id,
                'file_type': file_type,
                'group_id': group_id,
                'source': 'user_upload',
                'upload_date': datetime.utcnow().isoformat(),
                'filename': task.get('filename', 'unknown'),
                'file_size': len(text),
                'processed_by': 'advanced_rag_pipeline'
            }
        )
        
        # Process with advanced pipeline
        chunks_count = rag_pipeline.process_document(doc)
        
        # Update file status in MongoDB
        db.file_uploads.update_one(
            {'_id': ObjectId(file_id)},
            {
                '$set': {
                    'upload_status': 'completed',
                    'chunks_count': chunks_count,
                    'processed_at': datetime.utcnow(),
                    'text_length': len(text)
                }
            }
        )
            
        logger.info(f"Successfully processed file {file_id} with {chunks_count} chunks")
        
        # Store result in Redis
        result = {
            'file_id': file_id,
            'status': 'completed',
            'chunks_count': chunks_count,
            'text_length': len(text),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            redis_client.setex(
                f"file_processing_result:{file_id}",
                300,  # 5 minutes TTL
                json.dumps(result)
            )
        except:
            logger.warning("Failed to store result in Redis")
        
    except Exception as e:
        logger.error(f"Error processing file {file_id}: {e}")
        
        # Update file status to failed
        try:
            db.file_uploads.update_one(
                {'_id': ObjectId(file_id)},
                {
                    '$set': {
                        'upload_status': 'failed',
                        'error_message': str(e),
                        'processed_at': datetime.utcnow()
                    }
                }
            )
        except Exception as db_error:
            logger.error(f"Failed to update file status in MongoDB: {db_error}")
        
        # Store error result in Redis
        result = {
            'file_id': file_id,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            redis_client.setex(
                f"file_processing_result:{file_id}",
                300,
                json.dumps(result)
            )
        except:
            logger.warning("Failed to store error result in Redis")


def process_chat_task(task: Dict):
    """Process chat retrieval task"""
    try:
        start_time = time.time()
        
        # Apply filters if group_id is specified
        filters = None
        if task.get('group_id') and task['group_id'] != 'general':
            filters = {'group_id': task['group_id']}
        
        logger.info(f"Processing chat query: '{task['message'][:100]}...' with filters: {filters}")
        
        # Retrieve using advanced pipeline
        results = rag_pipeline.retrieve(
            query=task['message'],
            k=5,
            filters=filters
        )
        
        # Format results for response
        documents = []
        for result in results:
            documents.append({
                'id': result['chunk_id'],
                'content': result['content'],
                'metadata': result['metadata'],
                'relevance_score': result['final_score'],
                'scores': {
                    'initial': result['initial_score'],
                    'rerank': result['rerank_score'],
                    'final': result['final_score']
                }
            })
        
        retrieval_time = int((time.time() - start_time) * 1000)
        
        # Store result in Redis
        result = {
            'task_id': task['id'],
            'documents': documents,
            'retrieval_time_ms': retrieval_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            redis_client.setex(
                f"retrieval_result:{task['id']}", 
                60,
                json.dumps(result)
            )
        except:
            logger.warning("Failed to store retrieval result in Redis")
        
        # Log retrieval performance
        log_retrieval_performance(task['message'], len(documents), retrieval_time)
        
        logger.info(f"Retrieved {len(documents)} documents in {retrieval_time}ms")
        
    except Exception as e:
        logger.error(f"Chat task processing failed: {e}")
        # Store error result
        error_result = {
            'task_id': task['id'],
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
        try:
            redis_client.setex(
                f"retrieval_result:{task['id']}", 
                60,
                json.dumps(error_result)
            )
        except:
            logger.warning("Failed to store error result in Redis")


def process_embedding_task(task: Dict):
    """Process embedding computation task"""
    try:
        logger.info(f"Computing embeddings for message {task['message_id']}")
        
        # Compute embeddings for feedback/fine-tuning
        query_embedding = embedding_model.encode(task['query']).tolist()
        response_embedding = embedding_model.encode(task['response']).tolist()
        
        # Update MongoDB with embeddings
        db.messages.update_one(
            {'_id': ObjectId(task['message_id'])},
            {
                '$set': {
                    'query_embedding': query_embedding,
                    'response_embedding': response_embedding,
                    'embeddings_computed_at': datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Computed embeddings for message {task['message_id']}")
        
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")


def process_indexing_task(task: Dict):
    """Process document indexing task"""
    try:
        logger.info(f"Processing indexing task {task['id']} with {len(task.get('documents', []))} documents")
        
        success_count = 0
        for doc_data in task.get('documents', []):
            try:
                doc = Document(
                    id=doc_data['id'],
                    content=doc_data['content'],
                    metadata=doc_data.get('metadata', {})
                )
                chunks_created = rag_pipeline.process_document(doc)
                if chunks_created > 0:
                    success_count += 1
                logger.info(f"Indexed document {doc_data['id']} with {chunks_created} chunks")
            except Exception as e:
                logger.error(f"Error indexing document {doc_data['id']}: {e}")
                continue
        
        result = {
            'task_id': task['id'],
            'success': success_count == len(task.get('documents', [])),
            'document_count': success_count,
            'total_requested': len(task.get('documents', [])),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            redis_client.setex(
                f"indexing_result:{task['id']}", 
                60,
                json.dumps(result)
            )
        except:
            logger.warning("Failed to store indexing result in Redis")
        
        logger.info(f"Indexed {success_count}/{len(task.get('documents', []))} documents successfully")
        
    except Exception as e:
        logger.error(f"Indexing task failed: {e}")


def process_embedding_tasks():
    """
    Main loop to process embedding/retrieval tasks from Redis
    """
    logger.info("Advanced Embedding service started with MongoDB and ChromaDB v2")
    
    # Task type handlers
    task_handlers = {
        'chat': process_chat_task,
        'compute_embeddings': process_embedding_task,
        'process_file': process_uploaded_file,
        'index': process_indexing_task,
        'analyze_performance': lambda task: process_performance_analysis(task)
    }
    
    while True:
        try:
            # Check for multiple task types
            task_data = redis_client.brpop([
                'embedding_tasks', 
                'file_processing_tasks',
                'dataset_processing_tasks'
            ], timeout=1)
            
            if task_data:
                queue_name, task_json = task_data
                task = json.loads(task_json)
                
                logger.info(f"Processing task from {queue_name}: {task.get('id', 'unknown')} - Type: {task.get('type', 'unknown')}")
                
                # Route task to appropriate handler
                task_type = task.get('type')
                if task_type in task_handlers:
                    task_handlers[task_type](task)
                else:
                    logger.warning(f"Unknown task type: {task_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in task: {e}")
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down service...")
            break


def process_performance_analysis(task: Dict):
    """Process performance analysis task"""
    try:
        logger.info("Analyzing retrieval performance")
        
        # Analyze retrieval performance
        performance_metrics = analyze_retrieval_performance()
        
        result = {
            'task_id': task['id'],
            'metrics': performance_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            redis_client.setex(
                f"performance_analysis_result:{task['id']}",
                300,
                json.dumps(result)
            )
        except:
            logger.warning("Failed to store performance analysis in Redis")
        
        logger.info("Completed performance analysis")
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")


def analyze_retrieval_performance() -> Dict:
    """Analyze retrieval performance from stored metrics"""
    try:
        # Get recent retrieval metrics from MongoDB
        performance_data = rag_pipeline.get_performance_metrics(hours=24)
        
        # Get query patterns
        query_patterns = list(db.retrieval_logs.aggregate([
            {'$match': {'created_at': {'$gte': datetime.utcnow() - timedelta(hours=24)}}},
            {'$project': {
                'query_length': {'$strLenCP': '$query'},
                'response_time_ms': 1
            }},
            {'$sort': {'created_at': -1}},
            {'$limit': 1000}
        ]))
        
        # Analyze Redis metrics
        redis_metrics = []
        try:
            for i in range(min(100, redis_client.llen('retrieval_metrics') if hasattr(redis_client, 'llen') else 0)):
                metric_json = redis_client.lindex('retrieval_metrics', i)
                if metric_json:
                    redis_metrics.append(json.loads(metric_json))
        except:
            logger.warning("Failed to get Redis metrics")
        
        # Calculate advanced metrics
        metrics = {
            'database_metrics': performance_data,
            'query_patterns': {
                'avg_query_length': np.mean([p['query_length'] for p in query_patterns]) if query_patterns else 0,
                'query_length_vs_response_time_correlation': calculate_correlation(
                    [p['query_length'] for p in query_patterns],
                    [p['response_time_ms'] for p in query_patterns]
                ) if query_patterns else 0
            },
            'redis_metrics': {
                'total_queries': len(redis_metrics),
                'avg_initial_score': np.mean([m.get('avg_initial_score', 0) for m in redis_metrics]) if redis_metrics else 0,
                'avg_rerank_score': np.mean([m.get('avg_rerank_score', 0) for m in redis_metrics]) if redis_metrics else 0,
                'avg_final_score': np.mean([m.get('avg_final_score', 0) for m in redis_metrics]) if redis_metrics else 0,
                'chunk_type_distribution': get_chunk_type_distribution(redis_metrics)
            },
            'system_health': {
                'chroma_status': check_chroma_health(),
                'redis_status': check_redis_health(),
                'mongodb_status': check_mongodb_health(),
                'embedding_model_status': check_embedding_model_health()
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error analyzing performance: {e}")
        return {'error': str(e)}


def calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
    """Calculate correlation coefficient between two lists"""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    
    try:
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        correlation_matrix = np.corrcoef(x_array, y_array)
        return float(correlation_matrix[0, 1])
    except:
        return 0.0


def get_chunk_type_distribution(metrics: List[Dict]) -> Dict[str, int]:
    """Get distribution of chunk types from metrics"""
    chunk_type_counts = {}
    for metric in metrics:
        for chunk_type in metric.get('chunk_types', []):
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
    return chunk_type_counts


def check_chroma_health() -> bool:
    """Check ChromaDB health"""
    try:
        return rag_pipeline.retriever.health_check() if rag_pipeline else False
    except:
        return False


def check_redis_health() -> bool:
    """Check Redis health"""
    try:
        redis_client.ping()
        return True
    except:
        return False


def check_mongodb_health() -> bool:
    """Check MongoDB health"""
    try:
        db.command('ping')
        return True
    except:
        return False


def check_embedding_model_health() -> bool:
    """Check embedding model health"""
    try:
        test_embedding = embedding_model.encode("test")
        return len(test_embedding) > 0
    except:
        return False


def get_total_indexed_documents() -> int:
    """Get total number of indexed documents"""
    try:
        return db.indexed_documents.count_documents({})
    except:
        return 0


def get_total_chunks() -> int:
    """Get total number of chunks"""
    try:
        return db.document_chunks.count_documents({})
    except:
        return 0


def get_recent_query_count() -> int:
    """Get count of queries in last 24 hours"""
    try:
        from_date = datetime.utcnow() - timedelta(hours=24)
        return db.retrieval_logs.count_documents({'created_at': {'$gte': from_date}})
    except:
        return 0


def health_check() -> Dict:
    """Comprehensive health check"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'chroma': check_chroma_health(),
            'redis': check_redis_health(),
            'mongodb': check_mongodb_health(),
            'embedding_model': check_embedding_model_health()
        },
        'metrics': {
            'total_indexed_documents': get_total_indexed_documents(),
            'total_chunks': get_total_chunks(),
            'recent_queries': get_recent_query_count()
        },
        'configuration': {
            'embedding_model': EMBEDDING_MODEL,
            'reranker_model': RERANKER_MODEL,
            'chroma_persistent': CHROMA_PERSISTENT,
            'mongo_database': MONGO_DATABASE
        }
    }


def cleanup_services():
    """Cleanup services on shutdown"""
    global mongo_client, redis_client
    
    logger.info("Cleaning up services...")
    
    try:
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed")
    except:
        pass
    
    try:
        if redis_client and hasattr(redis_client, 'close'):
            redis_client.close()
            logger.info("Redis connection closed")
    except:
        pass


if __name__ == "__main__":
    try:
        # Initialize all services
        initialize_services()
        
        # Perform health check
        health_status = health_check()
        logger.info(f"Health check: {health_status}")
        
        # Start the main processing loop
        process_embedding_tasks()
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed to start: {e}")
        raise
    finally:
        cleanup_services()
        logger.info("Service shutdown complete")