"""
Advanced RAG System with Hybrid Retrieval and Reranking - MongoDB Version (Fixed for ChromaDB v2)
Implements semantic chunking, BM25, vector search, and cross-encoder reranking
"""

import os
import json
import re
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
from rank_bm25 import BM25Okapi
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import DuplicateKeyError
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with metadata and structure"""
    id: str
    content: str
    metadata: Dict
    chunks: List['Chunk'] = None
    
    def to_dict(self):
        """Convert to dictionary for MongoDB storage"""
        return {
            '_id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
    
@dataclass
class Chunk:
    """Document chunk with semantic boundaries"""
    id: str
    document_id: str
    content: str
    start_idx: int
    end_idx: int
    chunk_type: str  # paragraph, section, list, etc.
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        """Convert to dictionary for MongoDB storage"""
        return {
            '_id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'chunk_type': self.chunk_type,
            'metadata': self.metadata,
            'created_at': datetime.utcnow()
        }


class SemanticChunker:
    """Chunking that preserves document structure and semantic boundaries"""
    
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 512,
                 overlap_size: int = 50):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk document preserving semantic boundaries"""
        chunks = []
        
        # Extract structure from document
        sections = self._extract_sections(document.content)
        
        chunk_id = 0
        for section in sections:
            # Process each section
            section_chunks = self._chunk_section(
                section['content'],
                section['type'],
                document.id,
                chunk_id,
                section.get('metadata', {})
            )
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract document sections based on structure"""
        sections = []
        
        # Patterns for different section types
        patterns = {
            'heading': r'^#{1,6}\s+(.+)$',
            'bullet_list': r'^[\*\-\+]\s+(.+)$',
            'numbered_list': r'^\d+\.\s+(.+)$',
            'code_block': r'```[\s\S]*?```',
            'table': r'\|.*\|.*\|',
        }
        
        lines = text.split('\n')
        current_section = {'content': '', 'type': 'paragraph', 'metadata': {}}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for section boundaries
            section_type = None
            for pattern_type, pattern in patterns.items():
                if re.match(pattern, line):
                    section_type = pattern_type
                    break
            
            if section_type and current_section['content']:
                # Save current section
                sections.append(current_section)
                current_section = {
                    'content': line,
                    'type': section_type,
                    'metadata': {'line_number': i}
                }
            else:
                # Continue building current section
                if current_section['content']:
                    current_section['content'] += '\n'
                current_section['content'] += line
            
            i += 1
        
        # Don't forget the last section
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, 
                      text: str, 
                      section_type: str,
                      document_id: str,
                      start_chunk_id: int,
                      metadata: Dict) -> List[Chunk]:
        """Chunk a section while preserving semantic units"""
        chunks = []
        
        # Use spaCy for sentence segmentation
        doc = self.nlp(text[:1000000])  # Limit for performance
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if not sentences:
            return chunks
        
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # Check if adding this sentence exceeds max size
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = Chunk(
                    id=f"{document_id}_chunk_{start_chunk_id + len(chunks)}",
                    document_id=document_id,
                    content=chunk_text,
                    start_idx=text.find(current_chunk[0]),
                    end_idx=text.find(current_chunk[-1]) + len(current_chunk[-1]),
                    chunk_type=section_type,
                    metadata={**metadata, 'sentence_count': len(current_chunk)}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-(self.overlap_size // 20):]
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle remaining content
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunk = Chunk(
                id=f"{document_id}_chunk_{start_chunk_id + len(chunks)}",
                document_id=document_id,
                content=chunk_text,
                start_idx=text.find(current_chunk[0]),
                end_idx=text.find(current_chunk[-1]) + len(current_chunk[-1]),
                chunk_type=section_type,
                metadata={**metadata, 'sentence_count': len(current_chunk)}
            )
            chunks.append(chunk)
        
        return chunks


def safe_chroma_operation(operation, *args, max_retries=3, **kwargs):
    """Wrapper for safe ChromaDB operations with retry logic"""
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"ChromaDB operation failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Wait before retry


class HybridRetriever:
    """Retrieval combining dense vectors and BM25"""
    
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 chroma_client: chromadb.Client = None,
                 collection_name: str = "documents"):
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        
        # Initialize stores with better error handling
        self.chroma_client = chroma_client or self._init_default_client()
        self.collection = self._get_or_create_collection(collection_name)
        
        # BM25 components
        self.bm25_corpus = []
        self.bm25_index = None
        self.chunk_id_mapping = {}
        
        logger.info("HybridRetriever initialized successfully")
    
    def _init_default_client(self):
        """Initialize default ChromaDB client with v2 compatibility"""
        try:
            # Try persistent client first (recommended for production)
            client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Initialized ChromaDB PersistentClient")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize PersistentClient: {e}")
            # Fallback to in-memory client
            client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("Initialized ChromaDB in-memory Client")
            return client
    
    def _get_or_create_collection(self, name: str):
        """Get or create ChromaDB collection with v2 API compatibility"""
        try:
            # Try to get existing collection first
            collection = safe_chroma_operation(self.chroma_client.get_collection, name)
            logger.info(f"Retrieved existing collection: {name}")
            return collection
        except Exception as get_error:
            logger.info(f"Collection {name} not found, creating new one")
            try:
                # Create new collection with SentenceTransformer embedding function
                sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                
                collection = safe_chroma_operation(
                    self.chroma_client.create_collection,
                    name=name,
                    embedding_function=sentence_transformer_ef,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {name}")
                return collection
                
            except Exception as create_error:
                logger.warning(f"Failed to create collection: {create_error}")
                # Try delete and recreate as last resort
                try:
                    safe_chroma_operation(self.chroma_client.delete_collection, name)
                    collection = safe_chroma_operation(
                        self.chroma_client.create_collection,
                        name=name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Recreated collection: {name}")
                    return collection
                except Exception as recreate_error:
                    logger.error(f"Failed to recreate collection: {recreate_error}")
                    raise recreate_error
    
    def index_chunks(self, chunks: List[Chunk]):
        """Index chunks for both vector and BM25 search"""
        
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        # Prepare batch data
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []
        
        for chunk in chunks:
            # Prepare for ChromaDB
            chunk_ids.append(chunk.id)
            chunk_texts.append(chunk.content)
            chunk_metadatas.append({
                'document_id': chunk.document_id,
                'chunk_type': chunk.chunk_type,
                'start_idx': chunk.start_idx,
                'end_idx': chunk.end_idx,
                **chunk.metadata
            })
            
            # Add to BM25 corpus
            self.bm25_corpus.append(chunk.content)
            self.chunk_id_mapping[len(self.bm25_corpus) - 1] = chunk.id
        
        # Batch insert to ChromaDB
        if chunk_ids:
            try:
                safe_chroma_operation(
                    self.collection.add,
                    ids=chunk_ids,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas
                )
                logger.info(f"Successfully indexed {len(chunks)} chunks in ChromaDB")
            except Exception as e:
                logger.error(f"Failed to index chunks in ChromaDB: {e}")
                # Continue with BM25 indexing even if ChromaDB fails
        
        # Rebuild BM25 index
        self._rebuild_bm25_index()
        
        logger.info(f"Indexed {len(chunks)} chunks total")
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from corpus"""
        if self.bm25_corpus:
            try:
                tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
                self.bm25_index = BM25Okapi(tokenized_corpus)
                logger.info(f"Rebuilt BM25 index with {len(self.bm25_corpus)} documents")
            except Exception as e:
                logger.error(f"Failed to rebuild BM25 index: {e}")
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 20,
                     vector_weight: float = 0.7,
                     filters: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """Perform hybrid search combining vector and BM25"""
        
        combined_results = {}
        
        # Vector search with ChromaDB
        try:
            vector_results = safe_chroma_operation(
                self.collection.query,
                query_texts=[query],
                n_results=k,
                where=filters
            )
            
            # Add vector search results
            if vector_results and vector_results.get('ids') and vector_results['ids'][0]:
                for i, chunk_id in enumerate(vector_results['ids'][0]):
                    distance = vector_results['distances'][0][i] if vector_results.get('distances') else 0
                    score = max(0, 1.0 - distance)  # Convert distance to similarity
                    combined_results[chunk_id] = {
                        'vector_score': score * vector_weight,
                        'bm25_score': 0,
                        'content': vector_results['documents'][0][i],
                        'metadata': vector_results['metadatas'][0][i] if vector_results.get('metadatas') else {}
                    }
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # BM25 search
        try:
            if self.bm25_index and query.strip():
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(tokenized_query)
                
                # Get top-k BM25 results
                if len(bm25_scores) > 0:
                    bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]
                    max_score = max(bm25_scores) if bm25_scores else 1
                    
                    for idx in bm25_top_indices:
                        if idx in self.chunk_id_mapping and bm25_scores[idx] > 0:
                            chunk_id = self.chunk_id_mapping[idx]
                            bm25_score = bm25_scores[idx] / (max_score + 1e-6)  # Normalize
                            
                            if chunk_id in combined_results:
                                combined_results[chunk_id]['bm25_score'] = bm25_score * (1 - vector_weight)
                            else:
                                # Fetch content from ChromaDB
                                try:
                                    chunk_data = safe_chroma_operation(
                                        self.collection.get,
                                        ids=[chunk_id]
                                    )
                                    if chunk_data and chunk_data.get('ids'):
                                        combined_results[chunk_id] = {
                                            'vector_score': 0,
                                            'bm25_score': bm25_score * (1 - vector_weight),
                                            'content': chunk_data['documents'][0],
                                            'metadata': chunk_data.get('metadatas', [{}])[0]
                                        }
                                except Exception as e:
                                    logger.warning(f"Failed to fetch chunk {chunk_id}: {e}")
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
        
        # Calculate final scores and prepare results
        results = []
        for chunk_id, data in combined_results.items():
            final_score = data['vector_score'] + data['bm25_score']
            results.append((chunk_id, final_score, data))
        
        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def rerank(self, query: str, candidates: List[Tuple[str, float, Dict]], top_k: int = 5) -> List[Dict]:
        """Rerank candidates using cross-encoder"""
        
        if not candidates:
            return []
        
        try:
            # Prepare texts for reranking
            texts = [candidate[2]['content'] for candidate in candidates]
            
            # Score with cross-encoder
            scores = self.reranker.predict([(query, text) for text in texts])
            
            # Combine with initial scores (weighted)
            reranked_results = []
            for i, (chunk_id, initial_score, data) in enumerate(candidates):
                rerank_score = float(scores[i])
                # Weighted combination: 70% reranker, 30% initial
                final_score = 0.7 * rerank_score + 0.3 * initial_score
                
                reranked_results.append({
                    'chunk_id': chunk_id,
                    'content': data['content'],
                    'metadata': data['metadata'],
                    'initial_score': initial_score,
                    'rerank_score': rerank_score,
                    'final_score': final_score
                })
            
            # Sort by final score
            reranked_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original results without reranking
            fallback_results = []
            for chunk_id, initial_score, data in candidates[:top_k]:
                fallback_results.append({
                    'chunk_id': chunk_id,
                    'content': data['content'],
                    'metadata': data['metadata'],
                    'initial_score': initial_score,
                    'rerank_score': initial_score,
                    'final_score': initial_score
                })
            return fallback_results
    
    def health_check(self) -> bool:
        """Check ChromaDB health with v2 API compatibility"""
        try:
            # Try multiple methods to check health
            if hasattr(self.chroma_client, 'heartbeat'):
                safe_chroma_operation(self.chroma_client.heartbeat)
            else:
                # Alternative: list collections
                safe_chroma_operation(self.chroma_client.list_collections)
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False


class AdvancedRAGPipeline:
    """Complete RAG pipeline with advanced retrieval - MongoDB version"""
    
    def __init__(self,
                 mongo_uri: str,
                 database_name: str,
                 redis_client: redis.Redis,
                 chroma_client: chromadb.Client):
        
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[database_name]
        self.redis_client = redis_client
        
        # Initialize collections
        self._init_collections()
        
        # Initialize components
        self.chunker = SemanticChunker()
        self.retriever = HybridRetriever(chroma_client=chroma_client)
        
        logger.info("Advanced RAG pipeline initialized with MongoDB")
    
    def _init_collections(self):
        """Initialize MongoDB collections and indexes"""
        try:
            # Documents collection
            self.documents_collection = self.db['indexed_documents']
            self.documents_collection.create_index([('title', TEXT), ('content', TEXT)])
            self.documents_collection.create_index('source')
            self.documents_collection.create_index('indexed_at')
            
            # Chunks collection
            self.chunks_collection = self.db['document_chunks']
            self.chunks_collection.create_index('document_id')
            self.chunks_collection.create_index('chunk_type')
            self.chunks_collection.create_index([('content', TEXT)])
            self.chunks_collection.create_index('created_at')
            
            # Retrieval logs collection
            self.retrieval_logs_collection = self.db['retrieval_logs']
            self.retrieval_logs_collection.create_index('created_at')
            
            # Messages collection for embeddings
            self.messages_collection = self.db['messages']
            self.messages_collection.create_index('created_at')
            
            logger.info("MongoDB collections and indexes initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB collections: {e}")
            raise
    
    def process_document(self, document: Document) -> int:
        """Process and index a document"""
        
        try:
            # Semantic chunking
            chunks = self.chunker.chunk_document(document)
            logger.info(f"Created {len(chunks)} chunks for document {document.id}")
            
            # Index chunks
            self.retriever.index_chunks(chunks)
            
            # Store in MongoDB
            self._store_chunks_mongodb(document, chunks)
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process document {document.id}: {e}")
            raise
    
    def _store_chunks_mongodb(self, document: Document, chunks: List[Chunk]):
        """Store document and chunks in MongoDB"""
        try:
            # Store document metadata
            doc_data = document.to_dict()
            doc_data.update({
                'title': document.metadata.get('title', 'Untitled'),
                'source': document.metadata.get('source', 'unknown'),
                'chunk_count': len(chunks),
                'indexed_at': datetime.utcnow()
            })
            
            self.documents_collection.replace_one(
                {'_id': document.id},
                doc_data,
                upsert=True
            )
            
            # Store chunks
            if chunks:
                chunk_docs = [chunk.to_dict() for chunk in chunks]
                # Use bulk operations for better performance
                operations = []
                for chunk_doc in chunk_docs:
                    operations.append({
                        'replaceOne': {
                            'filter': {'_id': chunk_doc['_id']},
                            'replacement': chunk_doc,
                            'upsert': True
                        }
                    })
                
                if operations:
                    self.chunks_collection.bulk_write(operations)
            
            logger.info(f"Stored document {document.id} with {len(chunks)} chunks in MongoDB")
            
        except Exception as e:
            logger.error(f"Error storing document in MongoDB: {e}")
            raise
    
    def retrieve(self, 
                query: str,
                k: int = 5,
                filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        
        try:
            # Hybrid search
            candidates = self.retriever.hybrid_search(
                query=query,
                k=20,  # Get more candidates for reranking
                filters=filters
            )
            
            # Rerank
            reranked_results = self.retriever.rerank(
                query=query,
                candidates=candidates,
                top_k=k
            )
            
            # Log retrieval metrics
            self._log_retrieval_metrics(query, reranked_results)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return []
    
    def _log_retrieval_metrics(self, query: str, results: List[Dict]):
        """Log retrieval metrics for analysis"""
        try:
            metrics = {
                'query': query,
                'num_results': len(results),
                'avg_initial_score': np.mean([r['initial_score'] for r in results]) if results else 0,
                'avg_rerank_score': np.mean([r['rerank_score'] for r in results]) if results else 0,
                'avg_final_score': np.mean([r['final_score'] for r in results]) if results else 0,
                'chunk_types': [r['metadata'].get('chunk_type', 'unknown') for r in results],
                'created_at': datetime.utcnow()
            }
            
            # Store in MongoDB
            self.retrieval_logs_collection.insert_one(metrics)
            
            # Also store in Redis for real-time monitoring
            self.redis_client.lpush('retrieval_metrics', json.dumps(metrics, default=str))
            self.redis_client.ltrim('retrieval_metrics', 0, 1000)  # Keep last 1000
            
        except Exception as e:
            logger.warning(f"Failed to log retrieval metrics: {e}")
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """Retrieve a document by ID"""
        try:
            return self.documents_collection.find_one({'_id': document_id})
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def search_documents(self, text_query: str, limit: int = 10) -> List[Dict]:
        """Full-text search on documents"""
        try:
            results = self.documents_collection.find(
                {'$text': {'$search': text_query}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit)
            
            return list(results)
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    def get_performance_metrics(self, hours: int = 24) -> Dict:
        """Get performance metrics for the last N hours"""
        try:
            from_date = datetime.utcnow() - timedelta(hours=hours)
            
            pipeline = [
                {'$match': {'created_at': {'$gte': from_date}}},
                {'$group': {
                    '_id': None,
                    'total_queries': {'$sum': 1},
                    'avg_results': {'$avg': '$num_results'},
                    'avg_initial_score': {'$avg': '$avg_initial_score'},
                    'avg_rerank_score': {'$avg': '$avg_rerank_score'},
                    'avg_final_score': {'$avg': '$avg_final_score'}
                }}
            ]
            
            result = list(self.retrieval_logs_collection.aggregate(pipeline))
            return result[0] if result else {}
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}


# MongoDB connection helper
def get_mongodb_connection(uri: str, database: str) -> Tuple[MongoClient, object]:
    """Get MongoDB connection"""
    try:
        client = MongoClient(uri)
        db = client[database]
        # Test connection
        db.command('ping')
        logger.info("MongoDB connection established")
        return client, db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


def initialize_chroma_client(host: str = "localhost", port: int = 8000, use_persistent: bool = True) -> chromadb.Client:
    """Initialize ChromaDB client with v2 compatibility"""
    try:
        if use_persistent:
            # Use persistent client (recommended for production)
            client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Initialized ChromaDB PersistentClient")
        else:
            # Use HTTP client for distributed setup
            client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            logger.info(f"Initialized ChromaDB HttpClient at {host}:{port}")
        
        # Test the connection
        collections = client.list_collections()
        logger.info(f"ChromaDB connection successful. Found {len(collections)} collections")
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        # Fallback to in-memory client
        logger.info("Falling back to in-memory ChromaDB client")
        return chromadb.Client(settings=Settings(anonymized_telemetry=False))