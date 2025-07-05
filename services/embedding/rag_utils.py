"""
Advanced RAG System with Hybrid Retrieval and Reranking - Elasticsearch + MongoDB Version
Implements semantic chunking, Elasticsearch BM25, ChromaDB vector search, and cross-encoder reranking
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
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import DuplicateKeyError
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import redis
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, NotFoundError

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
    group_id: str  # Added for group-based filtering
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            '_id': self.id,
            'document_id': self.document_id,
            'content': self.content,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'chunk_type': self.chunk_type,
            'metadata': self.metadata,
            'group_id': self.group_id,
            'created_at': datetime.utcnow()
        }
    
    def to_elasticsearch_doc(self):
        """Convert to Elasticsearch document format"""
        return {
            'document_id': self.document_id,
            'content': self.content,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'chunk_type': self.chunk_type,
            'group_id': self.group_id,
            'metadata': self.metadata,
            'created_at': datetime.utcnow().isoformat()
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
    
    def chunk_document(self, document: Document, group_id: str = "general") -> List[Chunk]:
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
                section.get('metadata', {}),
                group_id
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
            'list': r'^[\*\-\+]\s+(.+)$',  
            'list': r'^\d+\.\s+(.+)$',    
            'code': r'```[\s\S]*?```',  
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
                    metadata: Dict,
                    group_id: str) -> List[Chunk]:
        """Chunk a section while preserving semantic units"""
        chunks = []
        
        # Map section types to valid MongoDB enum values
        valid_chunk_types = {
            'bullet_list': 'list',
            'numbered_list': 'list', 
            'code_block': 'code',
            'paragraph': 'paragraph',
            'heading': 'heading',
            'table': 'table'
        }
        
        # Ensure chunk_type is valid for MongoDB schema
        chunk_type = valid_chunk_types.get(section_type, 'other')
        
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
                    chunk_type=chunk_type,
                    metadata={**metadata, 'sentence_count': len(current_chunk)},
                    group_id=group_id
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
                chunk_type=chunk_type,
                metadata={**metadata, 'sentence_count': len(current_chunk)},
                group_id=group_id
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


def safe_elasticsearch_operation(operation, *args, max_retries=3, **kwargs):
    """Wrapper for safe Elasticsearch operations with retry logic"""
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Elasticsearch operation failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Wait before retry


class ElasticsearchBM25Retriever:
    """BM25 retrieval using Elasticsearch with group-based filtering"""
    
    def __init__(self, 
                 es_client: Elasticsearch,
                 index_name: str = "rag_chunks"):
        self.es_client = es_client
        self.index_name = index_name
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create Elasticsearch index if it doesn't exist"""
        try:
            if not safe_elasticsearch_operation(self.es_client.indices.exists, index=self.index_name):
                # Define mapping with proper text analysis for BM25
                mapping = {
                    "mappings": {
                        "properties": {
                            "document_id": {"type": "keyword"},
                            "content": {
                                "type": "text",
                                "analyzer": "standard",
                                "fields": {
                                    "keyword": {"type": "keyword"}
                                }
                            },
                            "start_idx": {"type": "integer"},
                            "end_idx": {"type": "integer"},
                            "chunk_type": {"type": "keyword"},
                            "group_id": {"type": "keyword"},
                            "metadata": {"type": "object"},
                            "created_at": {"type": "date"}
                        }
                    },
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "bm25_analyzer": {
                                    "type": "standard",
                                    "stopwords": "_english_"
                                }
                            }
                        }
                    }
                }
                
                safe_elasticsearch_operation(
                    self.es_client.indices.create,
                    index=self.index_name,
                    body=mapping
                )
                logger.info(f"Created Elasticsearch index: {self.index_name}")
            else:
                logger.info(f"Elasticsearch index {self.index_name} already exists")
        except Exception as e:
            logger.error(f"Failed to ensure Elasticsearch index exists: {e}")
            raise
    
    def index_chunks(self, chunks: List[Chunk]):
        """Index chunks in Elasticsearch"""
        if not chunks:
            logger.warning("No chunks to index in Elasticsearch")
            return
        
        try:
            # Prepare bulk operations
            bulk_operations = []
            for chunk in chunks:
                # Index operation
                bulk_operations.append({
                    "index": {
                        "_index": self.index_name,
                        "_id": chunk.id
                    }
                })
                # Document data
                bulk_operations.append(chunk.to_elasticsearch_doc())
            
            # Execute bulk indexing
            if bulk_operations:
                response = safe_elasticsearch_operation(
                    self.es_client.bulk,
                    body=bulk_operations,
                    refresh=True
                )
                
                # Check for errors
                if response.get('errors'):
                    error_count = sum(1 for item in response['items'] if 'error' in item.get('index', {}))
                    logger.warning(f"Elasticsearch bulk indexing had {error_count} errors")
                
                logger.info(f"Successfully indexed {len(chunks)} chunks in Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to index chunks in Elasticsearch: {e}")
            raise
    
    def search(self, query: str, group_id: Optional[str] = None, k: int = 20) -> List[Tuple[str, float, Dict]]:
        """Perform BM25 search with group filtering"""
        try:
            # Build query
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "analyzer": "standard"
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": k,
                "_source": True
            }
            
            # Add group filter if specified
            if group_id and group_id != "general":
                search_query["query"]["bool"]["filter"] = [
                    {"term": {"group_id": group_id}}
                ]
            
            # Execute search
            response = safe_elasticsearch_operation(
                self.es_client.search,
                index=self.index_name,
                body=search_query
            )
            
            # Process results
            results = []
            max_score = response.get('hits', {}).get('max_score', 1.0) or 1.0
            
            for hit in response.get('hits', {}).get('hits', []):
                chunk_id = hit['_id']
                score = hit['_score'] / max_score  # Normalize score
                source = hit['_source']
                
                # Prepare metadata
                metadata = {
                    'document_id': source.get('document_id'),
                    'chunk_type': source.get('chunk_type'),
                    'group_id': source.get('group_id'),
                    'start_idx': source.get('start_idx'),
                    'end_idx': source.get('end_idx'),
                    **source.get('metadata', {})
                }
                
                data = {
                    'content': source.get('content', ''),
                    'metadata': metadata
                }
                
                results.append((chunk_id, score, data))
            
            logger.info(f"Elasticsearch BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    def delete_by_group(self, group_id: str):
        """Delete all chunks for a specific group"""
        try:
            delete_query = {
                "query": {
                    "term": {"group_id": group_id}
                }
            }
            
            response = safe_elasticsearch_operation(
                self.es_client.delete_by_query,
                index=self.index_name,
                body=delete_query,
                refresh=True
            )
            
            deleted_count = response.get('deleted', 0)
            logger.info(f"Deleted {deleted_count} chunks for group {group_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for group {group_id}: {e}")
            return 0
    
    def health_check(self) -> bool:
        """Check Elasticsearch health"""
        try:
            health = safe_elasticsearch_operation(self.es_client.cluster.health)
            return health.get('status') in ['green', 'yellow']
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False


class HybridRetriever:
    """Retrieval combining dense vectors (ChromaDB) and BM25 (Elasticsearch)"""
    
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 chroma_client: chromadb.Client = None,
                 es_client: Elasticsearch = None,
                 collection_name: str = "documents"):
        
        # Initialize models
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.reranker = CrossEncoder(self.reranker_model_name)
        
        # Initialize stores with better error handling
        self.chroma_client = chroma_client or self._init_default_chroma_client()
        self.collection = self._get_or_create_collection(collection_name)
        
        # Initialize Elasticsearch BM25 retriever
        self.es_retriever = ElasticsearchBM25Retriever(es_client) if es_client else None
        
        logger.info("HybridRetriever initialized successfully with Elasticsearch")
    
    def _init_default_chroma_client(self):
        """Initialize default ChromaDB client with v2 compatibility"""
        try:
            # Try persistent client first (recommended for production)
            client = chromadb.PersistentClient(
                path="/app_data/chroma_db",
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
            logger.info(f"Retrieved existing ChromaDB collection: {name}")
            return collection
        except Exception as get_error:
            logger.info(f"ChromaDB collection {name} not found, creating new one")
            try:
                # Create new collection with SentenceTransformer embedding function
                sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
                
                collection = safe_chroma_operation(
                    self.chroma_client.create_collection,
                    name=name,
                    embedding_function=sentence_transformer_ef,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new ChromaDB collection: {name}")
                return collection
                
            except Exception as create_error:
                logger.warning(f"Failed to create ChromaDB collection: {create_error}")
                # Try delete and recreate as last resort
                try:
                    safe_chroma_operation(self.chroma_client.delete_collection, name)
                    collection = safe_chroma_operation(
                        self.chroma_client.create_collection,
                        name=name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Recreated ChromaDB collection: {name}")
                    return collection
                except Exception as recreate_error:
                    logger.error(f"Failed to recreate ChromaDB collection: {recreate_error}")
                    raise recreate_error
    
    def index_chunks(self, chunks: List[Chunk]):
        """Index chunks for both vector (ChromaDB) and BM25 (Elasticsearch) search"""
        
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        # Prepare batch data for ChromaDB
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
                'group_id': chunk.group_id,
                **chunk.metadata
            })
        
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
        
        # Index in Elasticsearch for BM25
        if self.es_retriever:
            try:
                self.es_retriever.index_chunks(chunks)
                logger.info(f"Successfully indexed {len(chunks)} chunks in Elasticsearch")
            except Exception as e:
                logger.error(f"Failed to index chunks in Elasticsearch: {e}")
        
        logger.info(f"Indexed {len(chunks)} chunks total")
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 20,
                     vector_weight: float = 0.7,
                     group_id: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """Perform hybrid search combining vector and BM25 with group filtering"""
        
        combined_results = {}
        
        # Build ChromaDB filters for group
        chroma_filters = None
        if group_id and group_id != "general":
            chroma_filters = {"group_id": group_id}
        
        # Vector search with ChromaDB
        try:
            vector_results = safe_chroma_operation(
                self.collection.query,
                query_texts=[query],
                n_results=k,
                where=chroma_filters
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
            logger.error(f"ChromaDB vector search failed: {e}")
        
        # BM25 search with Elasticsearch
        try:
            if self.es_retriever and query.strip():
                bm25_results = self.es_retriever.search(
                    query=query,
                    group_id=group_id,
                    k=k
                )
                
                for chunk_id, bm25_score, data in bm25_results:
                    if chunk_id in combined_results:
                        combined_results[chunk_id]['bm25_score'] = bm25_score * (1 - vector_weight)
                    else:
                        combined_results[chunk_id] = {
                            'vector_score': 0,
                            'bm25_score': bm25_score * (1 - vector_weight),
                            'content': data['content'],
                            'metadata': data['metadata']
                        }
        except Exception as e:
            logger.error(f"Elasticsearch BM25 search failed: {e}")
        
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
    
    def delete_by_group(self, group_id: str):
        """Delete all chunks for a specific group from both stores"""
        deleted_counts = {}
        
        # Delete from Elasticsearch
        if self.es_retriever:
            try:
                es_deleted = self.es_retriever.delete_by_group(group_id)
                deleted_counts['elasticsearch'] = es_deleted
            except Exception as e:
                logger.error(f"Failed to delete from Elasticsearch: {e}")
                deleted_counts['elasticsearch'] = 0
        
        # Delete from ChromaDB (more complex since ChromaDB doesn't have delete_by_query)
        try:
            # First, find all documents with the group_id
            chroma_results = safe_chroma_operation(
                self.collection.get,
                where={"group_id": group_id}
            )
            
            if chroma_results and chroma_results.get('ids'):
                chunk_ids = chroma_results['ids']
                safe_chroma_operation(
                    self.collection.delete,
                    ids=chunk_ids
                )
                deleted_counts['chromadb'] = len(chunk_ids)
            else:
                deleted_counts['chromadb'] = 0
                
        except Exception as e:
            logger.error(f"Failed to delete from ChromaDB: {e}")
            deleted_counts['chromadb'] = 0
        
        logger.info(f"Deleted chunks for group {group_id}: {deleted_counts}")
        return deleted_counts
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of both ChromaDB and Elasticsearch"""
        health_status = {}
        
        # Check ChromaDB
        try:
            if hasattr(self.chroma_client, 'heartbeat'):
                safe_chroma_operation(self.chroma_client.heartbeat)
            else:
                safe_chroma_operation(self.chroma_client.list_collections)
            health_status['chromadb'] = True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            health_status['chromadb'] = False
        
        # Check Elasticsearch
        if self.es_retriever:
            health_status['elasticsearch'] = self.es_retriever.health_check()
        else:
            health_status['elasticsearch'] = False
        
        return health_status


class AdvancedRAGPipeline:
    """Complete RAG pipeline with advanced retrieval - MongoDB + Elasticsearch version"""
    
    def __init__(self,
                 mongo_uri: str,
                 database_name: str,
                 redis_client: redis.Redis,
                 chroma_client: chromadb.Client,
                 es_client: Elasticsearch):
        
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[database_name]
        self.redis_client = redis_client
        self.es_client = es_client
        
        # Initialize collections
        self._init_collections()
        
        # Initialize components
        self.chunker = SemanticChunker()
        self.retriever = HybridRetriever(
            chroma_client=chroma_client,
            es_client=es_client,
        )
        logger.info("Advanced RAG pipeline initialized with MongoDB and Elasticsearch")
    
    def _init_collections(self):
        """Initialize MongoDB collections and indexes"""
        try:
            # Documents collection
            self.documents_collection = self.db['indexed_documents']
            self.documents_collection.create_index([('title', TEXT), ('content', TEXT)])
            self.documents_collection.create_index('source')
            self.documents_collection.create_index('indexed_at')
            self.documents_collection.create_index('group_id')  # Added group_id index
            

            # Retrieval logs collection
            self.retrieval_logs_collection = self.db['retrieval_logs']
            self.retrieval_logs_collection.create_index('created_at')
            self.retrieval_logs_collection.create_index('group_id')  # Added group_id index
            
            # Messages collection for embeddings
            self.messages_collection = self.db['messages']
            self.messages_collection.create_index('created_at')
            self.messages_collection.create_index('group_id')  # Added group_id index
            
            logger.info("MongoDB collections and indexes initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB collections: {e}")
            raise
    
    def process_document(self, document: Document, group_id: str = "general") -> int:
        """Process and index a document with group association"""
        
        try:
            # Semantic chunking with group_id
            chunks = self.chunker.chunk_document(document, group_id)
            logger.info(f"Created {len(chunks)} chunks for document {document.id} in group {group_id}")
            
            # Index chunks
            self.retriever.index_chunks(chunks)
            
            # Store in MongoDB
            self._store_chunks_mongodb(document, chunks, group_id)
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process document {document.id}: {e}")
            raise

    def _store_indexing_status_mongodb(self, document: Document, chunks: List[Chunk], group_id: str):
        """Store document  in MongoDB with group association"""
        try:
            # Store document metadata
            doc_data = document.to_dict()
            doc_data.update({
                'title': document.metadata.get('title', 'Untitled'),
                'source': document.metadata.get('source', 'unknown'),
                'chunk_count': len(chunks),
                'group_id': group_id,  # Add group_id to document
                'indexed_at': datetime.utcnow()
            })
            
            self.documents_collection.replace_one(
                {'_id': document.id},
                doc_data,
                upsert=True
            )
            
            logger.info(f"Stored document {document.id} with {len(chunks)} chunks in MongoDB for group {group_id}")
            
        except Exception as e:
            logger.error(f"Error storing document in MongoDB: {e}")
            raise
    
    def retrieve(self, 
                query: str,
                k: int = 5,
                group_id: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant chunks for a query with group filtering"""
        
        try:
            # Hybrid search with group filtering
            candidates = self.retriever.hybrid_search(
                query=query,
                k=20,  # Get more candidates for reranking
                group_id=group_id
            )
            
            # Rerank
            reranked_results =  self.retriever.rerank(
                query=query,
                candidates=candidates,
                top_k=k
            )
            
            # Log retrieval metrics
            self._log_retrieval_metrics(query, reranked_results, group_id)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}' in group '{group_id}': {e}")
            return []
    
    def _log_retrieval_metrics(self, query: str, results: List[Dict], group_id: Optional[str] = None):
        """Log retrieval metrics for analysis with group information"""
        try:
            metrics = {
                'query': query,
                'group_id': group_id or 'general',
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
    
    def get_document_by_id(self, document_id: str, group_id: Optional[str] = None) -> Optional[Dict]:
        """Retrieve a document by ID with optional group filtering"""
        try:
            query_filter = {'_id': document_id}
            if group_id and group_id != "general":
                query_filter['group_id'] = group_id
            return self.documents_collection.find_one(query_filter)
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def search_documents(self, text_query: str, group_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Full-text search on documents with group filtering"""
        try:
            query_filter = {'$text': {'$search': text_query}}
            if group_id and group_id != "general":
                query_filter['group_id'] = group_id
            
            results = self.documents_collection.find(
                query_filter,
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit)
            
            return list(results)
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    def delete_group_data(self, group_id: str) -> Dict[str, int]:
        """Delete all data for a specific group"""
        try:
            deletion_counts = {}
            
            # Delete from retriever (ChromaDB + Elasticsearch)
            retriever_counts = self.retriever.delete_by_group(group_id)
            deletion_counts.update(retriever_counts)
            
            # Delete from MongoDB
            # Delete chunks
            chunk_result = self.chunks_collection.delete_many({'group_id': group_id})
            deletion_counts['mongodb_chunks'] = chunk_result.deleted_count
            
            # Delete documents
            doc_result = self.documents_collection.delete_many({'group_id': group_id})
            deletion_counts['mongodb_documents'] = doc_result.deleted_count
            
            # Delete retrieval logs
            log_result = self.retrieval_logs_collection.delete_many({'group_id': group_id})
            deletion_counts['mongodb_logs'] = log_result.deleted_count
            
            logger.info(f"Deleted all data for group {group_id}: {deletion_counts}")
            return deletion_counts
            
        except Exception as e:
            logger.error(f"Failed to delete data for group {group_id}: {e}")
            return {}
    
    def get_group_statistics(self, group_id: str) -> Dict:
        """Get statistics for a specific group"""
        try:
            stats = {}
            
            # Document count
            stats['document_count'] = self.documents_collection.count_documents({'group_id': group_id})
            
            # Chunk count
            stats['chunk_count'] = self.chunks_collection.count_documents({'group_id': group_id})
            
            # Recent query count (last 24 hours)
            from_date = datetime.utcnow() - timedelta(hours=24)
            stats['recent_queries'] = self.retrieval_logs_collection.count_documents({
                'group_id': group_id,
                'created_at': {'$gte': from_date}
            })
            
            # Chunk type distribution
            pipeline = [
                {'$match': {'group_id': group_id}},
                {'$group': {'_id': '$chunk_type', 'count': {'$sum': 1}}}
            ]
            chunk_types = list(self.chunks_collection.aggregate(pipeline))
            stats['chunk_types'] = {item['_id']: item['count'] for item in chunk_types}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics for group {group_id}: {e}")
            return {}
    
    def get_performance_metrics(self, hours: int = 24, group_id: Optional[str] = None) -> Dict:
        """Get performance metrics for the last N hours with optional group filtering"""
        try:
            from_date = datetime.utcnow() - timedelta(hours=hours)
            
            match_filter = {'created_at': {'$gte': from_date}}
            if group_id and group_id != "general":
                match_filter['group_id'] = group_id
            
            pipeline = [
                {'$match': match_filter},
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


# Elasticsearch connection helper
def get_elasticsearch_connection(host: str, port: int, username: str = None, password: str = None) -> Elasticsearch:
    """Get Elasticsearch connection"""
    try:
        # Build connection configuration
        es_config = {
            'hosts': [{'host': host, 'port': port, 'scheme': 'http'}],
            'timeout': 30,
            'max_retries': 3,
            'retry_on_timeout': True
        }
        
        # Add authentication if provided
        if username and password:
            es_config['http_auth'] = (username, password)
        
        # Create client
        es_client = Elasticsearch(**es_config)
        
        # Test connection
        if es_client.ping():
            logger.info(f"Elasticsearch connection established at {host}:{port}")
            return es_client
        else:
            raise ConnectionError("Elasticsearch ping failed")
            
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        raise


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
                path="/app_data/chroma_db",
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