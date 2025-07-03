# RAG Chatbot System

A comprehensive Retrieval-Augmented Generation (RAG) chatbot system with content groups, scalable architecture, and production-ready deployment.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web Client    │────▶│   API Server    │────▶│     Redis       │
│    (Flask)      │     │    (Flask)      │     │  Message Queue  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                              ┌───────────────────────────┴───────────────┐
                              │                                           │
                    ┌─────────▼─────────┐                    ┌───────────▼──────────┐
                    │ Inference Service │                    │  Embedding Service   │
                    │   (vLLM/Ollama)   │                    │  (Sentence-BERT)     │
                    └───────────────────┘                    └──────────┬───────────┘
                                                                        │
                                                            ┌───────────▼──────────┐
                                                            │    ChromaDB          │
                                                            │  (Vector Database)   │
                                                            └──────────────────────┘
```

## Features

- **Advanced RAG Pipeline**:
  - Semantic chunking preserving document structure
  - Hybrid retrieval combining dense vectors and BM25
  - Cross-encoder re-ranking for improved relevance
  - Automatic performance monitoring and optimization
- **Content Groups**: Organize chats by topics/projects
- **Multi-turn Conversations**: Maintains context across chat sessions
- **File Upload Support**: Process PDF, Word documents, and text files
- **User Feedback**: Thumbs up/down feedback with detailed comments
- **MySQL Storage**: Persistent storage of chats, feedback, and embeddings
- **Scalable Architecture**: Kubernetes-ready with auto-scaling
- **Async Processing**: Redis-based message queue for non-blocking operations
- **Model Flexibility**: Support for multiple LLM backends (vLLM, Ollama, OpenAI)

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Kubernetes cluster (optional, for production deployment)
- At least 16GB RAM for running local models

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-chatbot
cd rag-chatbot
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

4. Access the application at `http://localhost:5000`

### First-time Setup

On first run, the system will:
1. Download the Wikipedia dataset from HuggingFace
2. Create embeddings for all documents (this may take 30-60 minutes)
3. Store embeddings in ChromaDB

## Configuration

### Model Selection

During Docker build, you'll be prompted to select a model. Supported options:

- **Ollama Models**: llama2, mistral, mixtral
- **vLLM Models**: Any HuggingFace model ID
- **OpenAI**: gpt-3.5-turbo, gpt-4 (requires API key)

### Environment Variables

```env
# Model Configuration
MODEL_TYPE=ollama              # ollama, vllm, or openai
MODEL_NAME=llama2              # Model identifier
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Service URLs
REDIS_URL=redis://redis:6379
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# MySQL Configuration
MYSQL_HOST=mysql
MYSQL_USER=raguser
MYSQL_PASSWORD=ragpassword
MYSQL_DATABASE=rag_chatbot

# API Keys (if using external services)
OPENAI_API_KEY=your-key-here

# Performance Settings
MAX_WORKERS=4
BATCH_SIZE=32
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# File Upload Settings
UPLOAD_FOLDER=/app/uploads
MAX_FILE_SIZE=52428800  # 50MB
```

## Development

### Project Structure

```
rag-chatbot/
├── docker-compose.yml
├── kubernetes/
│   ├── api-deployment.yaml
│   ├── inference-deployment.yaml
│   ├── embedding-deployment.yaml
│   ├── redis-deployment.yaml
│   ├── chromadb-deployment.yaml
│   └── ingress.yaml
├── services/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── web/
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   ├── templates/
│   │   └── static/
│   ├── inference/
│   │   ├── Dockerfile
│   │   ├── server.py
│   │   └── requirements.txt
│   └── embedding/
│       ├── Dockerfile
│       ├── server.py
│       └── requirements.txt
├── scripts/
│   ├── download_dataset.py
│   └── create_embeddings.py
└── README.md
```

### Running Locally

For development without Docker:

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Start Redis
redis-server

# Start ChromaDB
chroma run

# Start services (in separate terminals)
python services/embedding/server.py
python services/inference/server.py
python services/api/app.py
python services/web/app.py
```

## Kubernetes Deployment

Deploy to Kubernetes cluster:

```bash
# Create namespace
kubectl create namespace rag-chatbot

# Apply configurations
kubectl apply -f kubernetes/ -n rag-chatbot

# Scale inference service
kubectl scale deployment inference-service --replicas=3 -n rag-chatbot
```

### Auto-scaling Configuration

The Kubernetes deployment includes HPA (Horizontal Pod Autoscaler) for:
- API Server: Scales based on CPU usage (target 70%)
- Inference Service: Scales based on CPU and memory (target 80%)
- Embedding Service: Scales based on request rate

## API Documentation

### Chat Endpoints

- `POST /api/chat`: Send a message
  ```json
  {
    "message": "What is machine learning?",
    "group_id": "tech-topics",
    "session_id": "unique-session-id"
  }
  ```

- `GET /api/groups`: List content groups
- `POST /api/groups`: Create new content group
- `GET /api/history/{group_id}`: Get chat history

### Health Checks

- `GET /health`: Service health status
- `GET /metrics`: Prometheus metrics

## Architecture Decisions

### Why This Stack?

1. **Flask**: Lightweight, easy to extend, perfect for microservices
2. **Redis**: Fast message passing, supports pub/sub for real-time features
3. **ChromaDB**: Purpose-built for embeddings, great performance
4. **Docker/K8s**: Industry standard for scalable deployments

### RAG Implementation

- **Chunking Strategy**: 512 tokens with 50 token overlap for context preservation
- **Embedding Model**: all-MiniLM-L6-v2 for balance of speed and quality
- **Retrieval**: Top-5 most similar chunks, re-ranked by relevance
- **Context Window**: 2048 tokens max context per query

### Advanced RAG Implementation

The system implements state-of-the-art retrieval techniques:

1. **Semantic Chunking**: Documents are intelligently chunked based on:
   - Document structure (headings, paragraphs, lists)
   - Semantic boundaries using spaCy
   - Configurable chunk sizes with overlap

2. **Hybrid Retrieval**: Combines multiple retrieval methods:
   - Dense vector search using sentence transformers
   - BM25 sparse retrieval for keyword matching
   - Weighted combination of both approaches

3. **Cross-Encoder Reranking**: 
   - Initial retrieval gets top-20 candidates
   - Cross-encoder model reranks for final top-5
   - Significantly improves relevance

4. **Performance Monitoring**:
   - Real-time tracking of retrieval latency
   - Correlation analysis between scores and feedback
   - Automatic performance reports

### RAG Performance Metrics

Monitor your RAG system performance:

```bash
# Generate performance report
docker-compose exec api-server python /app/monitor_rag_performance.py

# View real-time metrics
curl http://localhost:5001/api/analytics/retrieval
```

### Advanced Configuration

```env
# Retrieval Settings
VECTOR_WEIGHT=0.7          # Weight for vector search (0-1)
BM25_WEIGHT=0.3           # Weight for BM25 search (0-1)
RERANK_TOP_K=5            # Final results after reranking
RETRIEVAL_TOP_K=20        # Initial candidates for reranking

# Chunking Settings
MIN_CHUNK_SIZE=100        # Minimum chunk size in tokens
MAX_CHUNK_SIZE=512        # Maximum chunk size in tokens
CHUNK_OVERLAP=50          # Overlap between chunks

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Monitoring & Debugging

### Logs

All services log to stdout/stderr. In Kubernetes:
```bash
kubectl logs -f deployment/api-server -n rag-chatbot
```

### Metrics

Prometheus metrics exposed on `/metrics`:
- Request latency
- Token usage
- Cache hit rates
- Queue depths

### Common Issues

1. **Slow first response**: Model loading on cold start
   - Solution: Keep inference service warm with health checks

2. **Out of memory**: Large models on limited hardware
   - Solution: Use quantized models or reduce batch size

3. **Embedding timeout**: Large dataset processing
   - Solution: Pre-compute embeddings, use persistent volume

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- HuggingFace for the Wikipedia dataset
- LangChain community for RAG patterns
- All contributors and testers

## Contact

For questions or support, please open an issue on GitHub.