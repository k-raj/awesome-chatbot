#!/bin/bash

# RAG Chatbot Setup Script
# This script sets up the complete RAG chatbot system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_info "✓ All prerequisites met"
}

# Create directory structure
create_directories() {
    print_info "Creating directory structure..."
    
    directories=(
        "services/api"
        "services/web/templates"
        "services/web/static"
        "services/embedding"
        "services/inference"
        "scripts"
        "data"
        "models"
        "kubernetes"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_info "✓ Directory structure created"
}

# Create environment file
create_env_file() {
    print_info "Creating environment configuration..."
    
    if [ -f .env ]; then
        print_warning ".env file already exists. Backing up to .env.backup"
        cp .env .env.backup
    fi
    
    cat > .env << 'EOF'
# Model Configuration
MODEL_TYPE=ollama
MODEL_NAME=llama2
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Service Configuration
REDIS_URL=redis://redis:6379
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# MySQL Configuration
MYSQL_HOST=mysql
MYSQL_USER=raguser
MYSQL_PASSWORD=ragpassword
MYSQL_DATABASE=rag_chatbot

# Optional: OpenAI Configuration
# OPENAI_API_KEY=your-key-here

# Performance Settings
MAX_WORKERS=4
BATCH_SIZE=32
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# File Upload Settings
UPLOAD_FOLDER=/app/uploads
MAX_FILE_SIZE=52428800

# Debug Mode
DEBUG=False
EOF
    
    print_info "✓ Environment configuration created"
}

# Prompt for model selection
select_model() {
    print_info "Select the LLM backend you want to use:"
    echo "1) Ollama (Local, recommended for testing)"
    echo "2) vLLM (Local, high performance)"
    echo "3) OpenAI (Cloud, requires API key)"
    
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            MODEL_TYPE="ollama"
            print_info "Selected Ollama. Available models:"
            echo "- llama2 (7B, recommended)"
            echo "- mistral (7B)"
            echo "- mixtral (8x7B, requires more RAM)"
            read -p "Enter model name (default: llama2): " model_name
            MODEL_NAME=${model_name:-llama2}
            ;;
        2)
            MODEL_TYPE="vllm"
            print_info "Selected vLLM."
            read -p "Enter HuggingFace model ID: " MODEL_NAME
            ;;
        3)
            MODEL_TYPE="openai"
            print_info "Selected OpenAI."
            read -p "Enter your OpenAI API key: " OPENAI_KEY
            echo "Available models: gpt-3.5-turbo, gpt-4"
            read -p "Enter model name (default: gpt-3.5-turbo): " model_name
            MODEL_NAME=${model_name:-gpt-3.5-turbo}
            ;;
        *)
            print_error "Invalid choice. Using default (Ollama with llama2)"
            MODEL_TYPE="ollama"
            MODEL_NAME="llama2"
            ;;
    esac
    
    # Update .env file
    sed -i "s/MODEL_TYPE=.*/MODEL_TYPE=$MODEL_TYPE/" .env
    sed -i "s/MODEL_NAME=.*/MODEL_NAME=$MODEL_NAME/" .env
    
    if [ "$MODEL_TYPE" = "openai" ] && [ ! -z "$OPENAI_KEY" ]; then
        sed -i "s/# OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" .env
    fi
    
    print_info "✓ Model configuration updated"
}

# Build Docker images
build_images() {
    print_info "Building Docker images..."
    
    # Build all services
    docker-compose build --parallel
    
    print_info "✓ Docker images built successfully"
}

# Start services
start_services() {
    print_info "Starting services..."
    
    # Start core services first
    docker-compose up -d redis chromadb mysql
    
    print_info "Waiting for core services to be ready..."
    
    # Wait for MySQL to be ready
    print_info "Waiting for MySQL to initialize..."
    until docker-compose exec -T mysql mysql -uraguser -pragpassword -e "SELECT 1" &> /dev/null; do
        sleep 2
    done
    print_info "✓ MySQL is ready"
    
    # Wait for other services
    sleep 10
    
    # Start remaining services
    docker-compose up -d
    
    print_info "✓ All services started"
}

# Load dataset
load_dataset() {
    print_info "Loading Wikipedia dataset..."
    print_warning "This may take 30-60 minutes depending on your system"
    
    docker-compose run --rm dataset-loader
    
    print_info "✓ Dataset loaded and indexed"
}

# Check service health
check_health() {
    print_info "Checking service health..."
    
    services=("redis:6379" "chromadb:8000" "mysql:3306" "api-server:5001" "web-server:5000")
    
    for service in "${services[@]}"; do
        container_name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if docker-compose ps | grep -q "$container_name.*Up"; then
            print_info "✓ $container_name is running"
        else
            print_error "✗ $container_name is not running"
        fi
    done
}

# Generate Kubernetes manifests
generate_k8s_manifests() {
    print_info "Generating Kubernetes manifests..."
    
    # Create secret for OpenAI API key if provided
    if grep -q "OPENAI_API_KEY=" .env && ! grep -q "# OPENAI_API_KEY=" .env; then
        OPENAI_KEY=$(grep "OPENAI_API_KEY=" .env | cut -d= -f2)
        kubectl create secret generic openai-secret \
            --from-literal=api-key="$OPENAI_KEY" \
            --dry-run=client -o yaml > kubernetes/openai-secret.yaml
    fi
    
    print_info "✓ Kubernetes manifests ready in kubernetes/"
}

# Main setup flow
main() {
    echo "==================================="
    echo "RAG Chatbot Setup Script"
    echo "==================================="
    echo
    
    check_prerequisites
    create_directories
    create_env_file
    select_model
    
    print_info "Ready to build and deploy. This will:"
    echo "  1. Build all Docker images"
    echo "  2. Start all services"
    echo "  3. Load and index the Wikipedia dataset"
    echo
    read -p "Continue? (y/n): " confirm
    
    if [ "$confirm" != "y" ]; then
        print_info "Setup cancelled"
        exit 0
    fi
    
    build_images
    start_services
    
    print_info "Services are starting up..."
    sleep 20
    
    check_health
    
    read -p "Load Wikipedia dataset now? (y/n): " load_data
    if [ "$load_data" = "y" ]; then
        load_dataset
    else
        print_warning "You can load the dataset later with: docker-compose run dataset-loader"
    fi
    
    generate_k8s_manifests
    
    echo
    echo "==================================="
    echo "Setup Complete!"
    echo "==================================="
    echo
    print_info "Access the chatbot at: http://localhost:5000"
    print_info "API endpoint: http://localhost:5001"
    echo
    print_info "To stop services: docker-compose down"
    print_info "To view logs: docker-compose logs -f [service-name]"
    echo
    print_info "For Kubernetes deployment:"
    echo "  kubectl apply -f kubernetes/"
    echo
}

# Run main function
main