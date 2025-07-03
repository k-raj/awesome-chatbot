FROM python:3.10-slim

WORKDIR /app

# Build arguments
ARG MODEL_TYPE=ollama
ARG MODEL_NAME=llama2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install model-specific dependencies
RUN if [ "$MODEL_TYPE" = "ollama" ]; then \
        curl -fsSL https://ollama.ai/install.sh | sh; \
    fi

# Copy application code
COPY server.py .

# Set environment variables
ENV MODEL_TYPE=$MODEL_TYPE
ENV MODEL_NAME=$MODEL_NAME

# Download model at build time (for Ollama)
RUN if [ "$MODEL_TYPE" = "ollama" ]; then \
        ollama serve & \
        sleep 10 && \
        ollama pull $MODEL_NAME && \
        pkill ollama; \
    fi

# Run the application
CMD if [ "$MODEL_TYPE" = "ollama" ]; then \
        ollama serve & \
        sleep 10 && \
        python server.py; \
    else \
        python server.py; \
    fi
