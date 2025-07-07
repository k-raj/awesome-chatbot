FROM python:3.10-slim

WORKDIR /app

# Build arguments
ARG MODEL_TYPE=ollama
ARG MODEL_NAME=llama3.2:3b

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh;

# Copy application code
COPY server.py .

# Set environment variables
ENV MODEL_TYPE=$MODEL_TYPE
ENV MODEL_NAME=$MODEL_NAME


# Run the application
CMD ollama serve & sleep 2 && ollama pull "$MODEL_NAME"