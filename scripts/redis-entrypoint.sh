#!/bin/sh
set -e

# Wait a moment for environment to be fully loaded
sleep 1

# Check if REDIS_PASSWORD is set
if [ -z "$REDIS_PASSWORD" ]; then
    echo "ERROR: REDIS_PASSWORD environment variable is not set"
    exit 1
fi

echo "Starting Redis with password authentication..."
exec redis-server --requirepass "$REDIS_PASSWORD"