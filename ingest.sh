#!/bin/bash
# Manually trigger document ingestion

echo "Starting document ingestion..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python ingest.py "$@"
