#!/bin/bash
# Start the AI Knowledge Assistant

echo "Starting AI Knowledge Assistant..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please run setup.py first."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Start the application
python app.py
