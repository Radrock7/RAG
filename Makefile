# AI Knowledge Assistant Makefile
# Provides convenient commands for development and deployment

.PHONY: help setup install dev run ingest clean test docker build deploy

# Default target
help:
	@echo "AI Knowledge Assistant - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup     - Initial setup and dependency installation"
	@echo "  make install   - Install Python dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  make dev       - Start development server with auto-reload"
	@echo "  make run       - Start production server"
	@echo "  make ingest    - Manually trigger document ingestion"
	@echo "  make test      - Run tests"
	@echo ""
	@echo "Docker:"
	@echo "  make docker    - Build and run with Docker"
	@echo "  make build     - Build Docker image"
	@echo "  make deploy    - Deploy with Docker Compose"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean     - Clean cache and temporary files"
	@echo "  make logs      - View recent logs"
	@echo "  make status    - Show system status"

# Setup and installation
setup:
	@echo "🚀 Setting up AI Knowledge Assistant..."
	python3 setup.py

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Development commands
dev:
	@echo "🔧 Starting development server..."
	python app.py --reload

run:
	@echo "🚀 Starting AI Knowledge Assistant..."
	python app.py

ingest:
	@echo "📚 Starting document ingestion..."
	python ingest.py

ingest-force:
	@echo "🔄 Force rebuilding all indices..."
	python ingest.py --force-rebuild

# Testing
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v || echo "No tests found - create tests/ directory"

# Docker commands
docker:
	@echo "🐳 Building and running with Docker..."
	docker-compose up --build

build:
	@echo "🏗️  Building Docker image..."
	docker build -t ai-knowledge-assistant .

deploy:
	@echo "🚢 Deploying with Docker Compose..."
	docker-compose up -d

deploy-with-watcher:
	@echo "🚢 Deploying with file watcher..."
	docker-compose --profile auto-ingest up -d

# Maintenance commands
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	@echo "✅ Cleanup complete"

logs:
	@echo "📋 Recent application logs:"
	@if [ -d "logs" ]; then \
		tail -n 50 logs/*.log 2>/dev/null || echo "No logs found"; \
	else \
		echo "Logs directory not found"; \
	fi

docker-logs:
	@echo "📋 Docker container logs:"
	docker-compose logs --tail=50

status:
	@echo "📊 System Status:"
	@echo "Data directory: $(shell ls -la data/ 2>/dev/null | wc -l) files"
	@echo "Indices directory: $(shell ls -la indices/ 2>/dev/null | wc -l) files"
	@if [ -f ".env" ]; then echo "✅ Environment configured"; else echo "❌ No .env file found"; fi
	@python -c "from pathlib import Path; import pickle; \
	try: \
		with open('indices/metadata.pkl', 'rb') as f: data = pickle.load(f); \
		print(f'Text chunks: {len(data.get(\"text_chunks\", []))}'); \
		print(f'Images: {len(data.get(\"image_paths\", []))}'); \
	except: print('No metadata found - run ingestion first')"

# Cron job setup
setup-cron:
	@echo "⏰ Setting up daily ingestion cron job..."
	@echo "Current directory: $(PWD)"
	@echo "Add this line to your crontab (crontab -e):"
	@echo "0 3 * * * cd $(PWD) && python scripts/cron_ingest.py"

# Virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv venv
	@echo "Activate with: source venv/bin/activate"

install-dev: venv
	@echo "📦 Installing development dependencies..."
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install pytest black flake8 mypy

# Code quality
format:
	@echo "🎨 Formatting code..."
	black *.py scripts/*.py

lint:
	@echo "🔍 Linting code..."
	flake8 *.py scripts/*.py --max-line-length=100 --ignore=E203,W503

type-check:
	@echo "🏷️  Type checking..."
	mypy *.py --ignore-missing-imports

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ indices/ .env
	@echo "✅ Backup created"

# Update models
update-models:
	@echo "🔄 Updating ML models..."
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
	python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
	@echo "✅ Models updated"