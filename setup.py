#!/usr/bin/env python3
"""
Setup script for AI Knowledge Assistant
Handles initial setup, dependency installation, and configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True):
    """Run shell command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "indices", 
        "indices/images",
        "logs",
        "scripts"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        print("Warning: requirements.txt not found")

def setup_environment():
    """Setup environment configuration"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your OpenAI API key")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        # Create basic .env file
        with open(env_file, "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("LOG_LEVEL=INFO\n")
        print("✅ Created basic .env file")
        print("⚠️  Please edit .env file and add your OpenAI API key")

def check_gpu_support():
    """Check if GPU/CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print("💡 Consider installing faiss-gpu for better performance:")
            print("   pip install faiss-gpu")
        else:
            print("ℹ️  CUDA not available, using CPU")
    except ImportError:
        print("ℹ️  PyTorch not yet installed")

def create_scripts():
    """Create helper scripts"""
    
    # Create start script
    start_script = """#!/bin/bash
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
"""
    
    with open("start.sh", "w") as f:
        f.write(start_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("start.sh", 0o755)
    
    print("✅ Created start.sh script")
    
    # Create ingest script
    ingest_script = """#!/bin/bash
# Manually trigger document ingestion

echo "Starting document ingestion..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python ingest.py "$@"
"""
    
    with open("ingest.sh", "w") as f:
        f.write(ingest_script)
    
    if os.name != 'nt':
        os.chmod("ingest.sh", 0o755)
    
    print("✅ Created ingest.sh script")

def test_installation():
    """Test if installation was successful"""
    print("\nTesting installation...")
    
    try:
        # Test imports
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import fitz
        import gradio
        
        print("✅ All core dependencies imported successfully")
        
        # Test model loading (quick test)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        test_embedding = model.encode(["test"])
        print(f"✅ Sentence transformer working (embedding shape: {test_embedding.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Knowledge Assistant Setup")
    print("=" * 40)
    
    # Check system requirements
    check_python_version()
    
    # Create directory structure
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Check GPU support
    check_gpu_support()
    
    # Create helper scripts
    create_scripts()
    
    # Test installation
    success = test_installation()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Add PDF files to the 'data/' directory")
        print("3. Run: python app.py (or ./start.sh)")
        print("4. Open http://localhost:7860 in your browser")
        print("\nFor automatic ingestion:")
        print("- Use Docker: docker-compose up")
        print("- Or run: python scripts/watch_files.py")
    else:
        print("❌ Setup completed with errors")
        print("Please check the error messages above and resolve them")
        sys.exit(1)

if __name__ == "__main__":
    main()