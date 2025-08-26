#!/usr/bin/env python3
"""
File watcher service for automatic document ingestion
Monitors the data directory for new PDF files and automatically ingests them
"""

import time
import logging
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFHandler(FileSystemEventHandler):
    """Handle PDF file events"""
    
    def __init__(self, data_dir: str = "data", cooldown: int = 30):
        self.data_dir = data_dir
        self.cooldown = cooldown  # Wait time before processing
        self.pending_files = set()
        self.last_process_time = 0
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Only process PDF files
        if file_path.suffix.lower() == '.pdf':
            logger.info(f"New PDF detected: {file_path.name}")
            self.pending_files.add(file_path)
            self._schedule_processing()
    
    def on_moved(self, event):
        """Handle file move events (e.g., downloads completing)"""
        if event.is_directory:
            return
            
        dest_path = Path(event.dest_path)
        
        if dest_path.suffix.lower() == '.pdf':
            logger.info(f"PDF moved to data directory: {dest_path.name}")
            self.pending_files.add(dest_path)
            self._schedule_processing()
    
    def _schedule_processing(self):
        """Schedule processing after cooldown period"""
        current_time = time.time()
        self.last_process_time = current_time
        
        # Start a background timer to process files
        def delayed_process():
            time.sleep(self.cooldown)
            # Only process if no new files were added during cooldown
            if time.time() - self.last_process_time >= self.cooldown - 1:
                self._process_pending_files()
        
        import threading
        threading.Thread(target=delayed_process, daemon=True).start()
    
    def _process_pending_files(self):
        """Process all pending files"""
        if not self.pending_files:
            return
        
        logger.info(f"Processing {len(self.pending_files)} pending files...")
        
        try:
            # Run the ingestion script
            result = subprocess.run([
                sys.executable, "ingest.py", 
                "--data-dir", self.data_dir
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Ingestion completed successfully")
                logger.info(result.stdout)
                self.pending_files.clear()
            else:
                logger.error(f"Ingestion failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            logger.error("Ingestion timed out")
        except Exception as e:
            logger.error(f"Error running ingestion: {e}")

def main():
    """Main file watcher loop"""
    data_dir = "data"
    
    # Ensure data directory exists
    Path(data_dir).mkdir(exist_ok=True)
    
    logger.info(f"Starting file watcher for directory: {data_dir}")
    
    # Create event handler and observer
    event_handler = PDFHandler(data_dir)
    observer = Observer()
    observer.schedule(event_handler, data_dir, recursive=False)
    
    # Start watching
    observer.start()
    logger.info("File watcher started. Monitoring for new PDF files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()
    
    observer.join()
    logger.info("File watcher stopped")

if __name__ == "__main__":
    main()