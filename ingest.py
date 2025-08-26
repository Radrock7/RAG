#!/usr/bin/env python3
"""
Standalone script for ingesting documents
Can be run via cron job for automated ingestion
Usage: python ingest.py [--data-dir DATA_DIR] [--force-rebuild]
"""

import argparse
import sys
import logging
from pathlib import Path

# Import our main application
from app import KnowledgeAssistant

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ingest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Ingest documents into AI Knowledge Assistant')
    parser.add_argument('--data-dir', default='data', 
                       help='Directory containing PDF documents')
    parser.add_argument('--index-dir', default='indices',
                       help='Directory to store indices')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of all indices')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Size of text chunks for embedding')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting document ingestion...")
        
        # Initialize assistant
        assistant = KnowledgeAssistant(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size
        )
        
        # If force rebuild, clear existing data
        if args.force_rebuild:
            logger.info("Force rebuild requested - clearing existing indices")
            assistant.text_chunks = []
            assistant.chunk_metadata = []
            assistant.image_paths = []
            assistant.text_index = None
            assistant.image_index = None
        
        # Check for new files
        data_path = Path(args.data_dir)
        pdf_files = list(data_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {args.data_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Track processed files to avoid duplicates
        processed_files = set()
        if not args.force_rebuild:
            for metadata in assistant.chunk_metadata:
                processed_files.add(metadata.get('source_file', ''))
        
        new_files = [f for f in pdf_files if f.name not in processed_files]
        
        if new_files:
            logger.info(f"Processing {len(new_files)} new files...")
            for pdf_file in new_files:
                assistant.ingest_document(str(pdf_file))
            
            # Rebuild indices with new data
            assistant.build_indices()
            assistant.save_indices()
            
            logger.info(f"Successfully ingested {len(new_files)} new documents")
        else:
            logger.info("No new files to process")
        
        # Print summary
        logger.info(f"""
Ingestion Summary:
- Total text chunks: {len(assistant.text_chunks)}
- Total images: {len(assistant.image_paths)}
- New files processed: {len(new_files)}
- Data directory: {args.data_dir}
- Index directory: {args.index_dir}
        """)
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()