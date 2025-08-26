#!/usr/bin/env python3
"""
Cron job script for daily document ingestion
Add to crontab: 0 3 * * * /path/to/python /path/to/scripts/cron_ingest.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import KnowledgeAssistant

def setup_logging():
    """Setup logging for cron job"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"cron_ingest_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def send_notification(message: str, is_error: bool = False):
    """Send notification about ingestion status"""
    # You can customize this to send emails, Slack messages, etc.
    logger = logging.getLogger(__name__)
    
    if is_error:
        logger.error(f"CRON INGESTION ERROR: {message}")
    else:
        logger.info(f"CRON INGESTION: {message}")
    
    # Optional: Send email notification
    # Uncomment and configure if you want email notifications
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        
        # Configure your email settings
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        email_user = os.getenv("EMAIL_USER")
        email_password = os.getenv("EMAIL_PASSWORD")
        notification_email = os.getenv("NOTIFICATION_EMAIL")
        
        if all([smtp_server, email_user, email_password, notification_email]):
            msg = MIMEText(message)
            msg['Subject'] = f"AI Assistant Ingestion {'ERROR' if is_error else 'Report'}"
            msg['From'] = email_user
            msg['To'] = notification_email
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(email_user, email_password)
                server.send_message(msg)
            
            logger.info("Email notification sent")
    except Exception as e:
        logger.warning(f"Failed to send email notification: {e}")
    """

def main():
    """Main cron job function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting scheduled document ingestion")
        start_time = datetime.now()
        
        # Change to script directory
        script_dir = Path(__file__).parent.parent
        os.chdir(script_dir)
        
        # Initialize assistant
        assistant = KnowledgeAssistant()
        
        # Get current state
        initial_chunks = len(assistant.text_chunks)
        initial_images = len(assistant.image_paths)
        
        # Check for PDF files
        pdf_files = list(Path("data").glob("*.pdf"))
        
        if not pdf_files:
            logger.info("No PDF files found in data directory")
            return
        
        # Track processed files to identify new ones
        processed_files = set()
        for metadata in assistant.chunk_metadata:
            processed_files.add(metadata.get('source_file', ''))
        
        new_files = [f for f in pdf_files if f.name not in processed_files]
        
        if not new_files:
            logger.info("No new files to process")
            return
        
        logger.info(f"Processing {len(new_files)} new files: {[f.name for f in new_files]}")
        
        # Process new files
        for pdf_file in new_files:
            try:
                assistant.ingest_document(str(pdf_file))
                logger.info(f"Successfully processed: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        # Rebuild indices
        assistant.build_indices()
        assistant.save_indices()
        
        # Calculate results
        final_chunks = len(assistant.text_chunks)
        final_images = len(assistant.image_paths)
        new_chunks = final_chunks - initial_chunks
        new_images = final_images - initial_images
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Prepare summary
        summary = f"""
Daily Ingestion Report - {start_time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}
Files processed: {len(new_files)}
New text chunks: {new_chunks}
New images: {new_images}
Total chunks: {final_chunks}
Total images: {final_images}
Processing time: {duration.total_seconds():.2f} seconds

Files processed:
{chr(10).join(f'- {f.name}' for f in new_files)}
"""
        
        logger.info(summary)
        send_notification(summary)
        
        logger.info("Scheduled ingestion completed successfully")
        
    except Exception as e:
        error_msg = f"Scheduled ingestion failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        send_notification(error_msg, is_error=True)
        sys.exit(1)

if __name__ == "__main__":
    main()