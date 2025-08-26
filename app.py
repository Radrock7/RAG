import os
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
import gradio as gr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
import pickle
from dotenv import load_dotenv
from PIL import Image
import io

# ML/AI imports
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeAssistant:
    def __init__(self, 
                 data_dir: str = "data",
                 index_dir: str = "indices",
                 chunk_size: int = 500,
                 max_chunks_in_context: int = 5):
        """
        Initialize the AI Knowledge Assistant
        
        Args:
            data_dir: Directory containing PDF documents
            index_dir: Directory to store FAISS indices
            chunk_size: Size of text chunks for embedding
            max_chunks_in_context: Maximum chunks to include in LLM context
        """
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.max_chunks_in_context = max_chunks_in_context
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self._load_models()
        
        # Initialize storage
        self.text_chunks = []
        self.image_paths = []
        self.chunk_metadata = []  # Store source info for each chunk
        self.text_index = None
        self.image_index = None
        
        # Load existing indices if available
        self._load_indices()
    
    def _load_models(self):
        """Load embedding and vision models"""
        logger.info("Loading models...")
        
        # Text embedding model
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Vision model for image embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        logger.info("Models loaded successfully")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF using PyMuPDF"""
        try:
            pdf = fitz.open(pdf_path)
            all_text = ""
            
            for page_num, page in enumerate(pdf):
                page_text = page.get_text()
                all_text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            pdf.close()
            return all_text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """Extract images from PDF and save them"""
        try:
            pdf = fitz.open(pdf_path)
            image_paths = []
            
            Path(output_dir).mkdir(exist_ok=True)
            pdf_name = Path(pdf_path).stem
            
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                images = page.get_images(full=True)
                
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = pdf.extract_image(xref)
                        img_bytes = base_image["image"]
                        
                        img_filename = f"{pdf_name}_page{page_index+1}_img{img_index+1}.png"
                        img_path = Path(output_dir) / img_filename
                        
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        
                        image_paths.append(str(img_path))
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_index}: {e}")
            
            pdf.close()
            return image_paths
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            return []
    
    def chunk_text(self, text: str, source_info: Dict) -> List[Dict]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            source_info: Metadata about the source document
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i+self.chunk_size])
            
            chunk_info = {
                'text': chunk_text,
                'source_file': source_info.get('filename', ''),
                'chunk_index': len(chunks),
                'word_start': i,
                'word_end': min(i + self.chunk_size, len(words)),
                'timestamp': datetime.now().isoformat()
            }
            chunks.append(chunk_info)
        
        return chunks
    
    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        return self.text_model.encode(texts)
    
    def generate_image_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Generate embeddings for images using CLIP"""
        embeddings = []
        
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = self.clip_processor(images=image, return_tensors="pt")
                image_features = self.clip_model.get_image_features(**inputs)
                embeddings.append(image_features.detach().numpy().flatten())
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {e}")
                # Add zero embedding as placeholder
                embeddings.append(np.zeros(512))
        
        return np.array(embeddings) if embeddings else np.array([]).reshape(0, 512)
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search"""
        if len(embeddings) == 0:
            return None
            
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype("float32"))
        return index
    
    def ingest_document(self, pdf_path: str):
        """Process a single PDF document"""
        logger.info(f"Ingesting document: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return
        
        # Extract images
        img_dir = self.index_dir / "images"
        image_paths = self.extract_images_from_pdf(pdf_path, str(img_dir))
        
        # Prepare source info
        source_info = {
            'filename': Path(pdf_path).name,
            'full_path': str(pdf_path),
            'ingestion_date': datetime.now().isoformat()
        }
        
        # Chunk text
        text_chunks = self.chunk_text(text, source_info)
        
        # Store chunks and metadata
        start_idx = len(self.text_chunks)
        self.text_chunks.extend([chunk['text'] for chunk in text_chunks])
        self.chunk_metadata.extend(text_chunks)
        self.image_paths.extend(image_paths)
        
        logger.info(f"Added {len(text_chunks)} text chunks and {len(image_paths)} images")
    
    def build_indices(self):
        """Build FAISS indices for text and images"""
        logger.info("Building indices...")
        
        if self.text_chunks:
            # Generate text embeddings
            text_embeddings = self.generate_text_embeddings(self.text_chunks)
            self.text_index = self.build_faiss_index(text_embeddings)
            logger.info(f"Built text index with {len(self.text_chunks)} chunks")
        
        if self.image_paths:
            # Generate image embeddings
            image_embeddings = self.generate_image_embeddings(self.image_paths)
            if len(image_embeddings) > 0:
                self.image_index = self.build_faiss_index(image_embeddings)
                logger.info(f"Built image index with {len(self.image_paths)} images")
    
    def save_indices(self):
        """Save indices and metadata to disk"""
        logger.info("Saving indices...")
        
        # Save FAISS indices
        if self.text_index:
            faiss.write_index(self.text_index, str(self.index_dir / "text_index.faiss"))
        if self.image_index:
            faiss.write_index(self.image_index, str(self.index_dir / "image_index.faiss"))
        
        # Save metadata
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump({
                'text_chunks': self.text_chunks,
                'chunk_metadata': self.chunk_metadata,
                'image_paths': self.image_paths,
            }, f)
        
        logger.info("Indices saved successfully")
    
    def _load_indices(self):
        """Load existing indices from disk"""
        try:
            # Load FAISS indices
            text_index_path = self.index_dir / "text_index.faiss"
            if text_index_path.exists():
                self.text_index = faiss.read_index(str(text_index_path))
                logger.info("Loaded text index")
            
            image_index_path = self.index_dir / "image_index.faiss"
            if image_index_path.exists():
                self.image_index = faiss.read_index(str(image_index_path))
                logger.info("Loaded image index")
            
            # Load metadata
            metadata_path = self.index_dir / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.text_chunks = data.get('text_chunks', [])
                    self.chunk_metadata = data.get('chunk_metadata', [])
                    self.image_paths = data.get('image_paths', [])
                logger.info(f"Loaded {len(self.text_chunks)} text chunks and {len(self.image_paths)} images")
        
        except Exception as e:
            logger.warning(f"Could not load existing indices: {e}")
    
    def search_text(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant text chunks"""
        if not self.text_index or not self.text_chunks:
            return []
        
        # Generate query embedding
        query_embedding = self.text_model.encode([query])
        
        # Search
        distances, indices = self.text_index.search(
            query_embedding.astype("float32"), k
        )
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunk_metadata):
                result = self.chunk_metadata[idx].copy()
                result['similarity_score'] = float(dist)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def search_images(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant images using text query"""
        if not self.image_index or not self.image_paths:
            return []
        
        # Generate query embedding using CLIP text encoder
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**inputs)
        query_embedding = text_features.detach().numpy()
        
        # Search
        distances, indices = self.image_index.search(
            query_embedding.astype("float32"), k
        )
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.image_paths):
                results.append({
                    'image_path': self.image_paths[idx],
                    'similarity_score': float(dist),
                    'rank': i + 1
                })
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[Dict], 
                         image_results: List[Dict] = None) -> str:
        """Generate response using OpenAI GPT"""
        # Prepare context
        context_texts = []
        sources = []
        
        for chunk in context_chunks[:self.max_chunks_in_context]:
            context_texts.append(chunk['text'])
            sources.append(f"Source: {chunk['source_file']} (Chunk {chunk['chunk_index']})")
        
        context = "\n\n".join([f"{text}\n[{source}]" 
                              for text, source in zip(context_texts, sources)])
        
        # Add image information if available
        image_context = ""
        if image_results:
            image_info = [f"Relevant image: {Path(img['image_path']).name}" 
                         for img in image_results[:3]]
            image_context = f"\n\nRelated images found: {', '.join(image_info)}"
        
        # Prepare prompt
        system_prompt = """You are a helpful AI assistant with access to a knowledge base. 
        Answer questions based on the provided context. Always cite your sources by mentioning 
        the source file and chunk number. If you cannot answer based on the context, say so clearly."""
        
        user_prompt = f"""Based on the following context, please answer this question:

Context:
{context}{image_context}

Question: {query}

Please provide a comprehensive answer and cite your sources."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def query(self, question: str) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Main query method that searches and generates response
        
        Returns:
            Tuple of (response_text, text_results, image_results)
        """
        # Search for relevant content
        text_results = self.search_text(question, k=10)
        image_results = self.search_images(question, k=5)
        
        # Generate response
        response = self.generate_response(question, text_results, image_results)
        
        return response, text_results, image_results
    
    def ingest_all_documents(self):
        """Ingest all PDF documents in the data directory"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to ingest")
        
        for pdf_path in pdf_files:
            self.ingest_document(str(pdf_path))
        
        if pdf_files:
            self.build_indices()
            self.save_indices()
        else:
            logger.warning("No PDF files found in data directory")

def create_gradio_interface(assistant: KnowledgeAssistant):
    """Create Gradio chat interface"""
    
    def chat_fn(message, history):
        """Chat function for Gradio interface"""
        try:
            response, text_results, image_results = assistant.query(message)
            
            # Format response with sources
            if text_results:
                sources_info = "\n\n**Sources:**\n"
                for result in text_results[:3]:  # Show top 3 sources
                    sources_info += f"- {result['source_file']} (Chunk {result['chunk_index']}, Score: {result['similarity_score']:.3f})\n"
                response += sources_info
            
            # Add image information
            if image_results:
                images_info = "\n\n**Related Images Found:**\n"
                for img in image_results[:3]:
                    images_info += f"- {Path(img['image_path']).name} (Score: {img['similarity_score']:.3f})\n"
                response += images_info
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat function: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    # Management functions
    def ingest_documents():
        """Ingest all documents in data directory"""
        try:
            assistant.ingest_all_documents()
            return "✅ Documents ingested successfully!"
        except Exception as e:
            return f"❌ Error ingesting documents: {str(e)}"
    
    def get_status():
        """Get system status"""
        return f"""**System Status:**
- Text chunks: {len(assistant.text_chunks)}
- Images: {len(assistant.image_paths)}
- Text index: {'✅ Ready' if assistant.text_index else '❌ Not built'}
- Image index: {'✅ Ready' if assistant.image_index else '❌ Not built'}
- Data directory: {assistant.data_dir}
"""
    
    # Create interface with multiple tabs
    with gr.Blocks(title="AI Knowledge Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧠 AI Knowledge Assistant")
        gr.Markdown("Upload PDFs to the `data/` folder and ask questions about their content!")
        
        with gr.Tab("Chat"):
            chat_interface = gr.ChatInterface(
                fn=chat_fn,
                title="Ask Questions",
                description="Ask me anything about your documents!"
            )
        
        with gr.Tab("Management"):
            gr.Markdown("## Document Management")
            
            with gr.Row():
                ingest_btn = gr.Button("🔄 Ingest All Documents", variant="primary")
                status_btn = gr.Button("📊 Check Status")
            
            output_box = gr.Textbox(
                label="Output", 
                lines=10, 
                interactive=False
            )
            
            ingest_btn.click(ingest_documents, outputs=output_box)
            status_btn.click(get_status, outputs=output_box)
    
    return interface

def main():
    """Main application entry point"""
    # Initialize the assistant
    assistant = KnowledgeAssistant()
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(assistant)
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()