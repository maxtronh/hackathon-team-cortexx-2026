"""
Configuration module for the RAG pipeline.
Centralizes chunking, embedding, indexing, and retrieval settings.
All values can be overridden via environment variables.
"""
import os
from pathlib import Path

# Base storage directory for FAISS index and chunk metadata
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", BASE_DIR / "data"))

# Chunking Configuration
DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 150))

# Embedding Configuration
EMBEDDING_PROVIDER = os.environ.get("RAG_EMBEDDING_PROVIDER", "gemini").lower()
GEMINI_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "models/text-embedding-004")
EMBEDDING_DIMENSION = 768  # text-embedding-004 native vector dimension

# Retrieval Configuration
DEFAULT_TOP_K = int(os.environ.get("RAG_DEFAULT_TOP_K", 5))
MAX_TOP_K = int(os.environ.get("RAG_MAX_TOP_K", 20))
DEFAULT_RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", 0.40))

# Upload Constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt'}
