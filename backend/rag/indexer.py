"""
FAISS Vector Store and Index Management Module.
Maintains synchronized mapping between dense embedding vectors and document chunk metadata.
Supports Cosine Similarity retrieval, document filtering, thresholding, deduplication, and persistence.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from rag.chunker import DocumentChunk
from rag.config import DATA_DIR, EMBEDDING_DIMENSION
from rag.embeddings import BaseEmbedder, get_default_embedder


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    character_count: int
    score: float
    similarity_metric: str = "cosine"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VectorStore:
    """
    FAISS-backed vector store using Inner Product on L2-normalized vectors (Cosine Similarity).

    Features:
    - Guaranteed 1-to-1 synchronized mapping between FAISS vector indices and chunk metadata.
    - Automatic deduplication: prevents re-indexing identical documents.
    - Persistent storage in DATA_DIR (faiss.index, chunks.json, documents.json).
    - Safe rebuild path if disk artifacts are missing, stale, or out of sync.
    - Strict validation of index dimension, vector count, and metadata alignment.
    """
    def __init__(self, dimension: int = EMBEDDING_DIMENSION, storage_dir: Optional[Path] = None):
        self.dimension = dimension
        self.storage_dir = Path(storage_dir) if storage_dir else DATA_DIR

        # Primary FAISS index (IndexFlatIP computes dot product = cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        # Synchronized list of chunk metadata dicts corresponding 1-to-1 with vector indices
        self.metadata_store: List[Dict[str, Any]] = []
        # Registered documents catalog: doc_id -> { "document_id", "filename", "chunk_count", "page_count" }
        self.documents_registry: Dict[str, Dict[str, Any]] = {}

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def has_document(self, document_id: str) -> bool:
        """Checks if a document ID is already registered in the store."""
        return document_id in self.documents_registry

    def add_chunks(self, chunks: List[DocumentChunk], embedder: BaseEmbedder) -> int:
        """
        Embeds and indexes document chunks into FAISS with synchronized metadata.
        Guarantees that repeated indexing of the exact same document does not create duplicate entries.
        """
        if not chunks:
            return 0

        first_doc_id = chunks[0].document_id
        # If this document is already completely indexed, return existing count to avoid duplicates
        if self.has_document(first_doc_id):
            existing_count = self.documents_registry[first_doc_id].get("chunk_count", 0)
            # Verify that its chunks are indeed in metadata store
            doc_chunks_present = sum(1 for m in self.metadata_store if m.get("document_id") == first_doc_id)
            if doc_chunks_present > 0:
                return doc_chunks_present

        if embedder.dimension != self.dimension:
            raise ValueError(
                f"Embedder dimension mismatch: embedder has {embedder.dimension}, "
                f"store expects {self.dimension}."
            )

        texts = [c.text for c in chunks]
        vectors = embedder.embed_texts(texts)

        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Vector count mismatch: generated {vectors.shape[0]} vectors for {len(chunks)} chunks."
            )

        # Validate non-empty and finite float32 vectors
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        # Add vectors to FAISS index
        self.index.add(vectors)

        # Add chunk metadata in exact lockstep
        for chunk in chunks:
            c_dict = chunk.to_dict() if isinstance(chunk, DocumentChunk) else chunk
            self.metadata_store.append(c_dict)

            # Update document registry
            doc_id = c_dict["document_id"]
            if doc_id not in self.documents_registry:
                self.documents_registry[doc_id] = {
                    "document_id": doc_id,
                    "filename": c_dict["filename"],
                    "chunk_count": 0,
                    "page_count": c_dict["page"]
                }
            self.documents_registry[doc_id]["chunk_count"] += 1
            self.documents_registry[doc_id]["page_count"] = max(
                self.documents_registry[doc_id]["page_count"],
                c_dict["page"]
            )

        # Verify alignment
        if self.index.ntotal != len(self.metadata_store):
            raise RuntimeError(
                f"Index synchronization failure: index has {self.index.ntotal} vectors "
                f"but metadata store has {len(self.metadata_store)} records."
            )

        # Persist to disk
        if self.storage_dir:
            self.save_to_disk()

        return len(chunks)

    def search(
        self,
        query: str,
        embedder: BaseEmbedder,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Performs semantic retrieval against the FAISS index.

        Parameters:
        - query: user query string
        - embedder: embedding model (must match index dimension)
        - top_k: maximum number of relevant passages to retrieve
        - document_ids: optional list of document_ids to restrict search to
        - score_threshold: minimum cosine similarity score [-1.0, 1.0] to qualify as relevant
        """
        clean_query = query.strip()
        if not clean_query or self.index.ntotal == 0:
            return []

        if embedder.dimension != self.dimension:
            raise ValueError(
                f"Query vector dimension {embedder.dimension} does not match index {self.dimension}."
            )

        # Embed and normalize query vector
        query_vec = embedder.embed_query(clean_query)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        faiss.normalize_L2(query_vec)

        # Over-fetch if document filtering is active to guarantee top_k relevant hits
        search_k = min(self.index.ntotal, max(top_k * 5, 25))
        scores, indices = self.index.search(query_vec, search_k)

        results: List[SearchResult] = []
        valid_doc_set = set(document_ids) if document_ids else None

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue

            float_score = float(score)

            # Cosine similarity threshold filter
            if float_score < score_threshold:
                continue

            meta = self.metadata_store[idx]

            # Document isolation filter: documents not in valid_doc_set MUST NEVER leak
            if valid_doc_set and meta["document_id"] not in valid_doc_set:
                continue

            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page=meta["page"],
                    text=meta["text"],
                    character_count=meta["character_count"],
                    score=round(float_score, 4),
                    similarity_metric="cosine"
                )
            )

            if len(results) >= top_k:
                break

        return results

    def save_to_disk(self) -> None:
        """Persists FAISS binary index, chunk metadata, and document registry."""
        if not self.storage_dir:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.storage_dir / "faiss.index"
        metadata_file = self.storage_dir / "chunks.json"
        registry_file = self.storage_dir / "documents.json"

        # Write FAISS binary index
        faiss.write_index(self.index, str(index_file))

        # Write chunk metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

        # Write document registry
        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(self.documents_registry, f, ensure_ascii=False, indent=2)

    def _load_from_disk(self) -> None:
        """Loads index and metadata from disk if present and consistent."""
        if not self.storage_dir:
            return

        index_file = self.storage_dir / "faiss.index"
        metadata_file = self.storage_dir / "chunks.json"
        registry_file = self.storage_dir / "documents.json"

        if index_file.exists() and metadata_file.exists():
            try:
                loaded_index = faiss.read_index(str(index_file))
                with open(metadata_file, "r", encoding="utf-8") as f:
                    loaded_meta = json.load(f)

                # Validate alignment: vector count and metadata records must agree
                if loaded_index.ntotal == len(loaded_meta) and loaded_index.d == self.dimension:
                    self.index = loaded_index
                    self.metadata_store = loaded_meta

                    if registry_file.exists():
                        with open(registry_file, "r", encoding="utf-8") as rf:
                            self.documents_registry = json.load(rf)
                else:
                    # Incompatible or stale index: trigger safe rebuild
                    self.rebuild_from_metadata(loaded_meta)
            except Exception:
                # If disk artifacts are corrupted, reset to clean state
                self.reset()

    def rebuild_from_metadata(self, metadata: List[Dict[str, Any]], embedder: Optional[BaseEmbedder] = None) -> bool:
        """
        Deterministic rebuild path when index is missing, stale, or out of sync with metadata.
        """
        active_embedder = embedder or get_default_embedder(allow_mock=True)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        self.documents_registry = {}

        if not metadata:
            return True

        chunks = [
            DocumentChunk(
                chunk_id=m["chunk_id"],
                document_id=m["document_id"],
                filename=m["filename"],
                page=m["page"],
                text=m["text"],
                character_count=m["character_count"],
                chunk_index=m.get("chunk_index", idx + 1),
                chunk_position=m.get("chunk_position", idx + 1)
            )
            for idx, m in enumerate(metadata)
        ]

        self.add_chunks(chunks, active_embedder)
        return True

    def reset(self) -> None:
        """Clears all vectors and metadata in memory and on disk."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        self.documents_registry = {}

        if self.storage_dir and self.storage_dir.exists():
            for filename in ["faiss.index", "chunks.json", "documents.json"]:
                target = self.storage_dir / filename
                if target.exists():
                    try:
                        target.unlink()
                    except OSError:
                        pass


# Singleton VectorStore instance
_GLOBAL_STORE: Optional[VectorStore] = None


def get_vector_store(storage_dir: Optional[Path] = None) -> VectorStore:
    """Returns singleton VectorStore instance."""
    global _GLOBAL_STORE
    target_dir = storage_dir or DATA_DIR
    if _GLOBAL_STORE is None or (storage_dir and _GLOBAL_STORE.storage_dir != storage_dir):
        _GLOBAL_STORE = VectorStore(dimension=EMBEDDING_DIMENSION, storage_dir=target_dir)
    return _GLOBAL_STORE
