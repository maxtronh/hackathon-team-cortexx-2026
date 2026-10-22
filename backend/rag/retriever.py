"""
Semantic Retrieval Engine for Document Intelligence.
Validates search requests, applies document isolation filters and relevance score thresholds,
and returns structured evidence passages with source citations.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from rag.config import DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K, MAX_TOP_K
from rag.embeddings import BaseEmbedder, get_default_embedder
from rag.indexer import SearchResult, VectorStore, get_vector_store


class RetrievalValidationError(Exception):
    """Raised when query payload fails validation."""
    pass


@dataclass
class RetrievalResponse:
    status: str
    query: str
    count: int
    results: List[Dict[str, Any]]
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentRetriever:
    """
    Orchestrates semantic search queries against the persistent vector store.
    """
    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[BaseEmbedder] = None
    ):
        self.store = store or get_vector_store()
        self.embedder = embedder or get_default_embedder()

    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    ) -> RetrievalResponse:
        """
        Executes semantic retrieval with strict input validation and document isolation.

        Parameters:
        - query: Non-empty search string
        - document_ids: Optional list restricting retrieval to specific document IDs
        - top_k: Maximum candidate passages to return (1 to MAX_TOP_K)
        - score_threshold: Minimum cosine similarity score [-1.0, 1.0] to qualify as relevant
        """
        # Validate query
        if not isinstance(query, str) or not query.strip():
            raise RetrievalValidationError("Search query must be a non-empty string.")

        clean_query = query.strip()

        # Validate and sanitize document_ids
        sanitized_doc_ids: Optional[List[str]] = None
        if document_ids is not None:
            if isinstance(document_ids, str):
                sanitized_doc_ids = [document_ids.strip()]
            elif isinstance(document_ids, (list, tuple)):
                sanitized_doc_ids = [str(d).strip() for d in document_ids if str(d).strip()]
                if not sanitized_doc_ids:
                    sanitized_doc_ids = None
            else:
                raise RetrievalValidationError("Field 'document_ids' must be a list of document IDs or a string.")

        # Bound top_k
        try:
            bounded_k = int(top_k)
            bounded_k = max(1, min(bounded_k, MAX_TOP_K))
        except (ValueError, TypeError):
            bounded_k = DEFAULT_TOP_K

        # Validate score_threshold
        try:
            threshold = float(score_threshold)
        except (ValueError, TypeError):
            threshold = DEFAULT_RELEVANCE_THRESHOLD

        # If vector store is empty, return clear no-results response without error
        if self.store.total_vectors == 0:
            return RetrievalResponse(
                status="success",
                query=clean_query,
                count=0,
                results=[],
                message="No documents have been indexed yet."
            )

        # Execute search
        results: List[SearchResult] = self.store.search(
            query=clean_query,
            embedder=self.embedder,
            top_k=bounded_k,
            document_ids=sanitized_doc_ids,
            score_threshold=threshold
        )

        message = None
        if not results:
            message = f"No evidence passages matched the query with relevance >= {threshold:.2f}."

        return RetrievalResponse(
            status="success",
            query=clean_query,
            count=len(results),
            results=[r.to_dict() for r in results],
            message=message
        )
