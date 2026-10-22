"""
Embedding generation module for Document Intelligence.
Provides dense vector embeddings using Google Gemini text-embedding-004
with content-hash caching, robust error handling, and test mock support.
"""
from abc import ABC, abstractmethod
import hashlib
import os
from typing import Dict, List, Optional
import faiss
import numpy as np
import requests

from rag.config import EMBEDDING_DIMENSION, EMBEDDING_PROVIDER, GEMINI_EMBEDDING_MODEL


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when required embedding credentials or configurations are missing."""
    pass


class EmbeddingAPIError(EmbeddingError):
    """Raised when an embedding API returns an error, timeout, or rate-limit."""
    pass


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings into an (N, dimension) float32 numpy array.
        Output vectors MUST be finite, non-zero, and L2-normalized.
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string into a (1, dimension) float32 numpy array.
        Output vector MUST be L2-normalized.
        """
        return self.embed_texts([query])


class GeminiEmbedder(BaseEmbedder):
    """
    Real dense embedding provider using Google Gemini's official REST API.
    Model: text-embedding-004 (768 dimensions).
    Features:
    - Content-hash vector caching to avoid re-embedding duplicate/unchanged chunks.
    - Explicit rate-limit (429), timeout, and error handling.
    - Strict validation of finite numeric values and dimensionality.
    - Zero credential exposure in exception messages.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise EmbeddingConfigurationError(
                "Embedding provider 'gemini' requires an API key. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in the server environment."
            )
        self.model = model
        self._dimension = dimension
        self._single_endpoint = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent"
        # In-memory content hash cache: sha256(text) -> List[float]
        self._cache: Dict[str, List[float]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_single(self, text: str) -> List[float]:
        # Check content-hash cache
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if content_hash in self._cache:
            return self._cache[content_hash]

        url = f"{self._single_endpoint}?key={self.api_key}"
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text[:4000]}]}
        }

        try:
            resp = requests.post(url, json=payload, timeout=25)
        except requests.exceptions.Timeout as e:
            raise EmbeddingAPIError("Gemini embedding API request timed out.") from e
        except requests.exceptions.RequestException as e:
            # Mask API key from error string if present
            err_str = str(e).replace(self.api_key, "[REDACTED]")
            raise EmbeddingAPIError(f"Embedding network transport error: {err_str}") from e

        if resp.status_code == 429:
            raise EmbeddingAPIError("Gemini embedding API rate limit reached (HTTP 429). Please retry shortly.")
        elif resp.status_code != 200:
            safe_err = resp.text.replace(self.api_key, "[REDACTED]")
            raise EmbeddingAPIError(f"Gemini embedding API error ({resp.status_code}): {safe_err}")

        try:
            data = resp.json()
        except Exception as e:
            raise EmbeddingAPIError("Failed to parse JSON response from Gemini embedding API.") from e

        values = data.get("embedding", {}).get("values")
        if not values or not isinstance(values, list):
            raise EmbeddingAPIError("Malformed response payload: missing 'embedding.values'.")

        if len(values) != self._dimension:
            raise EmbeddingAPIError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {len(values)}."
            )

        # Validate finite numeric values
        float_values = [float(v) for v in values]
        if not all(np.isfinite(float_values)):
            raise EmbeddingAPIError("Received non-finite (NaN or Inf) values in embedding vector.")

        # Cache result
        self._cache[content_hash] = float_values
        return float_values

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        vectors: List[List[float]] = []
        for text in texts:
            clean_text = text.strip() or " "
            vectors.append(self._embed_single(clean_text))

        arr = np.array(vectors, dtype=np.float32)

        # Validate non-zero vectors
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms < 1e-9):
            raise EmbeddingAPIError("Encountered near-zero vector during embedding generation.")

        faiss.normalize_L2(arr)
        return arr


class MockDeterministicEmbedder(BaseEmbedder):
    """
    Deterministic dense vector embedder for automated tests and offline verification.
    Clearly marked for testing: produces non-random, deterministic 768-dim float32 vectors
    based on token and n-gram hash projections with L2 unit normalization.
    """
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _text_to_vector(self, text: str) -> np.ndarray:
        clean = text.lower().strip()
        vec = np.zeros(self._dimension, dtype=np.float32)

        if not clean:
            vec[0] = 1.0
            return vec

        words = clean.split()
        for word in words:
            h = int(hashlib.sha256(word.encode('utf-8')).hexdigest()[:8], 16)
            idx = h % self._dimension
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign

        for i in range(max(0, len(clean) - 2)):
            gram = clean[i:i+3]
            h = int(hashlib.md5(gram.encode('utf-8')).hexdigest()[:6], 16)
            idx = h % self._dimension
            vec[idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        else:
            vec[0] = 1.0

        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        matrix = np.array([self._text_to_vector(t) for t in texts], dtype=np.float32)
        faiss.normalize_L2(matrix)
        return matrix


# Backwards compatibility alias for existing test imports
DeterministicOfflineEmbedder = MockDeterministicEmbedder


def get_default_embedder(allow_mock: bool = False) -> BaseEmbedder:
    """
    Factory to retrieve active embedder.
    If RAG_EMBEDDING_PROVIDER is 'mock' or allow_mock is True, returns MockDeterministicEmbedder.
    Otherwise attempts to configure GeminiEmbedder. If GEMINI_API_KEY is not set,
    raises EmbeddingConfigurationError with instructions.
    """
    provider = os.environ.get("RAG_EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()

    if provider == "mock" or allow_mock:
        return MockDeterministicEmbedder()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # If running in test or local without key, check if mock fallback is permitted
        if os.environ.get("DJANGO_TESTING") == "true":
            return MockDeterministicEmbedder()
        raise EmbeddingConfigurationError(
            "GEMINI_API_KEY is not configured in the server environment. "
            "Please set GEMINI_API_KEY in your environment or set RAG_EMBEDDING_PROVIDER='mock' for local offline testing."
        )

    return GeminiEmbedder(api_key=api_key)
