"""
API views for document ingestion, processing, and semantic retrieval.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from rag.chunker import TextChunker
from rag.config import (
    ALLOWED_EXTENSIONS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_UPLOAD_SIZE_BYTES,
)
from rag.embeddings import (
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    get_default_embedder,
)
from rag.extractor import extract_document, sanitize_filename, DocumentExtractionError
from rag.indexer import get_vector_store
from rag.retriever import DocumentRetriever, RetrievalValidationError


@csrf_exempt
@require_http_methods(["POST"])
def upload_document(request: HttpRequest) -> JsonResponse:
    """
    POST /api/upload/
    Accepts multipart/form-data upload containing a 'file' field (.pdf or .txt).
    Validates file size, extension, and content, extracting text and page metadata.
    Automatically chunks and indexes extracted text into the FAISS vector store.
    """
    if 'file' not in request.FILES:
        return JsonResponse(
            {
                'status': 'error',
                'error': "Missing upload file. Please provide a multipart/form-data request with 'file' key."
            },
            status=400
        )

    uploaded_file = request.FILES['file']
    raw_filename = uploaded_file.name or ""
    clean_filename = sanitize_filename(raw_filename)

    # Check file extension
    lower_name = clean_filename.lower()
    has_valid_ext = any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    if not has_valid_ext:
        return JsonResponse(
            {
                'status': 'error',
                'error': f"Unsupported file type for '{clean_filename}'. Only PDF (.pdf) and TXT (.txt) files are accepted."
            },
            status=400
        )

    # Check file size against 10 MB limit
    if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
        return JsonResponse(
            {
                'status': 'error',
                'error': (
                    f"File '{clean_filename}' exceeds the maximum allowed size of 10 MB "
                    f"({uploaded_file.size} bytes uploaded)."
                )
            },
            status=413
        )

    # Check empty file
    if uploaded_file.size == 0:
        return JsonResponse(
            {
                'status': 'error',
                'error': f"Uploaded file '{clean_filename}' is empty (0 bytes)."
            },
            status=400
        )

    # Read file content safely in memory
    try:
        file_bytes = uploaded_file.read()
    except Exception as e:
        return JsonResponse(
            {
                'status': 'error',
                'error': f"Failed to read uploaded file stream: {str(e)}"
            },
            status=400
        )

    # Perform text extraction
    try:
        extraction_result = extract_document(file_bytes, clean_filename)
    except DocumentExtractionError as e:
        return JsonResponse(
            {
                'status': 'error',
                'error': str(e)
            },
            status=400
        )
    except Exception as e:
        return JsonResponse(
            {
                'status': 'error',
                'error': f"Unexpected document processing error: {str(e)}"
            },
            status=500
        )

    # Generate deterministic document identifier based on content and filename
    doc_hash = hashlib.sha256(f"{clean_filename}:{len(file_bytes)}".encode('utf-8')).hexdigest()[:12]
    document_id = f"doc_{doc_hash}"

    # Chunk and index extracted text into FAISS vector store
    chunk_count = 0
    if extraction_result.segments and not extraction_result.is_empty:
        chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
        chunks = chunker.chunk_segments(extraction_result.segments, document_id, clean_filename)
        if chunks:
            try:
                store = get_vector_store()
                embedder = get_default_embedder()
                chunk_count = store.add_chunks(chunks, embedder)
            except EmbeddingConfigurationError as e:
                # If embedding provider is not configured, inform the user cleanly
                return JsonResponse(
                    {
                        'status': 'warning',
                        'document_id': document_id,
                        'chunk_count': 0,
                        'warning': f"Extraction succeeded, but indexing was skipped: {str(e)}",
                        **extraction_result.to_dict()
                    },
                    status=200
                )
            except Exception as e:
                return JsonResponse(
                    {
                        'status': 'error',
                        'error': f"Vector indexing failed: {str(e)}"
                    },
                    status=500
                )

    response_data = {
        'status': 'success',
        'document_id': document_id,
        'chunk_count': chunk_count,
        **extraction_result.to_dict()
    }

    return JsonResponse(response_data, status=200)


@csrf_exempt
@require_http_methods(["POST"])
def retrieve_passages(request: HttpRequest) -> JsonResponse:
    """
    POST /api/retrieve/
    Performs semantic retrieval against the FAISS vector index.

    Expected JSON body:
    {
        "query": "string (required)",
        "document_ids": ["doc_123"] (optional, filter to specific documents),
        "top_k": 5 (optional, default 5, max 20),
        "score_threshold": 0.40 (optional, cosine similarity threshold)
    }
    """
    try:
        body = json.loads(request.body.decode('utf-8') or '{}')
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {'status': 'error', 'error': 'Invalid JSON request payload.'},
            status=400
        )

    query = body.get('query', '')
    document_ids = body.get('document_ids')
    top_k = body.get('top_k', DEFAULT_TOP_K)
    score_threshold = body.get('score_threshold', DEFAULT_RELEVANCE_THRESHOLD)

    try:
        retriever = DocumentRetriever()
        response = retriever.retrieve(
            query=query,
            document_ids=document_ids,
            top_k=top_k,
            score_threshold=score_threshold
        )
        return JsonResponse(response.to_dict(), status=200)
    except RetrievalValidationError as e:
        return JsonResponse({'status': 'error', 'error': str(e)}, status=400)
    except EmbeddingConfigurationError as e:
        return JsonResponse({'status': 'error', 'error': str(e)}, status=503)
    except EmbeddingAPIError as e:
        return JsonResponse({'status': 'error', 'error': f"Embedding service error: {str(e)}"}, status=502)
    except Exception as e:
        return JsonResponse({'status': 'error', 'error': f"Retrieval error: {str(e)}"}, status=500)


@require_http_methods(["GET"])
def list_documents(request: HttpRequest) -> JsonResponse:
    """
    GET /api/documents/
    Returns catalog of indexed documents and vector count.
    """
    store = get_vector_store()
    return JsonResponse(
        {
            'status': 'success',
            'documents': list(store.documents_registry.values()),
            'total_vectors': store.total_vectors
        },
        status=200
    )
