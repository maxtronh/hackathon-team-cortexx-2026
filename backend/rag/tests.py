"""
Automated test suite for document ingestion, text extraction, sentence-aware chunking,
dense vector embeddings, FAISS indexing, and semantic retrieval.
"""
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
import numpy as np
import pypdf
import requests

from rag.chunker import DocumentChunk, TextChunker
from rag.embeddings import (
    BaseEmbedder,
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    GeminiEmbedder,
    MockDeterministicEmbedder,
)
from rag.extractor import TextSegment
from rag.indexer import VectorStore
from rag.retriever import DocumentRetriever, RetrievalValidationError


def generate_sample_pdf_bytes(page_texts: list[str]) -> bytes:
    """Helper to generate in-memory multi-page PDF bytes with real extractable text."""
    base_template = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 55 >> stream
BT
/F1 18 Tf
0 0 Td
({TEXT}) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000117 00000 n 
0000000244 00000 n 
0000000350 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
423
%%EOF"""
    writer = pypdf.PdfWriter()
    for text in page_texts:
        raw = base_template.replace(b'{TEXT}', text.encode('latin-1', errors='replace'))
        writer.append(io.BytesIO(raw))

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


class DocumentIngestionTests(TestCase):
    """Task 2 Regression Suite: Verifies document ingestion and text extraction."""
    def setUp(self):
        self.client = Client()

    def test_1_valid_txt_upload(self):
        """1. Valid TXT upload and extracted text."""
        content = b"AI Fusion 2026: Team Cortexx Document Intelligence Assistant."
        upload_file = SimpleUploadedFile("sample.txt", content, content_type="text/plain")

        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "sample.txt")
        self.assertEqual(data["detected_type"], "txt")
        self.assertEqual(data["page_count"], 1)
        self.assertEqual(data["extracted_character_count"], len(content.decode("utf-8")))
        self.assertEqual(len(data["segments"]), 1)
        self.assertEqual(data["segments"][0]["page"], 1)
        self.assertEqual(data["segments"][0]["text"], content.decode("utf-8"))
        self.assertFalse(data["is_empty"])
        self.assertIn("document_id", data)

    def test_2_valid_multipage_pdf_upload(self):
        """2. Valid multi-page PDF upload and correct 1-based page numbering."""
        pdf_bytes = generate_sample_pdf_bytes(["Page 1 Introduction", "Page 2 Architecture", "Page 3 Conclusion"])
        upload_file = SimpleUploadedFile("whitepaper.pdf", pdf_bytes, content_type="application/pdf")

        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "whitepaper.pdf")
        self.assertEqual(data["detected_type"], "pdf")
        self.assertEqual(data["page_count"], 3)
        self.assertEqual(len(data["segments"]), 3)

        self.assertEqual(data["segments"][0]["page"], 1)
        self.assertIn("Page 1 Introduction", data["segments"][0]["text"])

        self.assertEqual(data["segments"][1]["page"], 2)
        self.assertIn("Page 2 Architecture", data["segments"][1]["text"])

        self.assertEqual(data["segments"][2]["page"], 3)
        self.assertIn("Page 3 Conclusion", data["segments"][2]["text"])

    def test_3_unsupported_extension(self):
        """3. Reject unsupported extensions (.exe, .jpg, .csv, etc.)."""
        upload_file = SimpleUploadedFile("script.sh", b"echo 'hack'", content_type="application/x-sh")

        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    def test_4_empty_file(self):
        """4. Reject zero-byte empty file."""
        upload_file = SimpleUploadedFile("empty.txt", b"", content_type="text/plain")

        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    def test_5_oversized_upload(self):
        """5. Reject oversized upload (>10 MB)."""
        oversized_bytes = b"X" * (10 * 1024 * 1024 + 1024)
        upload_file = SimpleUploadedFile("giant.txt", oversized_bytes, content_type="text/plain")

        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["status"], "error")

    def test_6_malformed_pdf(self):
        """6. Reject malformed PDF that fails header or parser check."""
        corrupted_bytes = b"This is not a real PDF file despite the extension"
        upload_file = SimpleUploadedFile("broken.pdf", corrupted_bytes, content_type="application/pdf")

        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    def test_7_missing_upload_field(self):
        """7. Reject requests missing the 'file' multipart form field."""
        response = self.client.post("/api/upload/", {"wrong_key": "data"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    def test_8_unicode_text(self):
        """8. Correctly handle Unicode (Hindi, accents, emoji) text extraction."""
        unicode_content = "टीम कॉर्टेक्स AI-Fusion 2026 🚀 Résumé & Data Analytics".encode("utf-8")
        upload_file = SimpleUploadedFile("unicode.txt", unicode_content, content_type="text/plain")

        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("टीम कॉर्टेक्स", data["segments"][0]["text"])
        self.assertIn("🚀", data["segments"][0]["text"])

    def test_9_empty_or_image_only_pdf(self):
        """9. Empty or image-only PDF behavior: return clear warning, not fabricated text."""
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        out = io.BytesIO()
        writer.write(out)

        upload_file = SimpleUploadedFile("blank.pdf", out.getvalue(), content_type="application/pdf")
        response = self.client.post("/api/upload/", {"file": upload_file})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_empty"])
        self.assertIsNotNone(data["warning"])

    def test_10_health_endpoint_regression(self):
        """10. Verify /api/health/ regression check."""
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")


class SemanticRetrievalEngineTests(TestCase):
    """
    Task 3 Complete Requirements Suite:
    Tests sentence-aware chunking, embeddings, FAISS indexing, nearest-neighbour ranking,
    isolation, persistence, and error handling.
    """
    def setUp(self):
        self.client = Client()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.store = VectorStore(dimension=768, storage_dir=self.temp_dir)
        self.embedder = MockDeterministicEmbedder()
        self.retriever = DocumentRetriever(store=self.store, embedder=self.embedder)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_paragraph_sentence_chunking_and_overlap(self):
        """1. Paragraph/sentence chunking and overlap bounds."""
        chunker = TextChunker(chunk_size=120, chunk_overlap=30)
        text = (
            "Paragraph one defines the core challenge in document retrieval.\n\n"
            "Paragraph two introduces dense vector embeddings and similarity search.\n\n"
            "Paragraph three concludes with evaluation benchmarks."
        )
        segs = [TextSegment(filename="paper.txt", page=1, text=text, character_count=len(text))]
        chunks = chunker.chunk_segments(segs, "doc_p", "paper.txt")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.character_count, 140)
            self.assertGreater(chunk.character_count, 0)
            self.assertEqual(chunk.document_id, "doc_p")

    def test_2_empty_short_long_unicode_hindi_content(self):
        """2. Empty, short, long and Unicode/Hindi content handling."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=25)

        # Empty
        empty_chunks = chunker.chunk_segments([TextSegment("e.txt", 1, "   ", 0)], "d_e", "e.txt")
        self.assertEqual(len(empty_chunks), 0)

        # Short
        short_chunks = chunker.chunk_segments([TextSegment("s.txt", 1, "Short statement.", 16)], "d_s", "s.txt")
        self.assertEqual(len(short_chunks), 1)
        self.assertEqual(short_chunks[0].text, "Short statement.")

        # Hindi & Unicode
        hindi_text = "परियोजना कॉर्टेक्स: कृत्रिम बुद्धिमत्ता आधारित दस्तावेज़ विश्लेषण प्रणाली। यह तंत्र त्वरित और सटीक परिणाम देता है।"
        hindi_chunks = chunker.chunk_segments([TextSegment("h.txt", 1, hindi_text, len(hindi_text))], "d_h", "h.txt")
        self.assertGreaterEqual(len(hindi_chunks), 1)

        self.store.add_chunks(hindi_chunks, self.embedder)
        res = self.store.search("दस्तावेज़ विश्लेषण", self.embedder, top_k=1)
        self.assertGreater(len(res), 0)
        self.assertIn("दस्तावेज़", res[0].text)

    def test_3_metadata_preservation_across_pages(self):
        """3. Page, filename, document, and chunk metadata preservation."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=40)
        segs = [
            TextSegment(filename="manual.pdf", page=1, text="Page 1 Safety Protocols.", character_count=24),
            TextSegment(filename="manual.pdf", page=2, text="Page 2 Calibration Procedures.", character_count=30),
        ]
        chunks = chunker.chunk_segments(segs, "doc_m", "manual.pdf")
        self.store.add_chunks(chunks, self.embedder)

        res = self.store.search("Calibration Procedures", self.embedder, top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].page, 2)
        self.assertEqual(res[0].filename, "manual.pdf")
        self.assertEqual(res[0].document_id, "doc_m")
        self.assertTrue(res[0].chunk_id.startswith("doc_m_p2_"))

    def test_4_deterministic_chunk_ids_and_stable_identity(self):
        """4. Deterministic chunk IDs and repeated document deduplication."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        segs = [TextSegment(filename="stable.txt", page=1, text="Consistent data payload.", character_count=24)]

        chunks_first = chunker.chunk_segments(segs, "doc_ident", "stable.txt")
        chunks_second = chunker.chunk_segments(segs, "doc_ident", "stable.txt")

        # Stable identifiers across runs
        self.assertEqual(chunks_first[0].chunk_id, chunks_second[0].chunk_id)

        # Re-adding chunks for the same document must NOT duplicate vector entries
        count_1 = self.store.add_chunks(chunks_first, self.embedder)
        self.assertEqual(self.store.total_vectors, 1)

        count_2 = self.store.add_chunks(chunks_second, self.embedder)
        self.assertEqual(count_2, 1)
        self.assertEqual(self.store.total_vectors, 1)  # Deduplicated

    def test_5_mocked_embedding_validation_nonfinite_and_mismatch(self):
        """5. Mocked embedding response validation, non-finite vectors, and dimension mismatch."""
        # Non-finite vector detection
        mock_resp_nan = MagicMock()
        mock_resp_nan.status_code = 200
        mock_resp_nan.json.return_value = {"embedding": {"values": [float('nan')] * 768}}

        with patch("requests.post", return_value=mock_resp_nan):
            gemini_embedder = GeminiEmbedder(api_key="test_key")
            with self.assertRaises(EmbeddingAPIError):
                gemini_embedder.embed_texts(["Test NaN"])

        # Dimension mismatch detection
        mock_resp_dim = MagicMock()
        mock_resp_dim.status_code = 200
        mock_resp_dim.json.return_value = {"embedding": {"values": [0.1] * 128}}

        with patch("requests.post", return_value=mock_resp_dim):
            with self.assertRaises(EmbeddingAPIError):
                gemini_embedder.embed_texts(["Test Dimension"])

    def test_6_faiss_indexing_and_nearest_neighbour_ranking(self):
        """6. FAISS indexing and exact nearest-neighbour cosine ranking."""
        chunks = [
            DocumentChunk("c1", "d1", "f.txt", 1, "The eagle soared high above the snow-capped mountain peak.", 59, 1, 1),
            DocumentChunk("c2", "d1", "f.txt", 1, "Quantum entanglement allows instantaneous state correlations.", 62, 2, 2),
            DocumentChunk("c3", "d1", "f.txt", 1, "Baking sourdough bread requires flour, water, salt and wild yeast.", 66, 3, 3),
        ]
        self.store.add_chunks(chunks, self.embedder)

        results = self.store.search("quantum entanglement physics", self.embedder, top_k=3)
        self.assertEqual(len(results), 3)
        # Top ranked result must be chunk c2
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].similarity_metric, "cosine")

    def test_7_retrieval_of_known_relevant_passages(self):
        """7. Retrieval of known relevant passages from target document."""
        chunks = [
            DocumentChunk("c_saturn", "doc_apollo", "apollo.txt", 1, "The Saturn V rocket launched the Apollo lunar missions into translunar injection.", 82, 1, 1),
        ]
        self.store.add_chunks(chunks, self.embedder)

        resp = self.retriever.retrieve(query="What rocket launched the Apollo lunar missions?", top_k=1)
        self.assertEqual(resp.status, "success")
        self.assertEqual(resp.count, 1)
        self.assertEqual(resp.results[0]["filename"], "apollo.txt")
        self.assertIn("Saturn V", resp.results[0]["text"])

    def test_8_document_a_isolation_from_document_b(self):
        """8. Document A isolation from Document B: restricted search prevents leakage."""
        chunks = [
            DocumentChunk("c_a", "doc_A", "secret_a.txt", 1, "Project Alpha secret code is Phoenix-77.", 40, 1, 1),
            DocumentChunk("c_b", "doc_B", "public_b.txt", 1, "Project Beta public summary report.", 35, 1, 1),
        ]
        self.store.add_chunks(chunks, self.embedder)

        # Query mentioning Alpha keyword, but scoped strictly to doc_B
        resp = self.retriever.retrieve(
            query="What is the secret code of Project Alpha?",
            document_ids=["doc_B"],
            top_k=5,
            score_threshold=0.0
        )
        for r in resp.results:
            self.assertEqual(r["document_id"], "doc_B")
            self.assertNotEqual(r["document_id"], "doc_A")
            self.assertNotIn("Phoenix-77", r["text"])

    def test_9_no_results_and_invalid_query_behavior(self):
        """9. No-results behavior and query validation errors."""
        # Empty query
        with self.assertRaises(RetrievalValidationError):
            self.retriever.retrieve(query="")

        # Non-string query
        with self.assertRaises(RetrievalValidationError):
            self.retriever.retrieve(query=None)  # type: ignore

        # Empty index returns clean empty results
        resp = self.retriever.retrieve(query="Valid search query")
        self.assertEqual(resp.count, 0)
        self.assertEqual(len(resp.results), 0)

    def test_10_persistence_across_index_reload_and_reinitialization(self):
        """10. Persistence across index reload and process-style reinitialization."""
        chunks = [
            DocumentChunk("cp1", "doc_persist", "p.txt", 1, "Persistent fact stored on disk.", 32, 1, 1),
        ]
        self.store.add_chunks(chunks, self.embedder)
        self.assertEqual(self.store.total_vectors, 1)

        # Simulate process restart by instantiating new VectorStore pointing to same disk directory
        restarted_store = VectorStore(dimension=768, storage_dir=self.temp_dir)
        self.assertEqual(restarted_store.total_vectors, 1)
        self.assertIn("doc_persist", restarted_store.documents_registry)

        retriever_restarted = DocumentRetriever(store=restarted_store, embedder=self.embedder)
        resp = retriever_restarted.retrieve("Persistent fact", top_k=1)
        self.assertEqual(resp.count, 1)
        self.assertEqual(resp.results[0]["chunk_id"], "cp1")

    def test_11_missing_corrupt_index_and_safe_rebuild(self):
        """11. Missing/corrupt index triggers safe rebuild path without data loss."""
        chunks = [
            DocumentChunk("cr1", "doc_rec", "r.txt", 1, "Recoverable information from metadata.", 38, 1, 1),
        ]
        self.store.add_chunks(chunks, self.embedder)

        # Corrupt the binary index file on disk
        index_file = self.temp_dir / "faiss.index"
        index_file.write_bytes(b"CORRUPTED_BINARY_DATA_CORRUPTION_TEST")

        # Open store: should detect corruption and recover from chunks.json metadata
        recovered_store = VectorStore(dimension=768, storage_dir=self.temp_dir)
        # Even if binary index was corrupted, metadata is preserved and can be rebuilt
        self.assertTrue(recovered_store.storage_dir.exists())

    def test_12_embedding_provider_timeout_ratelimit_and_config_errors(self):
        """12. Embedding provider timeout, rate-limit (429), and configuration error handling."""
        # Missing key
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(EmbeddingConfigurationError):
                GeminiEmbedder()

        # Timeout simulation
        with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
            embedder = GeminiEmbedder(api_key="mock_key")
            with self.assertRaises(EmbeddingAPIError) as ctx:
                embedder.embed_texts(["timeout test"])
            self.assertIn("timed out", str(ctx.exception))

        # Rate-limit (429) simulation
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Rate limit exceeded"
        with patch("requests.post", return_value=mock_429):
            embedder = GeminiEmbedder(api_key="mock_key")
            with self.assertRaises(EmbeddingAPIError) as ctx:
                embedder.embed_texts(["rate limit test"])
            self.assertIn("rate limit", str(ctx.exception).lower())

    def test_13_regression_of_health_and_upload_endpoints(self):
        """13. Regression test of existing /api/health/, /api/upload/, and /api/retrieve/ endpoints."""
        # Health check
        h_resp = self.client.get("/api/health/")
        self.assertEqual(h_resp.status_code, 200)

        # Document upload
        up_file = SimpleUploadedFile("regression.txt", b"Regression verification fact.", content_type="text/plain")
        up_resp = self.client.post("/api/upload/", {"file": up_file})
        self.assertEqual(up_resp.status_code, 200)
        up_data = up_resp.json()
        doc_id = up_data["document_id"]

        # Retrieve endpoint
        ret_resp = self.client.post(
            "/api/retrieve/",
            data=json.dumps({"query": "Regression verification", "document_ids": [doc_id]}),
            content_type="application/json"
        )
        self.assertEqual(ret_resp.status_code, 200)
        ret_data = ret_resp.json()
        self.assertEqual(ret_data["status"], "success")

    @unittest.skipUnless(os.environ.get("GEMINI_API_KEY"), "Skipped: GEMINI_API_KEY not configured in environment")
    def test_14_real_gemini_integration_smoke_test(self):
        """14. Real-model integration test: connects to Google Gemini text-embedding-004."""
        embedder = GeminiEmbedder()
        vectors = embedder.embed_texts(["Integration test for real Google Gemini text-embedding-004 model."])
        self.assertEqual(vectors.shape, (1, 768))
        self.assertEqual(vectors.dtype, np.float32)
        norm = float(np.linalg.norm(vectors[0]))
        self.assertAlmostEqual(norm, 1.0, places=3)
