"""
Document text chunking module.
Splits extracted document segments into sentence-aware, overlapping chunks
while preserving page-level and source document metadata for accurate citations.
"""
from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Dict, List, Optional
from rag.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from rag.extractor import TextSegment


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    character_count: int
    chunk_index: int
    chunk_position: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextChunker:
    """
    Sentence-aware, paragraph-respecting text chunker.

    Design choices:
    - Default chunk_size: 1,000 characters (~150-200 words).
      Provides enough context for complex legal, scientific, or technical concepts.
    - Default chunk_overlap: 150 characters (~20-30 words).
      Guarantees multi-sentence clauses and cross-boundary references are preserved.
    """
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """
        Splits clean text into bounded overlapping chunks respecting natural boundaries.
        Guarantees termination (no infinite loops) and excludes empty chunks.
        """
        # Normalize excessive whitespace but preserve newlines
        clean_text = re.sub(r'[ \t]+', ' ', text).strip()
        if not clean_text:
            return []

        if len(clean_text) <= self.chunk_size:
            return [clean_text]

        chunks: List[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        total_len = len(clean_text)

        while start < total_len:
            end = min(start + self.chunk_size, total_len)
            candidate = clean_text[start:end]

            # If not at the very end of the text, seek natural boundaries in the overlap zone
            if end < total_len:
                # Look for boundaries in the last 30% of the candidate chunk
                search_region_start = max(0, len(candidate) - int(self.chunk_overlap * 1.2))
                last_break = -1

                # Ordered by preference: paragraph -> sentence (including Hindi purna viram) -> clause -> word
                delimiters = [
                    '\n\n', '.\n', '.\n\n',
                    '. ', '? ', '! ', '। ',
                    '; ', ':\n', '\n', ', ', ' '
                ]

                for delim in delimiters:
                    idx = candidate.rfind(delim, search_region_start)
                    if idx != -1:
                        last_break = idx + len(delim)
                        break

                if last_break != -1 and last_break > int(self.chunk_size * 0.4):
                    end = start + last_break
                    candidate = clean_text[start:end]

            chunk_str = candidate.strip()
            if chunk_str:
                chunks.append(chunk_str)

            if end >= total_len:
                break

            # Advance by step, but at least forward past start to guarantee progression
            start = max(start + step, end - self.chunk_overlap)

        return chunks

    def chunk_segments(
        self,
        segments: List[TextSegment],
        document_id: str,
        filename: str
    ) -> List[DocumentChunk]:
        """
        Chunks a list of TextSegments while preserving source filename and page numbers.
        Segments are processed page-by-page to guarantee citations never cross page boundaries incorrectly.
        Deduplicates identical chunks within the same document and page.
        """
        all_chunks: List[DocumentChunk] = []
        seen_hashes = set()
        chunk_counter = 0

        for segment in segments:
            seg_text = segment.text.strip() if hasattr(segment, 'text') else str(segment).strip()
            page_num = segment.page if hasattr(segment, 'page') else 1
            source_file = segment.filename if hasattr(segment, 'filename') else filename

            if not seg_text:
                continue

            raw_chunks = self._split_text(seg_text)

            for raw_chunk in raw_chunks:
                # Deduplicate identical chunks on the same page
                chunk_hash = hashlib.sha256(f"{document_id}:{page_num}:{raw_chunk}".encode('utf-8')).hexdigest()
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)

                chunk_counter += 1
                chunk_id = f"{document_id}_p{page_num}_c{chunk_counter}"

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=source_file,
                        page=page_num,
                        text=raw_chunk,
                        character_count=len(raw_chunk),
                        chunk_index=chunk_counter,
                        chunk_position=chunk_counter
                    )
                )

        return all_chunks
