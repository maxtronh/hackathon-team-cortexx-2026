"""
Document extraction module for PDF and TXT files.
Extracts text segments with source metadata (filename, page numbers).
"""
import io
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import pypdf
from pypdf.errors import EmptyFileError, PdfReadError


class DocumentExtractionError(Exception):
    """Raised when document extraction fails due to formatting or read issues."""
    pass


@dataclass
class TextSegment:
    filename: str
    page: int
    text: str
    character_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    filename: str
    detected_type: str
    page_count: int
    extracted_character_count: int
    segments: List[TextSegment]
    is_empty: bool
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['segments'] = [s.to_dict() if isinstance(s, TextSegment) else s for s in self.segments]
        return data


def sanitize_filename(filename: Optional[str]) -> str:
    """
    Sanitize an untrusted user-supplied filename.
    Removes directory traversal components, path separators, and null bytes.
    """
    if not filename:
        return "uploaded_document"
    # Remove null bytes
    clean = filename.replace('\0', '').strip()
    # Strip any directory path components (both unix and windows)
    clean = os.path.basename(clean.replace('\\', '/'))
    # Remove non-whitelisted control characters but keep alphanumeric, dots, hyphens, underscores
    clean = re.sub(r'[^a-zA-Z0-9._\- ]', '_', clean).strip()
    return clean or "uploaded_document"


def extract_txt(file_bytes: bytes, filename: str) -> ExtractionResult:
    """
    Extract text from TXT bytes.
    Tries UTF-8 first, then UTF-8 with BOM, CP1252, Latin-1, and finally UTF-8 with errors replaced.
    """
    safe_name = sanitize_filename(filename)
    if not file_bytes:
        return ExtractionResult(
            filename=safe_name,
            detected_type="txt",
            page_count=1,
            extracted_character_count=0,
            segments=[TextSegment(filename=safe_name, page=1, text="", character_count=0)],
            is_empty=True,
            warning="Text file is empty."
        )

    text = None
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            text = file_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        # Documented safe fallback for corrupted or unknown encodings
        text = file_bytes.decode("utf-8", errors="replace")

    text = text.strip()
    char_count = len(text)
    is_empty = (char_count == 0)

    segment = TextSegment(
        filename=safe_name,
        page=1,
        text=text,
        character_count=char_count
    )

    return ExtractionResult(
        filename=safe_name,
        detected_type="txt",
        page_count=1,
        extracted_character_count=char_count,
        segments=[segment],
        is_empty=is_empty,
        warning="Text file contains only whitespace or no readable characters." if is_empty else None
    )


def extract_pdf(file_bytes: bytes, filename: str) -> ExtractionResult:
    """
    Extract page-indexed text from PDF bytes using pypdf.
    Validates PDF header and records 1-based page numbers.
    """
    safe_name = sanitize_filename(filename)

    if not file_bytes:
        raise DocumentExtractionError("PDF file is completely empty (0 bytes).")

    # Fast header check for valid PDF magic signature
    if not file_bytes.startswith(b'%PDF-'):
        # Some PDFs might have a small amount of whitespace/bom before %PDF-
        prefix = file_bytes[:1024].lstrip()
        if not prefix.startswith(b'%PDF-'):
            raise DocumentExtractionError("Invalid PDF signature: file does not begin with '%PDF-'.")

    stream = io.BytesIO(file_bytes)
    try:
        reader = pypdf.PdfReader(stream)
        num_pages = len(reader.pages)
    except (PdfReadError, EmptyFileError) as e:
        raise DocumentExtractionError(f"Malformed or unreadable PDF: {str(e)}")
    except Exception as e:
        raise DocumentExtractionError(f"Failed to parse PDF document: {str(e)}")

    if num_pages == 0:
        return ExtractionResult(
            filename=safe_name,
            detected_type="pdf",
            page_count=0,
            extracted_character_count=0,
            segments=[],
            is_empty=True,
            warning="PDF contains 0 pages."
        )

    segments: List[TextSegment] = []
    total_chars = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            # If a single page has font parsing errors, do not crash the entire document
            page_text = f"[Page {page_idx} extraction error: {str(e)}]"

        cleaned_page_text = page_text.strip()
        p_len = len(cleaned_page_text)
        total_chars += p_len
        segments.append(TextSegment(
            filename=safe_name,
            page=page_idx,
            text=cleaned_page_text,
            character_count=p_len
        ))

    is_empty = (total_chars == 0)
    warning = None
    if is_empty:
        warning = (
            "No extractable text found in PDF. The document may be empty, password-protected, "
            "or consist of scanned images/raster graphics without an embedded text layer."
        )

    return ExtractionResult(
        filename=safe_name,
        detected_type="pdf",
        page_count=num_pages,
        extracted_character_count=total_chars,
        segments=segments,
        is_empty=is_empty,
        warning=warning
    )


def extract_document(file_bytes: bytes, filename: str) -> ExtractionResult:
    """
    Main extraction dispatcher based on file extension and magic signature.
    """
    safe_name = sanitize_filename(filename)
    lower_name = safe_name.lower()

    if lower_name.endswith('.pdf'):
        return extract_pdf(file_bytes, safe_name)
    elif lower_name.endswith('.txt'):
        return extract_txt(file_bytes, safe_name)
    else:
        raise DocumentExtractionError(
            f"Unsupported file format '{os.path.splitext(safe_name)[1]}'. Supported formats are .pdf and .txt."
        )
