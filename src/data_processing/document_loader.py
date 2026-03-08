from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.exceptions import NotFoundError, ValidationError
from src.common.logger import get_logger

_BASE64_IMAGE_PATTERN = re.compile(r"!\[\]\(data:image/[^)]+\)")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[\u3002\uff01\uff1f\u002e\u0021\u003f])\s*")


@dataclass
class Document:
    source_path: Path
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """A semantically meaningful section of a document."""

    content: str
    heading_chain: list[str]
    level: int
    index: int


class DocumentLoader:
    def __init__(self, strip_images: bool = True) -> None:
        """Initialize the loader with base64 image stripping behavior."""

        self.strip_images = strip_images
        self._logger = get_logger(__name__)

    def load(self, path: Path | str) -> Document:
        """Load a single .docx file and return a Document instance."""

        file_path = Path(path)
        self._validate_path(file_path)
        content = self._convert_docx(file_path)
        metadata = self._build_metadata(file_path)
        return Document(source_path=file_path, content=content, metadata=metadata)

    def load_directory(
        self, directory: Path | str, pattern: str = "*.docx"
    ) -> list[Document]:
        """Load all .docx files in a directory matching the pattern."""

        dir_path = Path(directory)
        if not dir_path.exists():
            raise NotFoundError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise ValidationError(f"Path is not a directory: {dir_path}")

        documents: list[Document] = []
        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file():
                documents.append(self.load(file_path))
        return documents

    def _convert_docx(self, path: Path) -> str:
        self._require_docx(path)
        try:
            markitdown = importlib.import_module("markitdown")
        except ModuleNotFoundError as exc:
            message = (
                "markitdown is required to load .docx files. "
                "Install it with `uv add --group data markitdown[docx]`."
            )
            raise ModuleNotFoundError(message) from exc

        converter = markitdown.MarkItDown()
        result = converter.convert(str(path))
        content = result.text_content
        if self.strip_images:
            content = self._strip_base64_images(content)
        return content

    def _strip_base64_images(self, content: str) -> str:
        return _BASE64_IMAGE_PATTERN.sub("[图片]", content)

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise NotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        self._require_docx(path)

    def _require_docx(self, path: Path) -> None:
        if path.suffix.lower() != ".docx":
            raise ValidationError(f"Unsupported file format: {path.suffix}")

    def _build_metadata(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "filename": path.name,
            "file_size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    # ── Semantic Section Splitting ──────────────────────────────────

    def split_into_sections(
        self, content: str, max_chunk_size: int = 1500
    ) -> list[Section]:
        """Split markdown content into semantic sections by headings."""

        if not content.strip():
            return []

        headings = list(_HEADING_PATTERN.finditer(content))

        if not headings:
            return self._split_plain_text(content, max_chunk_size)

        sections: list[Section] = []
        heading_stack: list[tuple[int, str]] = []

        # Handle content before the first heading
        pre_content = content[: headings[0].start()].strip()
        if pre_content:
            for chunk in self._ensure_size(pre_content, max_chunk_size):
                sections.append(
                    Section(
                        content=chunk, heading_chain=[], level=0, index=len(sections)
                    )
                )

        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Content between this heading and the next (or EOF)
            body_start = match.end()
            body_end = (
                headings[i + 1].start() if i + 1 < len(headings) else len(content)
            )
            body = content[body_start:body_end].strip()

            # Maintain heading stack (pop siblings / children of same level+)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            chain = [h[1] for h in heading_stack]

            if not body:
                continue

            for chunk in self._ensure_size(body, max_chunk_size):
                sections.append(
                    Section(
                        content=chunk,
                        heading_chain=list(chain),
                        level=level,
                        index=len(sections),
                    )
                )

        return sections

    # ── Internal helpers ───────────────────────────────────────────

    def _ensure_size(self, text: str, max_size: int) -> list[str]:
        """Return text as-is if small enough, otherwise split."""
        if len(text) <= max_size:
            return [text]
        return self._split_long_section(text, max_size)

    def _split_long_section(self, text: str, max_size: int) -> list[str]:
        """Split by paragraphs → sentences → characters.

        Uses hierarchical separators: ``\\n\\n`` → ``\\n`` → sentence
        punctuation → fixed-window characters.
        """

        # Try double-newline first (standard markdown paragraph break)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Fallback: single newline (common in Chinese docs / markitdown output)
        if len(paragraphs) <= 1:
            single = [p.strip() for p in text.split("\n") if p.strip()]
            if len(single) > len(paragraphs):
                paragraphs = single

        # If every paragraph fits, just merge small ones together
        if all(len(p) <= max_size for p in paragraphs):
            return self._merge_small(paragraphs, max_size)

        # Some paragraphs are still too long – break them by sentences
        pieces: list[str] = []
        for para in paragraphs:
            if len(para) <= max_size:
                pieces.append(para)
            else:
                pieces.extend(self._split_by_sentences(para, max_size))

        return self._merge_small(pieces, max_size)

    def _split_by_sentences(self, text: str, max_size: int) -> list[str]:
        """Split text on sentence-ending punctuation."""

        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return self._split_by_chars(text, max_size)

        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(sent) > max_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_by_chars(sent, max_size))
            elif len(current) + len(sent) > max_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = current + sent if current else sent
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _split_by_chars(text: str, max_size: int) -> list[str]:
        """Last resort: fixed-window character split with small overlap."""

        overlap = min(100, max_size // 10)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _merge_small(chunks: list[str], max_size: int) -> list[str]:
        """Merge consecutive small chunks up to *max_size*."""

        merged: list[str] = []
        current = ""
        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current) + len(chunk) + 2 <= max_size:
                current = current + "\n\n" + chunk
            else:
                merged.append(current)
                current = chunk
        if current:
            merged.append(current)
        return merged

    def _split_plain_text(self, text: str, max_size: int) -> list[Section]:
        """Fallback for documents with no markdown headings.

        Always detects paragraph boundaries (``\\n\\n`` then ``\\n``)
        before falling back to size-based splitting, so that chunks
        respect the natural structure of the document.
        """

        # Detect paragraphs: prefer \n\n, fallback to \n
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        has_strong_breaks = len(paragraphs) > 1
        if not has_strong_breaks:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        if len(paragraphs) > 1:
            # Split oversized paragraphs by sentences
            pieces: list[str] = []
            for para in paragraphs:
                if len(para) <= max_size:
                    pieces.append(para)
                else:
                    pieces.extend(self._split_by_sentences(para, max_size))
            if has_strong_breaks:
                # \n\n = explicit paragraph boundary → each paragraph
                # stays as its own section (no merging)
                chunks = pieces
            else:
                # \n = weaker boundary → merge small lines together
                merge_target = max(max_size // 3, 300)
                chunks = self._merge_small(pieces, merge_target)
        else:
            # No paragraph breaks at all — pure size-based splitting
            chunks = self._ensure_size(text, max_size)

        return [
            Section(content=c, heading_chain=[], level=0, index=i)
            for i, c in enumerate(chunks)
        ]
