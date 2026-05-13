"""
Text chunking module.

Reads raw chunk from a .txt or .pdf file and splits it into overlapping chunks
using strategies:
-   SLIDING_WINDOW
-   RECURSIVE_SPLIT
"""

import re
from pathlib import Path
from typing import List, Dict
from src.config import CHUNK_CFG, ChunkingConfig, ChunkStrategy

class DocumentChunker:
    """
    Reads a document and produces overlapping text chunks.

    Args:
        chunk_size: target character per chunk
        chunk_overlap: character overlap between consecutive chunks
        strategy: chunkStrategy to apply
    """
    _SEPARATORS: List[str] = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
            self,
            chunk_size: int = CHUNK_CFG.chunk_size,
            chunk_overlap: int = CHUNK_CFG.chunk_overlap,
            strategy: str = CHUNK_CFG.strategy
        ) -> None:

        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy
        )
        
    def chunk_file(self, file_path: str | Path) -> List[Dict[str, str]]:
        """
        Read a .txt or .pdf file and return list of chunk dicts.
        Each dict: {'id': "chunk_0001", 'text': '...'}
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        
        raw_text = self._read_file(path)
        return self._to_records(self._split(raw_text))

    def chunk_text(self, text: str, source: str = "inline") -> List[Dict[str, str]]:
        """
        Chunk a raw string directly
        """
        if not text.strip():
            raise ValueError("Input text is empty")
        chunks = self._split(text)
        return self._to_records(chunks)
    
    # File Reading

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8")
        if suffix == ".pdf":
            return self._read_pdf(path)
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: .txt, .pdf"
        )
    
    @staticmethod
    def _read_pdf(path: Path) -> str:
        try: 
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files. "
                "Install it with: pip install pypdf"
            )
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    
    # splitting strategies

    def _split(self, text: str) -> List[str]:
        normalised = self._normalise_whitespace(text)
        if self.strategy == ChunkStrategy.SLIDING_WINDOW:
            return self._sliding_window(normalised)
        return self._recursive_split(normalised, self._SEPARATORS)
    
    def _sliding_window(self, text: str) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = ""
        for sep in separators:
            if sep == "" or sep in text:
                separator = sep
                break

        raw_splits = text.split(separator) if separator else list(text)
        return self._merge_splits(raw_splits, separator, separators)
    
    def _merge_splits(
        self, splits: List[str], separator: str, remaining_separators: List[str]
    ) -> List[str]:
        result: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for part in splits:
            part = part.strip()
            if not part:
                continue

            part_len = len(part)

            if part_len > self.chunk_size:
                if current_parts:
                    merged = separator.join(current_parts).strip()
                    if merged:
                        result.append(merged)
                    current_parts, current_len = [], 0
                next_seps = (
                    remaining_separators[remaining_separators.index(separator) + 1:]
                    if separator in remaining_separators
                    else [""]
                )
                result.extend(self._recursive_split(part, next_seps or [""]))
                continue

            join_len = (
                part_len + len(separator) + current_len
                if current_parts else part_len
            )

            if join_len > self.chunk_size and current_parts:
                merged = separator.join(current_parts).strip()
                if merged:
                    result.append(merged)
                current_parts, current_len = [], 0

            current_parts.append(part)
            current_len = (
                sum(len(p) for p in current_parts)
                + len(separator) * max(0, len(current_parts) - 1)
            )

        if current_parts:
            merged = separator.join(current_parts).strip()
            if merged:
                result.append(merged)

        return result
    
    # Helpers

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _to_records(chunks: List[str]) -> List[Dict[str, str]]:
        return [
            {"id": f"chunk_{i:04d}", "text": chunk}
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]