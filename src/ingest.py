"""Ingest pipeline: read .md/.txt files, split front matter into metadata, chunk the body.

Per docs/DATA_COLLECTION.md the YAML front matter belongs in Document.metadata —
only the text below the closing fence is the content that gets embedded.
"""

from __future__ import annotations

from pathlib import Path

from .chunking import RecursiveChunker
from .models import Document

ALLOWED_EXTENSIONS = {".md", ".txt"}
FENCE = "---"
DEFAULT_CHUNK_SIZE = 500


def _clean_value(raw: str) -> str:
    """Unwrap a quoted YAML scalar, or drop a trailing inline comment."""
    raw = raw.strip()
    if not raw:
        return ""

    if raw[0] in {'"', "'"}:
        closing = raw.find(raw[0], 1)
        if closing != -1:
            return raw[1:closing]

    # Unquoted value: `audience: buyer   # buyer | seller | both`
    for marker in (" #", "\t#"):
        if marker in raw:
            raw = raw.split(marker, 1)[0]
    return raw.strip()


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return (metadata, body). Files without front matter yield ({}, text)."""
    stripped = text.lstrip("﻿").lstrip()
    if not stripped.startswith(FENCE):
        return {}, text.strip()

    lines = stripped.splitlines()
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == FENCE), None)
    if closing is None:
        # An opening fence with no closing one is not front matter.
        return {}, text.strip()

    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        if key:
            metadata[key] = _clean_value(raw)

    return metadata, "\n".join(lines[closing + 1 :]).strip()


def chunk_document(doc: Document, chunker=None) -> list[Document]:
    """Split one Document into chunk-level Documents.

    Every chunk keeps the parent id, so EmbeddingStore stores them under
    unique ids while delete_document(doc_id) still removes the whole document.
    """
    chunker = chunker or RecursiveChunker(chunk_size=DEFAULT_CHUNK_SIZE)
    pieces = chunker.chunk(doc.content)

    chunks: list[Document] = []
    for index, piece in enumerate(pieces):
        metadata = dict(doc.metadata)
        metadata["chunk_index"] = index
        metadata["chunk_total"] = len(pieces)
        chunks.append(Document(id=doc.id, content=piece, metadata=metadata))
    return chunks


def _iter_files(sources: list[str | Path]) -> list[Path]:
    """Expand a mix of file and directory paths into a sorted list of files."""
    files: list[Path] = []
    for raw in sources:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            files.append(path)
    return files


def load_documents(
    sources: list[str | Path],
    chunker=None,
    chunk: bool = True,
    verbose: bool = False,
) -> list[Document]:
    """Load .md/.txt files into (optionally chunked) Documents."""
    documents: list[Document] = []

    for path in _iter_files(sources):
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            if verbose:
                print(f"Skipping unsupported file type: {path}")
            continue
        if not path.exists():
            if verbose:
                print(f"Skipping missing file: {path}")
            continue

        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if not body:
            if verbose:
                print(f"Skipping empty body: {path}")
            continue

        metadata["source"] = str(path)
        metadata["extension"] = path.suffix.lower()
        # A doc_id from the front matter wins; the filename is the fallback.
        doc = Document(id=metadata.get("doc_id") or path.stem, content=body, metadata=metadata)

        documents.extend(chunk_document(doc, chunker) if chunk else [doc])

    return documents
