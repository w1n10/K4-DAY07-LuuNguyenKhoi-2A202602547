from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    # A boundary is a sentence-ending mark followed by whitespace: ". ", "! ", "? ", ".\n"
    _BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in self._BOUNDARY.split(text.strip())]
        sentences = [s for s in sentences if s]

        step = self.max_sentences_per_chunk
        return [" ".join(sentences[i : i + step]) for i in range(0, len(sentences), step)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text.strip():
            return []

        # Base case 1: the fragment already fits.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 2: no separator left to try — cut at fixed size.
        if not remaining_separators:
            return self._hard_split(current_text)

        separator, rest = remaining_separators[0], remaining_separators[1:]

        # The empty separator means "split between characters".
        if separator == "":
            return self._hard_split(current_text)

        # This separator does not appear here — try the next one.
        if separator not in current_text:
            return self._split(current_text, rest)

        # Split on the separator, then greedily merge the pieces back together
        # so each chunk stays as close to chunk_size as possible.
        chunks: list[str] = []
        buffer = ""
        for piece in current_text.split(separator):
            candidate = piece if not buffer else buffer + separator + piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = ""

            if len(piece) <= self.chunk_size:
                buffer = piece
            else:
                # A single piece is still too long: recurse with finer separators.
                chunks.extend(self._split(piece, rest))

        if buffer:
            chunks.append(buffer)

        return [c for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> list[str]:
        size = max(1, self.chunk_size)
        return [text[i : i + size] for i in range(0, len(text), size)]


class SemanticChunker:
    """
    Split policy text at clause boundaries instead of at a character count.

    Legal/policy documents are already structured by their authors: "A. PHẠM VI",
    "2. THỜI HẠN BẢO HÀNH:", "1.1. Đối Tượng Áp Dụng". Each such clause is one
    self-contained idea, which is exactly what a retrievable chunk should be.

    Two rules make this different from RecursiveChunker:
        - A heading always stays attached to the body it introduces, so no chunk
          is ever a bare title with no content.
        - When a clause is longer than max_chunk_size it is split further, but the
          heading is repeated on every piece so each one keeps its context.

    Sub-items ("a.", "(b)", "(iii)") are NOT treated as boundaries — they belong
    to the clause above them and splitting there would fragment one rule.
    """

    HEADING = re.compile(
        r"""^(?:
              \#{1,6}\s+\S            # markdown heading:  # Chính sách ...
            | [A-ZĐ]\.\s*\S           # section letter:    E. XỬ LÝ VI PHẠM
            | \d+(?:\.\d+)*\.\s*\S    # clause number:     2. / 1.1. / 3.2.
        )""",
        re.VERBOSE,
    )

    def __init__(self, max_chunk_size: int = 800) -> None:
        self.max_chunk_size = max(1, max_chunk_size)

    def _is_heading(self, line: str) -> bool:
        # Match on a normalized copy: crawled text is littered with \xa0.
        return bool(self.HEADING.match(line.replace("\xa0", " ").strip()))

    def _split_sections(self, text: str) -> list[list[str]]:
        """Group lines into sections, each starting at a heading line."""
        sections: list[list[str]] = []
        current: list[str] = []

        for line in text.splitlines():
            if self._is_heading(line) and current:
                sections.append(current)
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append(current)
        return sections

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        chunks: list[str] = []
        for section in self._split_sections(text):
            section_text = "\n".join(section).strip()
            if not section_text:
                continue

            if len(section_text) <= self.max_chunk_size:
                chunks.append(section_text)
                continue

            # Too long: keep the heading as a prefix on every piece of the body.
            heading = section[0].strip()
            body = "\n".join(section[1:]).strip()
            if not body:
                chunks.append(section_text)
                continue

            budget = max(1, self.max_chunk_size - len(heading) - 1)
            for piece in RecursiveChunker(chunk_size=budget).chunk(body):
                chunks.append(f"{heading}\n{piece.strip()}")

        return [c for c in chunks if c.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size // 10),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(c) for c in chunks]
            comparison[name] = {
                "count": len(chunks),
                "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "chunks": chunks,
            }
        return comparison
