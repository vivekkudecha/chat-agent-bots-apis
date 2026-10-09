import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.ai.rag.text_extraction import (
    ExtractedSection,
    ExtractionResult,
)


@dataclass
class TextChunk:
    index: int
    text: str

    page: int | None = None
    char_count: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class _Piece:
    text: str
    page: int | None
    section: str | None
    metadata: dict[str, Any]


# Sentence ends across scripts; CJK full stops need no trailing space.
_SENTENCE_END = re.compile(r"(?<=[.!?;।؟۔])\s+|(?<=[。！？])")


class ChunkingService:
    """
    Structure-aware chunker.

    - Packs whole paragraphs up to ``chunk_size`` characters; only
      oversized paragraphs are split (sentence, then word boundaries).
    - Chunks may span page breaks within one section, and break when the
      heading path changes, so a chunk never mixes two topics.
    - Overlap carries whole trailing sentences (never half words) into
      the next chunk of the same section.
    - Streaming: ``feed()`` batches of sections and ``flush()`` at the
      end; chunk indexes are contiguous across batches.

    Chunk metadata carries ``page_start``/``page_end``, ``section`` and
    ``overlap_chars`` (length of the carried prefix) so retrieval can
    merge neighbouring chunks without duplicating text.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        "। ",   # Devanagari / Hindi / Bengali full stop
        "。 ",   # CJK (Chinese, Japanese) full stop with space
        "。",    # CJK full stop without space
        "؟ ",   # Arabic / Persian / Urdu question mark
        "۔ ",   # Arabic / Urdu full stop
        ". ",   # Latin / Western period
        "! ",
        "? ",
        "; ",
        " ",
        "",
    ]

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        min_chunk_size: int | None = None,
    ):

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Below this size a chunk absorbs the next section instead of
        # being emitted on its own (e.g. a lone heading line).
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else max(1, chunk_size // 3)
        )

        self.separators = (
            separators
            or self.DEFAULT_SEPARATORS
        )

        self._pending: list[_Piece] = []
        self._pending_chars = 0
        self._carry = ""
        self._next_index = 0

    # -----------------------------------------------------
    # Extraction Result → Chunks
    # -----------------------------------------------------

    def chunk_extraction(
        self,
        extraction: ExtractionResult,
    ) -> list[TextChunk]:

        chunks = self.feed(extraction.sections)
        chunks.extend(self.flush())
        return chunks

    # -----------------------------------------------------
    # Streaming API
    # -----------------------------------------------------

    def feed(
        self,
        sections: Iterable[ExtractedSection],
    ) -> list[TextChunk]:

        chunks: list[TextChunk] = []

        for section in sections:
            section_path = section.metadata.get("section")

            for paragraph in section.text.split("\n\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                for text in self._fit(paragraph):
                    piece = _Piece(
                        text=text,
                        page=section.page,
                        section=section_path,
                        metadata=section.metadata,
                    )
                    chunks.extend(self._add(piece))

        return chunks

    def flush(self) -> list[TextChunk]:
        chunks = []
        if self._pending:
            chunks.append(self._emit(carry_overlap=False))
        self._carry = ""
        return chunks

    @staticmethod
    def embedding_text(
        text: str,
        *,
        file_name: str | None = None,
        section: str | None = None,
    ) -> str:
        """
        Text that is embedded/indexed for a chunk: a short contextual
        header (document + heading path) followed by the chunk body. The
        header resolves pronouns and generic passages ("Eligibility: 3
        years") to the document and topic they belong to.
        """
        header = []
        if file_name:
            header.append(f"Document: {file_name}")
        if section:
            header.append(f"Section: {section}")
        if not header:
            return text
        return "\n".join(header) + "\n\n" + text

    # -----------------------------------------------------
    # Packing
    # -----------------------------------------------------

    def _add(self, piece: _Piece) -> list[TextChunk]:
        emitted: list[TextChunk] = []

        if self._pending:
            section_changed = piece.section != self._pending[-1].section
            too_big = (
                self._pending_chars + len(piece.text) + 2
                > self.chunk_size
            )

            if section_changed and self._pending_chars >= self.min_chunk_size:
                emitted.append(self._emit(carry_overlap=False))
            elif too_big:
                emitted.append(self._emit(carry_overlap=not section_changed))

        self._pending.append(piece)
        self._pending_chars += len(piece.text) + 2
        return emitted

    def _emit(self, *, carry_overlap: bool) -> TextChunk:
        pieces = self._pending
        body = "\n\n".join(p.text for p in pieces)
        prefix = self._carry

        text = f"{prefix}\n{body}" if prefix else body
        overlap_chars = len(prefix) + 1 if prefix else 0

        pages = [p.page for p in pieces if p.page is not None]
        first = pieces[0]
        sections = [p.section for p in pieces if p.section]

        metadata: dict[str, Any] = {
            **first.metadata,
            "char_count": len(text),
            "overlap_chars": overlap_chars,
            "section": sections[-1] if sections else None,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
        }
        if any(p.metadata.get("is_ocr") for p in pieces):
            metadata["is_ocr"] = True

        chunk = TextChunk(
            index=self._next_index,
            text=text,
            page=metadata["page_start"],
            char_count=len(text),
            metadata=metadata,
        )

        self._next_index += 1
        self._carry = self._tail(body) if carry_overlap else ""
        self._pending = []
        self._pending_chars = len(self._carry)

        return chunk

    def _tail(self, text: str) -> str:
        """Trailing whole sentences that fit within ``chunk_overlap``."""

        if self.chunk_overlap == 0:
            return ""

        sentences = [
            s.strip()
            for s in _SENTENCE_END.split(text.replace("\n", " "))
            if s.strip()
        ]

        tail: list[str] = []
        size = 0
        for sentence in reversed(sentences):
            if size + len(sentence) + 1 > self.chunk_overlap:
                break
            tail.insert(0, sentence)
            size += len(sentence) + 1

        if tail:
            return " ".join(tail)

        # No sentence fits: fall back to trailing words.
        words = text[-self.chunk_overlap:].split(" ", 1)
        return words[-1].strip() if len(words) > 1 else ""

    def _fit(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        return [
            part.strip()
            for part in self._split_recursive(
                paragraph,
                self.separators[1:],
            )
            if part.strip()
        ]

    # -----------------------------------------------------
    # Recursive Split
    # -----------------------------------------------------

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        separator = separators[0]

        if separator == "":
            return self._hard_split(text)

        if separator not in text:
            return self._split_recursive(
                text,
                separators[1:],
            )

        parts = text.split(separator)
        # Keep sentence punctuation on the left part ("end." not "end").
        keep = separator.rstrip()

        chunks: list[str] = []
        current = ""

        for i, part in enumerate(parts):

            part = part.strip()

            if not part:
                continue

            if keep and i < len(parts) - 1:
                part += keep

            candidate = (
                part
                if not current
                else current + " " + part
            )

            if len(candidate) <= self.chunk_size:

                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(part) > self.chunk_size:

                chunks.extend(
                    self._split_recursive(
                        part,
                        separators[1:],
                    )
                )

                current = ""

            else:
                current = part

        if current:
            chunks.append(current)

        return chunks

    # -----------------------------------------------------
    # Hard Split
    # -----------------------------------------------------

    def _hard_split(
        self,
        text: str,
    ) -> list[str]:

        return [
            text[
                i:i + self.chunk_size
            ]
            for i in range(
                0,
                len(text),
                self.chunk_size,
            )
        ]
