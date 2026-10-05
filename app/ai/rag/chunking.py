from dataclasses import dataclass, field
from typing import Any

from app.ai.rag.text_extraction import (
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


class ChunkingService:

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
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

        self.separators = (
            separators
            or self.DEFAULT_SEPARATORS
        )

    # -----------------------------------------------------
    # Extraction Result → Chunks
    # -----------------------------------------------------

    def chunk_extraction(
        self,
        extraction: ExtractionResult,
    ) -> list[TextChunk]:

        chunks: list[TextChunk] = []

        chunk_index = 0

        for section in extraction.sections:

            section_chunks = (
                self._split_recursive(
                    section.text,
                    self.separators,
                )
            )

            section_chunks = (
                self._apply_overlap(
                    section_chunks
                )
            )

            for text in section_chunks:

                text = text.strip()

                if not text:
                    continue

                chunks.append(
                    TextChunk(
                        index=chunk_index,
                        text=text,
                        page=section.page,
                        char_count=len(text),
                        metadata={
                            "char_count": len(text),
                            **section.metadata,
                        },
                    )
                )

                chunk_index += 1

        return chunks

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

        chunks: list[str] = []
        current = ""

        for part in parts:

            part = part.strip()

            if not part:
                continue

            candidate = (
                part
                if not current
                else current + separator + part
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

    # -----------------------------------------------------
    # Apply Overlap
    # -----------------------------------------------------

    def _apply_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:

        if (
            self.chunk_overlap == 0
            or len(chunks) <= 1
        ):
            return chunks

        result = []

        previous = ""

        for chunk in chunks:

            if previous:

                overlap = previous[
                    -self.chunk_overlap:
                ]

                combined = (
                    overlap
                    + " "
                    + chunk
                )

                # Keep overlap bounded.
                if (
                    len(combined)
                    > self.chunk_size
                    + self.chunk_overlap
                ):
                    combined = combined[
                        :self.chunk_size
                        + self.chunk_overlap
                    ]

                result.append(combined)

            else:
                result.append(chunk)

            previous = chunk

        return result