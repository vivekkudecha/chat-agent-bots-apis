from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import asyncio
import json

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from app.core.exceptions import ValidationException


@dataclass
class ExtractedSection:
    text: str
    page: int | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ExtractionResult:
    sections: list[ExtractedSection]
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class TextExtractionService:

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
    }

    # -----------------------------------------------------
    # Main Entry
    # -----------------------------------------------------

    async def extract(
        self,
        file_path: str | Path,
    ) -> ExtractionResult:

        path = Path(file_path)

        if not path.exists():
            raise ValidationException(
                "Document file does not exist."
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValidationException(
                f"Unsupported file type: {extension}"
            )

        if extension == ".pdf":
            return await asyncio.to_thread(
                self._extract_pdf,
                path,
            )

        if extension == ".docx":
            return await asyncio.to_thread(
                self._extract_docx,
                path,
            )

        if extension == ".json":
            return await asyncio.to_thread(
                self._extract_json,
                path,
            )

        return await asyncio.to_thread(
            self._extract_text,
            path,
        )

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    def _extract_pdf(
        self,
        path: Path,
    ) -> ExtractionResult:

        sections: list[ExtractedSection] = []

        document = fitz.open(path)

        try:

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                text = page.get_text(
                    "text"
                ).strip()

                if not text:
                    continue

                sections.append(
                    ExtractedSection(
                        text=text,
                        page=page_number,
                        metadata={
                            "source_type": "pdf",
                        },
                    )
                )

            metadata = {
                "source_type": "pdf",
                "page_count": len(document),
            }

            return ExtractionResult(
                sections=sections,
                metadata=metadata,
            )

        finally:
            document.close()

    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    def _extract_docx(
        self,
        path: Path,
    ) -> ExtractionResult:

        document = DocxDocument(path)

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        content = "\n\n".join(paragraphs)

        sections = []

        if content:
            sections.append(
                ExtractedSection(
                    text=content,
                    metadata={
                        "source_type": "docx",
                    },
                )
            )

        return ExtractionResult(
            sections=sections,
            metadata={
                "source_type": "docx",
                "paragraph_count": len(
                    paragraphs
                ),
            },
        )

    # -----------------------------------------------------
    # TXT / Markdown / CSV
    # -----------------------------------------------------

    def _extract_text(
        self,
        path: Path,
    ) -> ExtractionResult:

        try:
            content = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        content = content.strip()

        sections = []

        if content:
            sections.append(
                ExtractedSection(
                    text=content,
                    metadata={
                        "source_type": (
                            path.suffix
                            .lower()
                            .lstrip(".")
                        )
                    },
                )
            )

        return ExtractionResult(
            sections=sections,
            metadata={
                "source_type": (
                    path.suffix
                    .lower()
                    .lstrip(".")
                ),
            },
        )

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    def _extract_json(
        self,
        path: Path,
    ) -> ExtractionResult:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        content = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )

        return ExtractionResult(
            sections=[
                ExtractedSection(
                    text=content,
                    metadata={
                        "source_type": "json",
                    },
                )
            ],
            metadata={
                "source_type": "json",
            },
        )