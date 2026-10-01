import asyncio
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
import pymupdf as fitz  # PyMuPDF

from app.config import settings
from app.core.exceptions import ValidationException

logger = logging.getLogger(__name__)


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

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
    }

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        *IMAGE_EXTENSIONS,
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

        if extension in self.IMAGE_EXTENSIONS:
            return await asyncio.to_thread(
                self._extract_image,
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
    # OCR & Tessdata Resolution (PyMuPDF only)
    # -----------------------------------------------------

    def _resolve_tessdata(self) -> str | None:
        """
        Locates the tessdata directory for PyMuPDF OCR.
        Checks settings.TESSDATA_PREFIX, project data/tessdata, and system paths.
        """
        configured = getattr(settings, "TESSDATA_PREFIX", None)
        if configured:
            p = Path(configured)
            if not p.is_absolute():
                p = (Path.cwd() / configured).resolve()
            if p.exists() and (p / f"{getattr(settings, 'OCR_LANGUAGE', 'eng')}.traineddata").exists():
                return str(p)

        candidates = [
            Path.cwd() / "data" / "tessdata",
        ]
        lang_file = f"{getattr(settings, 'OCR_LANGUAGE', 'eng')}.traineddata"
        for candidate in candidates:
            if candidate.exists() and (candidate / lang_file).exists():
                return str(candidate)

        return None

    def _ocr_page(
        self,
        page: fitz.Page,
    ) -> str:
        """
        Perform OCR on a single PyMuPDF page using PyMuPDF's built-in OCR.
        """
        try:
            tessdata_path = self._resolve_tessdata()
            language = getattr(settings, "OCR_LANGUAGE", "eng")
            dpi = getattr(settings, "OCR_DPI", 150)

            tp = page.get_textpage_ocr(
                tessdata=tessdata_path,
                language=language,
                dpi=dpi,
                full=True,
            )
            return page.get_text(textpage=tp).strip()
        except Exception as exc:
            logger.warning(
                f"PyMuPDF OCR failed on page {page.number + 1}: {exc}"
            )
            return ""

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
            has_ocr = False

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                text = page.get_text("text").strip()
                is_ocr = False

                # Fallback to PyMuPDF OCR if no digital text exists on the page
                if not text or len(text) < 20:
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        text = f"{text}\n{ocr_text}".strip() if text else ocr_text
                        is_ocr = True
                        has_ocr = True

                if not text:
                    continue

                sections.append(
                    ExtractedSection(
                        text=text,
                        page=page_number,
                        metadata={
                            "source_type": "pdf",
                            "is_ocr": is_ocr,
                        },
                    )
                )

            metadata = {
                "source_type": "pdf",
                "page_count": len(document),
                "has_ocr": has_ocr,
            }

            return ExtractionResult(
                sections=sections,
                metadata=metadata,
            )

        finally:
            document.close()

    # -----------------------------------------------------
    # Images (PyMuPDF OCR)
    # -----------------------------------------------------

    def _extract_image(
        self,
        path: Path,
    ) -> ExtractionResult:
        """
        Extract text from image files (png, jpg, tiff, bmp, webp) using PyMuPDF OCR.
        """
        doc = fitz.open(path)
        try:
            pdf_bytes = doc.convert_to_pdf()
            pdf_doc = fitz.open("pdf", pdf_bytes)
            try:
                page = pdf_doc[0]
                text = self._ocr_page(page)
                sections = []
                if text:
                    sections.append(
                        ExtractedSection(
                            text=text,
                            page=1,
                            metadata={
                                "source_type": "image",
                                "extension": path.suffix.lower(),
                                "is_ocr": True,
                            },
                        )
                    )

                return ExtractionResult(
                    sections=sections,
                    metadata={
                        "source_type": "image",
                        "page_count": 1,
                        "has_ocr": True,
                        "extension": path.suffix.lower(),
                    },
                )
            finally:
                pdf_doc.close()
        finally:
            doc.close()

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