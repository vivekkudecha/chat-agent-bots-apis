import asyncio
import csv
import json
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from docx.table import Table as DocxTable
import pymupdf as fitz  # PyMuPDF

from app.config import settings
from app.core.exceptions import ValidationException

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSection:
    """
    A run of text sharing one page and one heading path.

    Paragraphs inside ``text`` are separated by blank lines; the heading
    path (e.g. "Chapter 7 > 7.2 Ellipse") is in ``metadata["section"]``.
    """

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


@dataclass
class ExtractionStats:
    source_type: str = ""
    page_count: int | None = None
    pages_with_text: int = 0
    ocr_pages: int = 0
    section_count: int = 0

    def as_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source_type": self.source_type,
            "has_ocr": self.ocr_pages > 0,
            "section_count": self.section_count,
        }
        if self.page_count is not None:
            metadata["page_count"] = self.page_count
            metadata["pages_with_text"] = self.pages_with_text
            metadata["ocr_pages"] = self.ocr_pages
        return metadata


# Private-use glyphs (math/symbol fonts) and control characters are noise
# for both embeddings and the LLM.
_NOISE_CHARS = re.compile(
    r"[-\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]"
)
_SPACES = re.compile(r"[ \t ]+")
_PAGE_NUMBER = re.compile(
    r"^\W*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\W*$",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"\d+")
_TOC_ENTRY = re.compile(r"(\.{3,}|…)\s*\d+$|^\d+(\.\d+)*\.?\s+\S.*\s\d{1,3}(\s*[-–]\s*\d{1,3})?$")
_TOC_TITLES = {"contents", "table of contents", "index", "table of content"}
_NUMBERED_HEADING = re.compile(
    r"^(\d+(\.\d+)*\.?|[A-Z]\.|[IVXLC]+\.|(chapter|section|part|article|annex(ure)?|appendix)\b)\s*",
    re.IGNORECASE,
)


# Calibri-style "ti" ligature that many PDFs map to U+019F ("enƟtlement").
_TI_LIGATURE = re.compile(r"(?<=[a-z])Ɵ|Ɵ(?=[a-z])")


def clean_text(text: str) -> str:
    # NFKC folds ligatures (ﬁ, ﬂ) and full-width forms so they match queries.
    text = unicodedata.normalize("NFKC", text or "")
    text = _TI_LIGATURE.sub("ti", text)
    text = _NOISE_CHARS.sub("", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def _has_content(text: str, minimum: int = 2) -> bool:
    return sum(c.isalnum() for c in text) >= minimum


@dataclass
class _Block:
    text: str
    size: float
    bold: bool
    top: float
    bottom: float


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

    # Pages with less digital text than this are OCRed.
    OCR_MIN_CHARS = 20

    # Header/footer zone as a fraction of page height.
    MARGIN_ZONE = 0.08

    # -----------------------------------------------------
    # Main Entry
    # -----------------------------------------------------

    async def extract(
        self,
        file_path: str | Path,
    ) -> ExtractionResult:
        """Extract the whole document into memory."""

        stats = ExtractionStats()
        sections: list[ExtractedSection] = []

        async for batch in self.iter_batches(
            file_path,
            stats=stats,
        ):
            sections.extend(batch)

        return ExtractionResult(
            sections=sections,
            metadata=stats.as_metadata(),
        )

    async def iter_batches(
        self,
        file_path: str | Path,
        *,
        pages_per_batch: int | None = None,
        stats: ExtractionStats | None = None,
    ) -> AsyncIterator[list[ExtractedSection]]:
        """
        Stream sections in bounded batches so that very large documents
        (thousands of pages) never need to be held in memory at once.
        Blocking parsing runs in a worker thread.
        """

        path = Path(file_path)
        stats = stats if stats is not None else ExtractionStats()

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
            async for batch in self._iter_pdf(
                path,
                pages_per_batch=(
                    pages_per_batch
                    or settings.RAG_INGEST_PAGE_BATCH
                ),
                stats=stats,
            ):
                yield batch
            return

        if extension in self.IMAGE_EXTENSIONS:
            parser = self._extract_image
        elif extension == ".docx":
            parser = self._extract_docx
        elif extension == ".json":
            parser = self._extract_json
        elif extension == ".csv":
            parser = self._extract_csv
        elif extension in (".md", ".markdown"):
            parser = self._extract_markdown
        else:
            parser = self._extract_text

        sections = await asyncio.to_thread(
            parser,
            path,
            stats,
        )
        stats.section_count += len(sections)

        if sections:
            yield sections

    # -----------------------------------------------------
    # OCR & Tessdata Resolution (PyMuPDF only)
    # -----------------------------------------------------

    def _resolve_tessdata(self) -> str | None:
        """
        Locates the tessdata directory for PyMuPDF OCR.
        Checks settings.TESSDATA_PREFIX, then project data/tessdata.
        """
        languages = [
            lang
            for lang in (settings.OCR_LANGUAGE or "eng").split("+")
            if lang
        ]

        def has_languages(directory: Path) -> bool:
            return directory.exists() and all(
                (directory / f"{lang}.traineddata").exists()
                for lang in languages
            )

        candidates = []
        if settings.TESSDATA_PREFIX:
            configured = Path(settings.TESSDATA_PREFIX)
            if not configured.is_absolute():
                configured = (Path.cwd() / configured).resolve()
            candidates.append(configured)
        candidates.append(Path.cwd() / "data" / "tessdata")

        for candidate in candidates:
            if has_languages(candidate):
                return str(candidate)

        return None

    def _ocr_textpage(
        self,
        page: fitz.Page,
    ) -> Any | None:
        try:
            return page.get_textpage_ocr(
                tessdata=self._resolve_tessdata(),
                language=settings.OCR_LANGUAGE,
                dpi=settings.OCR_DPI,
                full=True,
            )
        except Exception as exc:
            logger.warning(
                "PyMuPDF OCR failed on page %s: %s",
                page.number + 1,
                exc,
            )
            return None

    def _ocr_page(
        self,
        page: fitz.Page,
    ) -> str:
        """
        Perform OCR on a single PyMuPDF page using PyMuPDF's built-in OCR.
        """
        textpage = self._ocr_textpage(page)
        if textpage is None:
            return ""
        return page.get_text(textpage=textpage).strip()

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    async def _iter_pdf(
        self,
        path: Path,
        *,
        pages_per_batch: int,
        stats: ExtractionStats,
    ) -> AsyncIterator[list[ExtractedSection]]:

        stats.source_type = "pdf"

        try:
            document = await asyncio.to_thread(fitz.open, path)
        except Exception as exc:
            raise ValidationException(
                f"Unable to open PDF: {exc}"
            ) from exc

        try:
            if document.needs_pass:
                raise ValidationException(
                    "Password-protected PDFs are not supported."
                )

            stats.page_count = len(document)

            # Shared across batches so heading levels and body font size
            # stay consistent for the whole document.
            state = {
                "font_sizes": Counter(),
                "headings": [],
            }

            for start in range(0, len(document), pages_per_batch):
                end = min(len(document), start + pages_per_batch)

                batch = await asyncio.to_thread(
                    self._extract_pdf_pages,
                    document,
                    start,
                    end,
                    state,
                    stats,
                )
                stats.section_count += len(batch)

                if batch:
                    yield batch
        finally:
            document.close()

    def _extract_pdf_pages(
        self,
        document: fitz.Document,
        start: int,
        end: int,
        state: dict[str, Any],
        stats: ExtractionStats,
    ) -> list[ExtractedSection]:

        pages: list[tuple[int, list[_Block], float, bool]] = []

        for index in range(start, end):
            page = document[index]
            blocks = self._page_blocks(page)
            is_ocr = False

            if sum(len(b.text) for b in blocks) < self.OCR_MIN_CHARS:
                textpage = self._ocr_textpage(page)
                if textpage is not None:
                    ocr_blocks = self._page_blocks(page, textpage=textpage)
                    if ocr_blocks:
                        blocks = ocr_blocks
                        is_ocr = True
                        stats.ocr_pages += 1

            # OCR font sizes are estimates; keep them out of heading detection.
            for block in blocks if not is_ocr else ():
                state["font_sizes"][round(block.size, 1)] += len(block.text)

            pages.append((index + 1, blocks, page.rect.height or 1.0, is_ocr))

        repeated = self._repeated_margin_lines(pages)
        body_size = self._body_font_size(state["font_sizes"])

        sections: list[ExtractedSection] = []

        for page_number, blocks, height, is_ocr in pages:
            paragraphs: list[str] = []
            page_has_text = False

            def flush() -> None:
                if paragraphs:
                    sections.append(
                        ExtractedSection(
                            text="\n\n".join(paragraphs),
                            page=page_number,
                            metadata={
                                "source_type": "pdf",
                                "is_ocr": is_ocr,
                                "section": self._heading_path(state["headings"]),
                            },
                        )
                    )
                    paragraphs.clear()

            for block in blocks:
                in_margin = (
                    block.bottom < height * self.MARGIN_ZONE
                    or block.top > height * (1 - self.MARGIN_ZONE)
                )
                if in_margin and (
                    self._margin_key(block.text) in repeated
                    or _PAGE_NUMBER.match(block.text)
                ):
                    continue

                if not is_ocr and self._is_heading(block, body_size):
                    flush()
                    self._push_heading(state["headings"], block)

                paragraphs.append(block.text)
                page_has_text = True

            flush()

            if page_has_text:
                stats.pages_with_text += 1

        return sections

    def _page_blocks(
        self,
        page: fitz.Page,
        textpage: Any | None = None,
    ) -> list[_Block]:

        data = page.get_text(
            "dict",
            textpage=textpage,
            sort=True,
            flags=(fitz.TEXTFLAGS_DICT | fitz.TEXT_DEHYPHENATE)
            & ~fitz.TEXT_PRESERVE_IMAGES,
        )

        blocks: list[_Block] = []

        for raw in data.get("blocks", []):
            if raw.get("type") != 0:
                continue

            lines: list[str] = []
            size_weight: Counter = Counter()
            bold_chars = 0
            total_chars = 0

            for line in raw.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans)
                if line_text.strip():
                    lines.append(line_text.strip())
                for span in spans:
                    n = len(span.get("text", "").strip())
                    total_chars += n
                    size_weight[round(span.get("size", 0.0), 1)] += n
                    if span.get("flags", 0) & 16:
                        bold_chars += n

            text = clean_text(" ".join(lines))
            if not _has_content(text):
                continue

            bbox = raw.get("bbox", (0, 0, 0, 0))
            blocks.append(
                _Block(
                    text=text,
                    size=size_weight.most_common(1)[0][0] if size_weight else 0.0,
                    bold=total_chars > 0 and bold_chars / total_chars > 0.6,
                    top=bbox[1],
                    bottom=bbox[3],
                )
            )

        return blocks

    @staticmethod
    def _margin_key(text: str) -> str:
        return _DIGITS.sub("#", text.lower()).strip()

    def _repeated_margin_lines(
        self,
        pages: list[tuple[int, list[_Block], float, bool]],
    ) -> set[str]:
        """Running headers/footers: margin text repeated on many pages."""

        if len(pages) < 3:
            return set()

        counts: Counter = Counter()

        for _, blocks, height, _ in pages:
            keys = {
                self._margin_key(block.text)
                for block in blocks
                if block.bottom < height * self.MARGIN_ZONE
                or block.top > height * (1 - self.MARGIN_ZONE)
            }
            counts.update(keys)

        threshold = max(3, int(len(pages) * 0.4))
        return {key for key, count in counts.items() if count >= threshold}

    @staticmethod
    def _body_font_size(font_sizes: Counter) -> float:
        if not font_sizes:
            return 0.0
        return font_sizes.most_common(1)[0][0]

    @staticmethod
    def _is_heading(block: _Block, body_size: float) -> bool:
        text = block.text
        if body_size <= 0 or len(text) > 120 or not any(c.isalpha() for c in text):
            return False
        # Table-of-contents titles and entries ("4. Eligibility 2-3").
        if _TOC_ENTRY.search(text) or text.lower().strip(" :") in _TOC_TITLES:
            return False
        if block.size >= body_size * 1.18:
            return True
        # Bold body-size text is a heading only when it looks like one;
        # bold table headers and emphasis are not.
        return (
            block.bold
            and block.size >= body_size * 0.98
            and len(text) <= 80
            and not text.endswith((".", ",", ";", ":"))
            and (_NUMBERED_HEADING.match(text) is not None or text.isupper())
        )

    @staticmethod
    def _push_heading(stack: list[tuple[float, str]], block: _Block) -> None:
        # Pop headings that are the same size or smaller (siblings/children).
        while stack and stack[-1][0] <= block.size + 0.1:
            stack.pop()
        stack.append((block.size, block.text[:100]))
        del stack[:-3]

    @staticmethod
    def _heading_path(stack: list[tuple[float, str]]) -> str | None:
        if not stack:
            return None
        return " > ".join(text for _, text in stack)

    # -----------------------------------------------------
    # Images (PyMuPDF OCR)
    # -----------------------------------------------------

    def _extract_image(
        self,
        path: Path,
        stats: ExtractionStats,
    ) -> list[ExtractedSection]:
        """
        Extract text from image files (png, jpg, tiff, bmp, webp) using PyMuPDF OCR.
        """
        stats.source_type = "image"
        stats.page_count = 1

        doc = fitz.open(path)
        try:
            pdf_doc = fitz.open("pdf", doc.convert_to_pdf())
            try:
                text = clean_text(self._ocr_page(pdf_doc[0]))
            finally:
                pdf_doc.close()
        finally:
            doc.close()

        if not text:
            return []

        stats.ocr_pages = 1
        stats.pages_with_text = 1

        return [
            ExtractedSection(
                text=text,
                page=1,
                metadata={
                    "source_type": "image",
                    "extension": path.suffix.lower(),
                    "is_ocr": True,
                },
            )
        ]

    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    TABLE_ROWS_PER_BLOCK = 15

    def _extract_docx(
        self,
        path: Path,
        stats: ExtractionStats,
    ) -> list[ExtractedSection]:

        stats.source_type = "docx"
        document = DocxDocument(path)

        sections: list[ExtractedSection] = []
        headings: list[tuple[int, str]] = []
        paragraphs: list[str] = []

        def flush() -> None:
            if paragraphs:
                sections.append(
                    ExtractedSection(
                        text="\n\n".join(paragraphs),
                        metadata={
                            "source_type": "docx",
                            "section": " > ".join(h for _, h in headings) or None,
                        },
                    )
                )
                paragraphs.clear()

        # Body order matters: paragraphs and tables are interleaved.
        for item in document.iter_inner_content():

            if isinstance(item, DocxTable):
                paragraphs.extend(self._docx_table_blocks(item))
                continue

            text = clean_text(item.text)
            if not text:
                continue

            level = self._docx_heading_level(item)
            if level is not None:
                flush()
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, text[:100]))
                del headings[:-3]

            paragraphs.append(text)

        flush()
        return sections

    @staticmethod
    def _docx_heading_level(paragraph: Any) -> int | None:
        style = (getattr(paragraph.style, "name", "") or "").lower()
        if style == "title":
            return 0
        if style.startswith("heading"):
            digits = "".join(c for c in style if c.isdigit())
            return int(digits) if digits else 1
        return None

    def _docx_table_blocks(self, table: DocxTable) -> list[str]:
        rows: list[list[str]] = []

        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                value = clean_text(cell.text.replace("\n", " "))
                # Merged cells repeat; keep one copy.
                if not cells or cells[-1] != value:
                    cells.append(value)
            if any(cells):
                rows.append(cells)

        if not rows:
            return []

        header, body = rows[0], rows[1:]
        if not body:
            return [" | ".join(header)]

        # Each block repeats the header so every chunk is self-describing.
        blocks = []
        for i in range(0, len(body), self.TABLE_ROWS_PER_BLOCK):
            lines = [" | ".join(header)]
            lines.extend(" | ".join(r) for r in body[i:i + self.TABLE_ROWS_PER_BLOCK])
            blocks.append("\n".join(lines))
        return blocks

    # -----------------------------------------------------
    # TXT / Markdown
    # -----------------------------------------------------

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")

    def _extract_text(
        self,
        path: Path,
        stats: ExtractionStats,
    ) -> list[ExtractedSection]:

        source_type = path.suffix.lower().lstrip(".") or "txt"
        stats.source_type = source_type

        # Form feeds mark page breaks in text exports.
        pages = self._read_text(path).split("\f")
        sections = []

        for number, page_text in enumerate(pages, start=1):
            text = self._normalize_paragraphs(page_text)
            if text:
                sections.append(
                    ExtractedSection(
                        text=text,
                        page=number if len(pages) > 1 else None,
                        metadata={"source_type": source_type},
                    )
                )

        if len(pages) > 1:
            stats.page_count = len(pages)
            stats.pages_with_text = len(sections)

        return sections

    _MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def _extract_markdown(
        self,
        path: Path,
        stats: ExtractionStats,
    ) -> list[ExtractedSection]:

        stats.source_type = "md"

        sections: list[ExtractedSection] = []
        headings: list[tuple[int, str]] = []
        lines: list[str] = []
        in_code = False

        def flush() -> None:
            text = self._normalize_paragraphs("\n".join(lines))
            if text:
                sections.append(
                    ExtractedSection(
                        text=text,
                        metadata={
                            "source_type": "md",
                            "section": " > ".join(h for _, h in headings) or None,
                        },
                    )
                )
            lines.clear()

        for line in self._read_text(path).splitlines():
            if line.lstrip().startswith("```"):
                in_code = not in_code

            match = None if in_code else self._MD_HEADING.match(line)
            if match:
                flush()
                level = len(match.group(1))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, clean_text(match.group(2))[:100]))
                del headings[:-3]

            lines.append(line)

        flush()
        return sections

    @staticmethod
    def _normalize_paragraphs(text: str) -> str:
        paragraphs = [
            "\n".join(
                clean_text(line)
                for line in block.splitlines()
                if clean_text(line)
            )
            for block in re.split(r"\n\s*\n", text)
        ]
        return "\n\n".join(p for p in paragraphs if p)

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    def _extract_csv(
        self,
        path: Path,
        stats: ExtractionStats,
    ) -> list[ExtractedSection]:

        stats.source_type = "csv"

        with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            sample = file.read(8192)
            file.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel
            rows = list(csv.reader(file, dialect))

        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            return []

        header = [clean_text(h) or f"column_{i + 1}" for i, h in enumerate(rows[0])]

        # One "column: value" line per row keeps every chunk self-describing.
        records = []
        for row in rows[1:]:
            pairs = [
                f"{header[i] if i < len(header) else f'column_{i + 1}'}: {clean_text(value)}"
                for i, value in enumerate(row)
                if value.strip()
            ]
            if pairs:
                records.append("; ".join(pairs))

        if not records:
            records = [" | ".join(header)]

        return [
            ExtractedSection(
                text="\n\n".join(records),
                metadata={"source_type": "csv", "row_count": len(records)},
            )
        ]

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    def _extract_json(
        self,
        path: Path,
        stats: ExtractionStats,
    ) -> list[ExtractedSection]:

        stats.source_type = "json"

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list) and data and all(isinstance(i, dict) for i in data):
            # Record arrays: one paragraph per record.
            content = "\n\n".join(
                json.dumps(item, ensure_ascii=False)
                for item in data
            )
        else:
            content = json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )

        return [
            ExtractedSection(
                text=content,
                metadata={
                    "source_type": "json",
                },
            )
        ]
