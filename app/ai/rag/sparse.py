"""BM25 sparse vectors for Qdrant hybrid search.

Documents carry the BM25 term-frequency component; Qdrant applies IDF at
query time (collection sparse vector configured with ``Modifier.IDF``), so
the index never needs global corpus statistics and stays correct as
documents are added or removed.
"""

import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass

from app.config import settings


# Scripts written without spaces: index character bigrams instead of words.
_UNSEGMENTED = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯฀-๿]"
)

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Small default list; the Qdrant IDF modifier already discounts frequent
# terms in any language, this only trims the commonest English noise.
_STOP_WORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its me my no not of on or our so than that the
    their them then there these they this to was we were what when where which
    who why will with would you your
    """.split()
)


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


class BM25SparseEncoder:

    def __init__(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        avg_doc_tokens: int | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.avg_doc_tokens = max(
            1,
            avg_doc_tokens or settings.RAG_BM25_AVG_DOC_TOKENS,
        )

        custom = settings.RAG_STOP_WORDS
        self.stop_words = _STOP_WORDS | {
            w.strip().lower()
            for w in (custom or "").split(",")
            if w.strip()
        }

    # -----------------------------------------------------
    # Tokenisation
    # -----------------------------------------------------

    def tokenize(self, text: str) -> list[str]:

        normalized = unicodedata.normalize(
            "NFKC",
            text or "",
        ).lower()

        tokens: list[str] = []

        for word in _TOKEN.findall(normalized):

            if _UNSEGMENTED.search(word):
                chars = [c for c in word if not c.isspace()]
                if len(chars) == 1:
                    tokens.append(chars[0])
                tokens.extend(
                    chars[i] + chars[i + 1]
                    for i in range(len(chars) - 1)
                )
                continue

            if word in self.stop_words:
                continue

            # Single ASCII letters/digits and underscores carry no signal.
            if len(word) < 2 and word.isascii():
                continue

            tokens.append(word)

        return tokens

    @staticmethod
    def term_id(token: str) -> int:
        return zlib.crc32(token.encode("utf-8"))

    # -----------------------------------------------------
    # Encoding
    # -----------------------------------------------------

    def encode_document(self, text: str) -> SparseVectorData:

        tokens = self.tokenize(text)

        if not tokens:
            return SparseVectorData([], [])

        length_norm = (
            1 - self.b
            + self.b * len(tokens) / self.avg_doc_tokens
        )

        weights: dict[int, float] = {}

        for token, tf in Counter(tokens).items():
            score = (
                tf * (self.k1 + 1)
                / (tf + self.k1 * length_norm)
            )
            term = self.term_id(token)
            # crc32 collisions are rare; keep the stronger weight.
            weights[term] = max(weights.get(term, 0.0), score)

        return SparseVectorData(
            indices=list(weights),
            values=list(weights.values()),
        )

    def encode_query(self, text: str) -> SparseVectorData:

        terms = {
            self.term_id(token)
            for token in self.tokenize(text)
        }

        return SparseVectorData(
            indices=list(terms),
            values=[1.0] * len(terms),
        )

    def query_terms(self, text: str) -> set[str]:
        return set(self.tokenize(text))
