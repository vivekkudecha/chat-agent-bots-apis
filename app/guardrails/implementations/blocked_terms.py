import re
from typing import Any

from app.guardrails.base import (
    BaseGuardrail,
    GuardrailContext,
    GuardrailResult,
)
from app.guardrails.registry import register_guardrail


@register_guardrail("blocked_terms")
class BlockedTermsGuardrail(BaseGuardrail):
    code = "blocked_terms"

    async def evaluate(
        self,
        *,
        text: str,
        config: dict[str, Any],
        context: GuardrailContext,
    ) -> GuardrailResult:

        if not text:
            return GuardrailResult(
                passed=True,
                guardrail_code=self.code,
            )

        terms = config.get("terms", [])

        if not terms:
            return GuardrailResult(
                passed=True,
                guardrail_code=self.code,
            )

        case_sensitive = bool(
            config.get(
                "case_sensitive",
                False,
            )
        )

        whole_word = bool(
            config.get(
                "whole_word",
                False,
            )
        )

        replacement = str(
            config.get(
                "replacement",
                "[BLOCKED]",
            )
        )

        flags = (
            0
            if case_sensitive
            else re.IGNORECASE
        )

        detected_terms: list[str] = []

        redacted_text = text

        for term in terms:

            if not isinstance(term, str):
                continue

            term = term.strip()

            if not term:
                continue

            escaped = re.escape(term)

            if whole_word:
                pattern = rf"\b{escaped}\b"
            else:
                pattern = escaped

            compiled = re.compile(
                pattern,
                flags,
            )

            if not compiled.search(
                redacted_text
            ):
                continue

            detected_terms.append(term)

            redacted_text = compiled.sub(
                replacement,
                redacted_text,
            )

        if not detected_terms:
            return GuardrailResult(
                passed=True,
                guardrail_code=self.code,
            )

        return GuardrailResult(
            passed=False,
            guardrail_code=self.code,
            message=(
                "Content contains restricted terms."
            ),
            modified_text=redacted_text,
            details={
                "match_count": len(
                    detected_terms
                ),
                # Avoid returning original content.
                "stage": context.stage.value,
            },
        )