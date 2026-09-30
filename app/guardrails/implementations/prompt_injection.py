import re
from typing import Any

from app.guardrails.base import (
    BaseGuardrail,
    GuardrailContext,
    GuardrailResult,
)

from app.guardrails.registry import (
    register_guardrail,
)


@register_guardrail("prompt_injection")
class PromptInjectionGuardrail(
    BaseGuardrail
):
    code = "prompt_injection"

    DEFAULT_PATTERNS = [
        r"\bignore\s+(all\s+)?previous\s+instructions?\b",
        r"\bignore\s+(all\s+)?prior\s+instructions?\b",

        r"\bdisregard\s+(all\s+)?previous\s+instructions?\b",
        r"\bforget\s+(all\s+)?previous\s+instructions?\b",

        r"\boverride\s+(the\s+)?system\s+prompt\b",
        r"\boverride\s+(the\s+)?system\s+instructions?\b",

        r"\breveal\s+(the\s+)?system\s+prompt\b",
        r"\bshow\s+(me\s+)?(the\s+)?system\s+prompt\b",
        r"\bprint\s+(the\s+)?system\s+prompt\b",

        r"\breveal\s+(your\s+)?hidden\s+instructions?\b",
        r"\bshow\s+(your\s+)?hidden\s+instructions?\b",

        r"\bdeveloper\s+message\b",
        r"\binternal\s+instructions?\b",

        r"\bdisable\s+(all\s+)?guardrails?\b",
        r"\bbypass\s+(the\s+)?guardrails?\b",

        r"\bignore\s+(the\s+)?safety\s+rules?\b",
        r"\bbypass\s+(the\s+)?safety\s+rules?\b",

        r"\bact\s+as\s+if\s+there\s+are\s+no\s+restrictions\b",

        r"\byou\s+are\s+now\s+in\s+developer\s+mode\b",

        r"\bjailbreak\b",
    ]

    def __init__(self):

        self._compiled_patterns = [
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern
            in self.DEFAULT_PATTERNS
        ]

    async def evaluate(
        self,
        *,
        text: str,
        config: dict[str, Any],
        context: GuardrailContext,
    ) -> GuardrailResult:

        if not text.strip():

            return GuardrailResult(
                passed=True,
                guardrail_code=self.code,
            )

        threshold = int(
            config.get(
                "threshold",
                1,
            )
        )

        threshold = max(
            threshold,
            1,
        )

        matched_patterns = []

        # ---------------------------------------------
        # Built-in patterns
        # ---------------------------------------------

        for pattern in self._compiled_patterns:

            if pattern.search(text):

                matched_patterns.append(
                    pattern.pattern
                )

        # ---------------------------------------------
        # Optional custom phrases
        # ---------------------------------------------

        custom_phrases = config.get(
            "phrases",
            [],
        )

        lowered_text = text.lower()

        for phrase in custom_phrases:

            if not isinstance(
                phrase,
                str,
            ):
                continue

            phrase = phrase.strip()

            if not phrase:
                continue

            if phrase.lower() in lowered_text:

                matched_patterns.append(
                    f"phrase:{phrase}"
                )

        detected = (
            len(matched_patterns)
            >= threshold
        )

        if not detected:

            return GuardrailResult(
                passed=True,
                guardrail_code=self.code,
            )

        return GuardrailResult(
            passed=False,
            guardrail_code=self.code,
            message=(
                "Potential prompt injection "
                "detected."
            ),
            details={
                "match_count": len(
                    matched_patterns
                ),

                # Don't return user text here.
                "signals": (
                    matched_patterns[:10]
                ),

                "stage": (
                    context.stage.value
                ),
            },
        )