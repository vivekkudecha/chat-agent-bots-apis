import re
from dataclasses import dataclass
from typing import Any

from app.ai.guardrails.base import (
    BaseGuardrail,
    GuardrailContext,
    GuardrailResult,
)

from app.ai.guardrails.registry import (
    register_guardrail,
)


@dataclass(frozen=True)
class PIIPattern:
    name: str
    pattern: re.Pattern
    replacement: str


@register_guardrail("pii_detection")
class PIIDetectionGuardrail(
    BaseGuardrail
):
    code = "pii_detection"

    PII_PATTERNS = [

        # -----------------------------------------
        # Email
        # -----------------------------------------

        PIIPattern(
            name="email",
            pattern=re.compile(
                r"""
                \b
                [A-Z0-9._%+-]+
                @
                [A-Z0-9.-]+
                \.
                [A-Z]{2,}
                \b
                """,
                re.IGNORECASE | re.VERBOSE,
            ),
            replacement="[EMAIL_REDACTED]",
        ),

        # -----------------------------------------
        # Phone
        #
        # Conservative generic international
        # detector.
        # -----------------------------------------

        PIIPattern(
            name="phone",
            pattern=re.compile(
                r"""
                (?<!\d)

                (?:\+\d{1,3}[\s.-]?)?

                (?:\(?\d{2,4}\)?[\s.-]?)?

                \d{3,4}
                [\s.-]?
                \d{4}

                (?!\d)
                """,
                re.VERBOSE,
            ),
            replacement="[PHONE_REDACTED]",
        ),

        # -----------------------------------------
        # IPv4
        # -----------------------------------------

        PIIPattern(
            name="ipv4",
            pattern=re.compile(
                r"""
                \b

                (?:
                    25[0-5]
                    |
                    2[0-4]\d
                    |
                    1?\d?\d
                )

                \.

                (?:
                    25[0-5]
                    |
                    2[0-4]\d
                    |
                    1?\d?\d
                )

                \.

                (?:
                    25[0-5]
                    |
                    2[0-4]\d
                    |
                    1?\d?\d
                )

                \.

                (?:
                    25[0-5]
                    |
                    2[0-4]\d
                    |
                    1?\d?\d
                )

                \b
                """,
                re.VERBOSE,
            ),
            replacement="[IP_REDACTED]",
        ),
    ]

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

        enabled_types = config.get(
            "types",
            [
                "email",
                "phone",
                "ipv4",
            ],
        )

        enabled_types = {
            str(item).lower()
            for item in enabled_types
        }

        detected_types: set[str] = set()

        redacted_text = text

        # ---------------------------------------------
        # Built-in detectors
        # ---------------------------------------------

        for pii_pattern in (
            self.PII_PATTERNS
        ):

            if (
                pii_pattern.name
                not in enabled_types
            ):
                continue

            if not pii_pattern.pattern.search(
                redacted_text
            ):
                continue

            detected_types.add(
                pii_pattern.name
            )

            redacted_text = (
                pii_pattern
                .pattern
                .sub(
                    pii_pattern.replacement,
                    redacted_text,
                )
            )

        # ---------------------------------------------
        # Custom patterns
        # ---------------------------------------------

        custom_patterns = config.get(
            "custom_patterns",
            [],
        )

        for index, item in enumerate(
            custom_patterns
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            pattern = item.get(
                "pattern"
            )

            if not pattern:
                continue

            name = item.get(
                "name",
                f"custom_{index}",
            )

            replacement = item.get(
                "replacement",
                "[PII_REDACTED]",
            )

            try:

                compiled = re.compile(
                    pattern,
                    re.IGNORECASE,
                )

            except re.error:
                continue

            if not compiled.search(
                redacted_text
            ):
                continue

            detected_types.add(
                str(name)
            )

            redacted_text = (
                compiled.sub(
                    replacement,
                    redacted_text,
                )
            )

        # ---------------------------------------------
        # Nothing detected
        # ---------------------------------------------

        if not detected_types:

            return GuardrailResult(
                passed=True,
                guardrail_code=self.code,
            )

        return GuardrailResult(
            passed=False,
            guardrail_code=self.code,
            message=(
                "Potential personally identifiable "
                "information detected."
            ),
            modified_text=redacted_text,
            details={
                "pii_types": sorted(
                    detected_types
                ),
                "stage": (
                    context.stage.value
                ),
            },
        )