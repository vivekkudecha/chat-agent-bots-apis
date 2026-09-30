import re
from dataclasses import dataclass
from typing import Any

from app.guardrails.base import (
    BaseGuardrail,
    GuardrailContext,
    GuardrailResult,
)

from app.guardrails.registry import (
    register_guardrail,
)


@dataclass(frozen=True)
class SecretPattern:
    name: str
    pattern: re.Pattern


@register_guardrail("secret_detection")
class SecretDetectionGuardrail(
    BaseGuardrail
):
    code = "secret_detection"

    SECRET_PATTERNS = [

        # -----------------------------------------
        # Private keys
        # -----------------------------------------

        SecretPattern(
            name="private_key",
            pattern=re.compile(
                r"-----BEGIN\s+"
                r"(?:RSA |EC |OPENSSH |DSA )?"
                r"PRIVATE KEY-----"
                r"[\s\S]*?"
                r"-----END\s+"
                r"(?:RSA |EC |OPENSSH |DSA )?"
                r"PRIVATE KEY-----",
                re.IGNORECASE,
            ),
        ),

        # -----------------------------------------
        # AWS Access Key
        # -----------------------------------------

        SecretPattern(
            name="aws_access_key",
            pattern=re.compile(
                r"\b"
                r"(?:AKIA|ASIA)"
                r"[A-Z0-9]{16}"
                r"\b"
            ),
        ),

        # -----------------------------------------
        # GitHub tokens
        # -----------------------------------------

        SecretPattern(
            name="github_token",
            pattern=re.compile(
                r"\b"
                r"(?:"
                r"ghp_|"
                r"gho_|"
                r"ghu_|"
                r"ghs_|"
                r"ghr_"
                r")"
                r"[A-Za-z0-9]{20,255}"
                r"\b"
            ),
        ),

        # -----------------------------------------
        # Slack token
        # -----------------------------------------

        SecretPattern(
            name="slack_token",
            pattern=re.compile(
                r"\b"
                r"xox[baprs]-"
                r"[A-Za-z0-9-]{10,}"
                r"\b",
                re.IGNORECASE,
            ),
        ),

        # -----------------------------------------
        # JWT
        # -----------------------------------------

        SecretPattern(
            name="jwt",
            pattern=re.compile(
                r"\b"
                r"eyJ[A-Za-z0-9_-]{5,}"
                r"\."
                r"[A-Za-z0-9_-]{5,}"
                r"\."
                r"[A-Za-z0-9_-]{5,}"
                r"\b"
            ),
        ),

        # -----------------------------------------
        # Bearer token
        # -----------------------------------------

        SecretPattern(
            name="bearer_token",
            pattern=re.compile(
                r"(?i)"
                r"\bBearer\s+"
                r"[A-Za-z0-9._~+/=-]{20,}"
            ),
        ),

        # -----------------------------------------
        # Generic assignments
        #
        # API_KEY=...
        # secret: ...
        # access_token=...
        # -----------------------------------------

        SecretPattern(
            name="generic_secret",
            pattern=re.compile(
                r"""
                \b
                (
                    api[_-]?key
                    |
                    api[_-]?secret
                    |
                    secret[_-]?key
                    |
                    access[_-]?token
                    |
                    auth[_-]?token
                    |
                    client[_-]?secret
                    |
                    password
                )
                \b

                \s*
                [:=]
                \s*

                ["']?

                (
                    [A-Za-z0-9_\-./+=]{8,}
                )

                ["']?
                """,
                re.IGNORECASE | re.VERBOSE,
            ),
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

        redaction_text = str(
            config.get(
                "redaction_text",
                "[REDACTED]",
            )
        )

        detected_types: set[str] = set()

        redacted_text = text

        # ---------------------------------------------
        # Detect + redact
        # ---------------------------------------------

        for secret_pattern in (
            self.SECRET_PATTERNS
        ):

            matches = list(
                secret_pattern
                .pattern
                .finditer(
                    redacted_text
                )
            )

            if not matches:
                continue

            detected_types.add(
                secret_pattern.name
            )

            redacted_text = (
                secret_pattern
                .pattern
                .sub(
                    redaction_text,
                    redacted_text,
                )
            )

        # ---------------------------------------------
        # Optional custom regexes
        # ---------------------------------------------

        custom_patterns = config.get(
            "patterns",
            [],
        )

        for index, pattern in enumerate(
            custom_patterns
        ):

            if not isinstance(
                pattern,
                str,
            ):
                continue

            try:

                compiled = re.compile(
                    pattern,
                    re.IGNORECASE,
                )

            except re.error:
                continue

            if compiled.search(
                redacted_text
            ):

                detected_types.add(
                    f"custom_{index}"
                )

                redacted_text = (
                    compiled.sub(
                        redaction_text,
                        redacted_text,
                    )
                )

        # ---------------------------------------------
        # Nothing found
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
                "Potential secret or credential "
                "detected."
            ),
            modified_text=redacted_text,
            details={
                "secret_types": sorted(
                    detected_types
                ),
                "stage": (
                    context.stage.value
                ),
            },
        )