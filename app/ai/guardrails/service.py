import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.guardrails.base import (
    GuardrailAction,
    GuardrailContext,
    GuardrailResult,
    GuardrailStage,
)

from app.ai.guardrails.registry import (
    guardrail_registry,
)

from app.repositories.guardrail_repository import (
    GuardrailRepository,
)

from app.core.exceptions import (
    GuardrailBlockedException,
)


@dataclass
class GuardrailExecution:
    guardrail_id: uuid.UUID

    code: str
    handler: str

    action: GuardrailAction

    passed: bool

    message: str | None = None

    details: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class GuardrailEvaluation:
    allowed: bool

    original_text: str

    final_text: str

    executions: list[
        GuardrailExecution
    ] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )


class GuardrailService:

    # =====================================================
    # EVALUATE
    # =====================================================

    async def evaluate(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        text: str,
        stage: GuardrailStage,
        user_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GuardrailEvaluation:

        original_text = text
        current_text = text

        executions: list[
            GuardrailExecution
        ] = []

        warnings: list[str] = []

        # ---------------------------------------------
        # Load configured guardrails
        # ---------------------------------------------

        configured_guardrails = (
            await GuardrailRepository
            .list_for_bot(
                db,
                bot_id=bot_id,
                guardrail_type=stage.value,
                enabled_only=True,
            )
        )

        context = GuardrailContext(
            user_id=(
                str(user_id)
                if user_id
                else None
            ),
            bot_id=str(bot_id),
            conversation_id=(
                str(conversation_id)
                if conversation_id
                else None
            ),
            stage=stage,
            metadata=metadata or {},
        )

        # ---------------------------------------------
        # Execute in priority order
        # ---------------------------------------------

        for bot_guardrail in (
            configured_guardrails
        ):

            definition = (
                bot_guardrail.guardrail
            )

            if not definition:
                continue

            if not definition.is_active:
                continue

            handler_name = (
                definition.handler
                .strip()
                .lower()
            )

            # -----------------------------------------
            # Unknown handlers fail closed.
            # -----------------------------------------

            if not guardrail_registry.contains(
                handler_name
            ):

                raise GuardrailBlockedException(
                    "A configured security "
                    "guardrail is unavailable.",
                    details={
                        "guardrail": (
                            definition.code
                        )
                    },
                )

            guardrail = (
                guardrail_registry.resolve(
                    handler_name
                )
            )

            # -----------------------------------------
            # Merge configuration
            #
            # Bot config overrides defaults.
            # -----------------------------------------

            config = {
                **(
                    definition.default_config
                    or {}
                ),
                **(
                    bot_guardrail.config
                    or {}
                ),
            }

            # -----------------------------------------
            # Run
            # -----------------------------------------

            try:

                result = (
                    await guardrail.evaluate(
                        text=current_text,
                        config=config,
                        context=context,
                    )
                )

            except Exception as exc:

                # Security control failed.
                #
                # Fail closed rather than silently
                # bypassing a configured guardrail.

                raise GuardrailBlockedException(
                    "A security guardrail "
                    "could not be evaluated.",
                    details={
                        "guardrail": (
                            definition.code
                        )
                    },
                ) from exc

            action = self._parse_action(
                bot_guardrail.action
            )

            execution = GuardrailExecution(
                guardrail_id=(
                    definition.id
                ),
                code=definition.code,
                handler=handler_name,
                action=action,
                passed=result.passed,
                message=result.message,
                details=result.details,
            )

            executions.append(
                execution
            )

            # -----------------------------------------
            # Passed
            # -----------------------------------------

            if result.passed:
                continue

            # -----------------------------------------
            # ALLOW
            #
            # Record detection but continue.
            # -----------------------------------------

            if action == GuardrailAction.ALLOW:
                continue

            # -----------------------------------------
            # WARN
            # -----------------------------------------

            if action == GuardrailAction.WARN:

                warnings.append(
                    result.message
                    or (
                        f"Guardrail "
                        f"{definition.code} "
                        f"triggered."
                    )
                )

                continue

            # -----------------------------------------
            # REDACT
            # -----------------------------------------

            if action == GuardrailAction.REDACT:

                if (
                    result.modified_text
                    is None
                ):
                    # A REDACT guardrail that cannot
                    # produce safe text should not
                    # silently pass the original text.

                    raise GuardrailBlockedException(
                        result.message
                        or (
                            "Content could not "
                            "be safely redacted."
                        ),
                        details={
                            "guardrail": (
                                definition.code
                            )
                        },
                    )

                current_text = (
                    result.modified_text
                )

                continue

            # -----------------------------------------
            # BLOCK
            # -----------------------------------------

            if action == GuardrailAction.BLOCK:

                raise GuardrailBlockedException(
                    result.message
                    or (
                        "Request blocked by "
                        "security policy."
                    ),
                    details={
                        "guardrail": (
                            definition.code
                        )
                    },
                )

        return GuardrailEvaluation(
            allowed=True,
            original_text=original_text,
            final_text=current_text,
            executions=executions,
            warnings=warnings,
        )

    # =====================================================
    # ACTION PARSER
    # =====================================================

    @staticmethod
    def _parse_action(
        action: str,
    ) -> GuardrailAction:

        try:

            return GuardrailAction(
                action.lower().strip()
            )

        except ValueError as exc:

            # Invalid security configuration
            # should fail closed.

            raise GuardrailBlockedException(
                "Invalid guardrail action "
                "configuration."
            ) from exc