from collections.abc import Callable

from app.guardrails.base import (
    BaseGuardrail,
)


GuardrailFactory = Callable[
    [],
    BaseGuardrail,
]


class GuardrailRegistry:

    def __init__(self):

        self._handlers: dict[
            str,
            GuardrailFactory,
        ] = {}

    # -----------------------------------------------------
    # Register
    # -----------------------------------------------------

    def register(
        self,
        handler_name: str,
        factory: GuardrailFactory,
    ) -> None:

        normalized = (
            handler_name
            .strip()
            .lower()
        )

        if not normalized:
            raise ValueError(
                "Guardrail handler name "
                "cannot be empty."
            )

        if normalized in self._handlers:
            raise ValueError(
                "Guardrail handler already "
                f"registered: {normalized}"
            )

        self._handlers[
            normalized
        ] = factory

    # -----------------------------------------------------
    # Resolve
    # -----------------------------------------------------

    def resolve(
        self,
        handler_name: str,
    ) -> BaseGuardrail:

        normalized = (
            handler_name
            .strip()
            .lower()
        )

        factory = self._handlers.get(
            normalized
        )

        if not factory:
            raise KeyError(
                "Unknown guardrail handler: "
                f"{normalized}"
            )

        return factory()

    # -----------------------------------------------------
    # Exists
    # -----------------------------------------------------

    def contains(
        self,
        handler_name: str,
    ) -> bool:

        return (
            handler_name
            .strip()
            .lower()
            in self._handlers
        )

    # -----------------------------------------------------
    # List
    # -----------------------------------------------------

    def registered_handlers(
        self,
    ) -> list[str]:

        return sorted(
            self._handlers.keys()
        )


# =========================================================
# Global Registry
# =========================================================

guardrail_registry = (
    GuardrailRegistry()
)


# =========================================================
# Decorator
# =========================================================

def register_guardrail(
    handler_name: str,
):
    """
    Convenience decorator.

    Example:

        @register_guardrail("secret_detection")
        class SecretDetectionGuardrail(...):
            ...
    """

    def decorator(
        guardrail_class: type[
            BaseGuardrail
        ],
    ):

        guardrail_registry.register(
            handler_name,
            guardrail_class,
        )

        return guardrail_class

    return decorator