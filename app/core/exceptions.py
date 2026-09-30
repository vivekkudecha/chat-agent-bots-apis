from typing import Any


class AppException(Exception):
    """
    Base exception for application-level errors.
    """

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Any | None = None,
    ):
        self.message = message or self.message
        self.details = details

        super().__init__(self.message)


# ---------------------------------------------------------
# 400
# ---------------------------------------------------------

class BadRequestException(AppException):

    status_code = 400
    code = "bad_request"
    message = "Invalid request."


class ValidationException(AppException):

    status_code = 400
    code = "validation_error"
    message = "Request validation failed."


# ---------------------------------------------------------
# 401
# ---------------------------------------------------------

class AuthenticationException(AppException):

    status_code = 401
    code = "authentication_required"
    message = "Authentication required."


class InvalidCredentialsException(
    AuthenticationException
):

    code = "invalid_credentials"
    message = "Invalid email or password."


class InvalidTokenException(
    AuthenticationException
):

    code = "invalid_token"
    message = "Invalid or expired token."


# ---------------------------------------------------------
# 403
# ---------------------------------------------------------

class PermissionDeniedException(AppException):

    status_code = 403
    code = "permission_denied"
    message = "You do not have permission to perform this action."


# ---------------------------------------------------------
# 404
# ---------------------------------------------------------

class NotFoundException(AppException):

    status_code = 404
    code = "not_found"
    message = "Resource not found."


class UserNotFoundException(
    NotFoundException
):

    code = "user_not_found"
    message = "User not found."


class BotNotFoundException(
    NotFoundException
):

    code = "bot_not_found"
    message = "Bot not found."


class KnowledgeBaseNotFoundException(
    NotFoundException
):

    code = "knowledge_base_not_found"
    message = "Knowledge base not found."


class DocumentNotFoundException(
    NotFoundException
):

    code = "document_not_found"
    message = "Document not found."


class ModelNotFoundException(
    NotFoundException
):

    code = "model_not_found"
    message = "AI model not found."


class ToolNotFoundException(
    NotFoundException
):

    code = "tool_not_found"
    message = "Tool not found."


class GuardrailNotFoundException(
    NotFoundException
):

    code = "guardrail_not_found"
    message = "Guardrail not found."


class ConversationNotFoundException(
    NotFoundException
):

    code = "conversation_not_found"
    message = "Conversation not found."


# ---------------------------------------------------------
# 409
# ---------------------------------------------------------

class ConflictException(AppException):

    status_code = 409
    code = "conflict"
    message = "Resource conflict."


class EmailAlreadyExistsException(
    ConflictException
):

    code = "email_already_exists"
    message = "A user with this email already exists."


class BotSlugAlreadyExistsException(
    ConflictException
):

    code = "bot_slug_already_exists"
    message = "A bot with this slug already exists."


class DuplicateDocumentException(
    ConflictException
):

    code = "duplicate_document"
    message = "This document already exists in the knowledge base."


# ---------------------------------------------------------
# AI / Guardrail
# ---------------------------------------------------------

class GuardrailBlockedException(
    AppException
):

    status_code = 400
    code = "guardrail_blocked"
    message = "The request was blocked by a guardrail."


class ModelExecutionException(
    AppException
):

    status_code = 502
    code = "model_execution_failed"
    message = "AI model execution failed."


class ToolExecutionException(
    AppException
):

    status_code = 502
    code = "tool_execution_failed"
    message = "Tool execution failed."


class RetrievalException(
    AppException
):

    status_code = 500
    code = "retrieval_failed"
    message = "Knowledge retrieval failed."