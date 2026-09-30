import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import AppException


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Application Exception
# ---------------------------------------------------------

async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


# ---------------------------------------------------------
# Pydantic / FastAPI Validation
# ---------------------------------------------------------

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:

    errors = []

    for error in exc.errors():

        errors.append(
            {
                "field": ".".join(
                    str(item)
                    for item in error.get(
                        "loc",
                        [],
                    )
                ),
                "message": error.get("msg"),
                "type": error.get("type"),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": (
                    "Request validation failed."
                ),
                "details": errors,
            }
        },
    )


# ---------------------------------------------------------
# Database Integrity Error
# ---------------------------------------------------------

async def integrity_exception_handler(
    request: Request,
    exc: IntegrityError,
) -> JSONResponse:

    logger.warning(
        "Database integrity error",
        exc_info=exc,
    )

    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "code": "database_conflict",
                "message": (
                    "The requested operation "
                    "conflicts with existing data."
                ),
                "details": None,
            }
        },
    )


# ---------------------------------------------------------
# Unexpected Exception
# ---------------------------------------------------------

async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:

    logger.exception(
        "Unhandled application exception",
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": (
                    "An unexpected error occurred."
                ),
                "details": None,
            }
        },
    )


# ---------------------------------------------------------
# Register Handlers
# ---------------------------------------------------------

def register_exception_handlers(
    app: FastAPI,
) -> None:

    app.add_exception_handler(
        AppException,
        app_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    app.add_exception_handler(
        IntegrityError,
        integrity_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unexpected_exception_handler,
    )