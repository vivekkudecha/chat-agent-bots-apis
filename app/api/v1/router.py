from fastapi import APIRouter

# TODO: Uncomment these when corresponding endpoint files are implemented in app/api/v1/
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.bots import router as bots_router
from app.api.v1.knowledge_bases import router as knowledge_bases_router
from app.api.v1.documents import router as documents_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.models import router as models_router
from app.api.v1.guardrails import router as guardrails_router

from app.api.v1.chat import (
    router as chat_router,
)


api_router = APIRouter()


# =========================================================
# ROUTERS
# =========================================================

api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(bots_router)
api_router.include_router(knowledge_bases_router)
api_router.include_router(documents_router)
api_router.include_router(conversations_router)
api_router.include_router(models_router)
api_router.include_router(guardrails_router)


# =========================================================
# CHAT
# =========================================================

api_router.include_router(
    chat_router,
)