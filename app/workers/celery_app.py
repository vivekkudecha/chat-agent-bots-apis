from celery import Celery

from app.config import settings


celery_app = Celery(
    "ai_bot_platform",

    broker=settings.CELERY_BROKER_URL,

    backend=settings.CELERY_RESULT_BACKEND,

    include=[
        "app.workers.document_tasks",
        "app.workers.memory_tasks",
    ],
)


celery_app.conf.update(

    # -----------------------------------------
    # Serialization
    # -----------------------------------------

    task_serializer="json",
    result_serializer="json",

    accept_content=[
        "json"
    ],

    # -----------------------------------------
    # Timezone
    # -----------------------------------------

    timezone="UTC",

    enable_utc=True,

    # -----------------------------------------
    # Task reliability
    # -----------------------------------------

    task_acks_late=True,

    task_reject_on_worker_lost=True,

    # -----------------------------------------
    # Worker behavior
    # -----------------------------------------

    worker_prefetch_multiplier=1,

    # -----------------------------------------
    # Result expiration
    # -----------------------------------------

    result_expires=3600,

    # -----------------------------------------
    # Tracking
    # -----------------------------------------

    task_track_started=True,

    # -----------------------------------------
    # Time limits
    # -----------------------------------------

    task_soft_time_limit=(
        settings.CELERY_TASK_SOFT_TIME_LIMIT
    ),

    task_time_limit=(
        settings.CELERY_TASK_TIME_LIMIT
    ),
)


# ---------------------------------------------------------
# Routing
# ---------------------------------------------------------

celery_app.conf.task_routes = {

    "documents.process": {
        "queue": "documents"
    },

    "documents.delete_vectors": {
        "queue": "documents"
    },
}