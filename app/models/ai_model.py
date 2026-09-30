import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class AIModel(Base):
    __tablename__ = "ai_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    model_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_type: Mapped[str] = mapped_column(
        String(30),
        default="chat",
        nullable=False,
    )

    endpoint_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    context_window: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    capabilities: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "model_key",
            name="uq_ai_model_provider_key",
        ),
    )


class BotModelConfig(Base):
    __tablename__ = "bot_model_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    bot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "bots.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "ai_models.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    temperature: Mapped[float] = mapped_column(
        Float,
        default=0.7,
        nullable=False,
    )

    top_p: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        nullable=False,
    )

    max_tokens: Mapped[int] = mapped_column(
        Integer,
        default=2048,
        nullable=False,
    )

    config: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    bot = relationship(
        "Bot",
        back_populates="model_configs",
    )

    model = relationship(
        "AIModel",
    )