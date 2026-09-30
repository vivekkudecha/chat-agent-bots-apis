import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
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


class Guardrail(Base):
    __tablename__ = "guardrails"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    guardrail_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    handler: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    default_config: Mapped[dict] = mapped_column(
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

    bot_links: Mapped[list["BotGuardrail"]] = relationship(
        back_populates="guardrail",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class BotGuardrail(Base):
    __tablename__ = "bot_guardrails"

    bot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "bots.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    guardrail_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "guardrails.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    config: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    action: Mapped[str] = mapped_column(
        String(30),
        default="block",
        nullable=False,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=100,
        nullable=False,
    )

    is_enabled: Mapped[bool] = mapped_column(
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
        back_populates="guardrail_links",
    )

    guardrail = relationship(
        "Guardrail",
        back_populates="bot_links",
    )

    __table_args__ = (
        UniqueConstraint(
            "bot_id",
            "guardrail_id",
            name="uq_bot_guardrail",
        ),
    )