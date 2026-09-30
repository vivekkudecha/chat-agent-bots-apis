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


class Bot(Base):
    __tablename__ = "bots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    slug: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="draft",
        nullable=False,
        index=True,
    )

    visibility: Mapped[str] = mapped_column(
        String(30),
        default="private",
        nullable=False,
    )

    avatar_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    is_api_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    versions: Mapped[list["BotVersion"]] = relationship(
        back_populates="bot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    model_configs = relationship(
        "BotModelConfig",
        back_populates="bot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    knowledge_base_links = relationship(
        "BotKnowledgeBase",
        back_populates="bot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    tool_links = relationship(
        "BotTool",
        back_populates="bot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    guardrail_links = relationship(
        "BotGuardrail",
        back_populates="bot",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "slug",
            name="uq_bot_user_slug",
        ),
    )


class BotVersion(Base):
    __tablename__ = "bot_versions"

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

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    system_instruction: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    welcome_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    conversation_starters: Mapped[list] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    config: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    bot: Mapped["Bot"] = relationship(
        back_populates="versions",
    )

    __table_args__ = (
        UniqueConstraint(
            "bot_id",
            "version",
            name="uq_bot_version",
        ),
    )