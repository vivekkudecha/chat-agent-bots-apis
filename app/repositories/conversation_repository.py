import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.conversation import (
    Conversation,
    Message,
)


class ConversationRepository:

    # =====================================================
    # CONVERSATIONS
    # =====================================================

    @staticmethod
    async def create(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> Conversation:

        conversation = Conversation(
            user_id=user_id,
            bot_id=bot_id,
            title=title,
            metadata_=metadata or {},
        )

        db.add(conversation)

        await db.flush()
        await db.refresh(conversation)

        return conversation

    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> Conversation | None:

        result = await db.execute(
            select(Conversation).where(
                Conversation.id
                == conversation_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_owned(
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation | None:

        result = await db.execute(
            select(Conversation).where(
                Conversation.id
                == conversation_id,
                Conversation.user_id
                == user_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def get_with_messages(
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation | None:

        result = await db.execute(
            select(Conversation)
            .options(
                selectinload(
                    Conversation.messages
                )
            )
            .where(
                Conversation.id
                == conversation_id,

                Conversation.user_id
                == user_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_user(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        bot_id: uuid.UUID | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Conversation], int]:

        conditions = [
            Conversation.user_id == user_id
        ]

        if bot_id:
            conditions.append(
                Conversation.bot_id == bot_id
            )

        count_result = await db.execute(
            select(
                func.count(
                    Conversation.id
                )
            ).where(*conditions)
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(Conversation)
            .where(*conditions)
            .order_by(
                Conversation.updated_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        conversations = list(
            result.scalars().all()
        )

        return conversations, total

    @staticmethod
    async def update(
        db: AsyncSession,
        conversation: Conversation,
        **values,
    ) -> Conversation:

        allowed_fields = {
            "title",
            "metadata_",
        }

        for key, value in values.items():
            if key in allowed_fields:
                setattr(
                    conversation,
                    key,
                    value,
                )

        await db.flush()
        await db.refresh(conversation)

        return conversation

    @staticmethod
    async def delete(
        db: AsyncSession,
        conversation: Conversation,
    ) -> None:

        await db.delete(conversation)
        await db.flush()

    # =====================================================
    # MESSAGES
    # =====================================================

    @staticmethod
    async def create_message(
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        model_id: uuid.UUID | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        metadata: dict | None = None,
    ) -> Message:

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            metadata_=metadata or {},
        )

        db.add(message)

        await db.flush()
        await db.refresh(message)

        return message

    @staticmethod
    async def get_message(
        db: AsyncSession,
        message_id: uuid.UUID,
    ) -> Message | None:

        result = await db.execute(
            select(Message).where(
                Message.id == message_id
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    async def list_messages(
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Message], int]:

        count_result = await db.execute(
            select(
                func.count(Message.id)
            ).where(
                Message.conversation_id
                == conversation_id
            )
        )

        total = count_result.scalar_one()

        result = await db.execute(
            select(Message)
            .where(
                Message.conversation_id
                == conversation_id
            )
            .order_by(
                Message.created_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        messages = list(
            result.scalars().all()
        )

        return messages, total

    @staticmethod
    async def get_recent_messages(
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        limit: int = 20,
    ) -> list[Message]:

        result = await db.execute(
            select(Message)
            .where(
                Message.conversation_id
                == conversation_id
            )
            .order_by(
                Message.created_at.desc()
            )
            .limit(limit)
        )

        messages = list(
            result.scalars().all()
        )

        # Query fetched newest → oldest.
        # LLM needs oldest → newest.
        messages.reverse()

        return messages

    @staticmethod
    async def delete_message(
        db: AsyncSession,
        message: Message,
    ) -> None:

        await db.delete(message)
        await db.flush()