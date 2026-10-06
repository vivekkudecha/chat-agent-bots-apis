import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import (
    BotNotFoundException,
    KnowledgeBaseNotFoundException,
    ValidationException,
)
from app.models.bot import Bot, BotVersion
from app.repositories.audit_repository import AuditRepository
from app.repositories.bot_repository import BotRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_service import DocumentService
from app.ai.rag import TextExtractionService

logger = logging.getLogger(__name__)


@dataclass
class BotListResult:
    items: list[Any]
    total: int


class BotService:

    @staticmethod
    async def _generate_unique_slug(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        name: str,
        requested_slug: str | None = None,
    ) -> str:
        raw = requested_slug or name
        base = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:100]
        if not base:
            base = "bot"

        # If user did not provide a slug, check if base slug is free
        existing = await BotRepository.get_by_slug(db, user_id=user_id, slug=base)
        if not existing:
            return base

        # If base is taken, generate unique collision-free suffix
        for _ in range(10):
            candidate = f"{base}-{uuid.uuid4().hex[:6]}"
            if not await BotRepository.get_by_slug(db, user_id=user_id, slug=candidate):
                return candidate

        return f"{base}-{uuid.uuid4().hex[:10]}"

    # =====================================================
    # CREATE
    # =====================================================

    async def create(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        name: str,
        slug: str | None = None,
        description: str | None = None,
        system_instruction: str | None = None,
        welcome_message: str | None = None,
        conversation_starters: list[str] | None = None,
        visibility: str = "private",
        avatar_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Bot:

        bot_slug = await self._generate_unique_slug(
            db,
            user_id=user_id,
            name=name,
            requested_slug=slug,
        )

        bot = await BotRepository.create(
            db,
            user_id=user_id,
            name=name.strip(),
            slug=bot_slug,
            description=description,
            visibility=visibility,
            avatar_url=avatar_url,
            metadata=metadata or {},
        )

        if system_instruction:
            await BotRepository.create_version(
                db,
                bot_id=bot.id,
                system_instruction=system_instruction,
                welcome_message=welcome_message,
                conversation_starters=conversation_starters or [],
            )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot",
            resource_id=bot.id,
            action="bot.created",
            status="success",
        )

        await db.commit()

        bot_with_versions = await BotRepository.get_with_versions(
            db,
            bot_id=bot.id,
        )
        return bot_with_versions or bot

    # =====================================================
    # GET
    # =====================================================

    async def get(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> Bot:

        bot = await BotRepository.get_available_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
            is_admin=is_admin,
        )

        if not bot:
            raise BotNotFoundException()

        return bot

    # =====================================================
    # LIST
    # =====================================================

    async def list(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        is_admin: bool = False,
        page: int = 1,
        page_size: int = 20,
    ) -> BotListResult:

        offset = (page - 1) * page_size

        items, total = await BotRepository.list_available(
            db,
            user_id=user_id,
            is_admin=is_admin,
            offset=offset,
            limit=page_size,
        )

        return BotListResult(
            items=items,
            total=total,
        )

    # =====================================================
    # UPDATE DRAFT
    # =====================================================

    async def update(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
        **update_data,
    ) -> Bot:

        bot = await self.get(
            db,
            bot_id=bot_id,
            user_id=user_id,
            is_admin=is_admin,
        )

        protected_fields = {
            "id",
            "user_id",
            "created_at",
            "updated_at",
        }

        bot_updates = {
            key: value
            for key, value in update_data.items()
            if key not in protected_fields
            and key not in {"system_instruction", "welcome_message", "conversation_starters"}
            and value is not None
        }

        if bot_updates:
            bot = await BotRepository.update(
                db,
                bot=bot,
                **bot_updates,
            )

        instruction = update_data.get("system_instruction")
        welcome = update_data.get("welcome_message")
        starters = update_data.get("conversation_starters")

        if instruction or welcome or starters is not None:
            latest = await BotRepository.get_latest_version(
                db,
                bot_id=bot.id,
            )
            inst = instruction if instruction is not None else (latest.system_instruction if latest else "")
            welc = welcome if welcome is not None else (latest.welcome_message if latest else None)
            start = starters if starters is not None else (latest.conversation_starters if latest else [])

            if inst:
                await BotRepository.create_version(
                    db,
                    bot_id=bot.id,
                    system_instruction=inst,
                    welcome_message=welc,
                    conversation_starters=start,
                )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot",
            resource_id=bot.id,
            action="bot.updated",
            status="success",
            metadata={"fields": list(update_data.keys())},
        )

        await db.commit()

        bot_refreshed = await BotRepository.get_with_versions(
            db,
            bot_id=bot.id,
        )
        return bot_refreshed or bot

    # =====================================================
    # DELETE
    # =====================================================

    async def delete(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> None:

        if is_admin:
            bot = await BotRepository.get_by_id(
                db,
                bot_id=bot_id,
            )
        else:
            bot = await BotRepository.get_owned_bot(
                db,
                bot_id=bot_id,
                user_id=user_id,
            )

        if not bot:
            raise BotNotFoundException()

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot",
            resource_id=bot.id,
            action="bot.deleted",
            status="success",
        )

        await BotRepository.delete(
            db,
            bot=bot,
        )

        await db.commit()

    # =====================================================
    # PUBLISH VERSION
    # =====================================================

    async def publish(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> BotVersion:

        bot = await self.get(
            db,
            bot_id=bot_id,
            user_id=user_id,
            is_admin=is_admin,
        )

        latest_version = await BotRepository.get_latest_version(
            db,
            bot_id=bot.id,
        )

        instruction = latest_version.system_instruction if latest_version else "You are a helpful AI assistant."
        welcome = latest_version.welcome_message if latest_version else None
        starters = latest_version.conversation_starters if latest_version else []

        version = await BotRepository.create_version(
            db,
            bot_id=bot.id,
            system_instruction=instruction,
            welcome_message=welcome,
            conversation_starters=starters,
        )

        await BotRepository.update(
            db,
            bot=bot,
            status="published",
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="bot_version",
            resource_id=version.id,
            action="bot.published",
            status="success",
            metadata={"version": version.version},
        )

        await db.commit()
        await db.refresh(version)

        return version

    # =====================================================
    # LIST VERSIONS
    # =====================================================

    async def list_versions(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[BotVersion]:

        bot = await BotRepository.get_owned_bot(
            db,
            bot_id=bot_id,
            user_id=user_id,
        )

        if not bot:
            raise BotNotFoundException()

        return await BotRepository.list_versions(
            db,
            bot_id=bot.id,
        )

    # =====================================================
    # ATTACH KNOWLEDGE BASE
    # =====================================================

    async def attach_knowledge_base(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> None:

        if is_admin:
            bot = await BotRepository.get_by_id(
                db,
                bot_id=bot_id,
            )
        else:
            bot = await BotRepository.get_owned_bot(
                db,
                bot_id=bot_id,
                user_id=user_id,
            )

        if not bot:
            raise BotNotFoundException()

        if is_admin:
            knowledge_base = await KnowledgeRepository.get_by_id(
                db,
                knowledge_base_id=knowledge_base_id,
            )
        else:
            knowledge_base = await KnowledgeRepository.get_owned(
                db,
                knowledge_base_id=knowledge_base_id,
                user_id=user_id,
            )

        if not knowledge_base:
            raise KnowledgeBaseNotFoundException()

        existing = await KnowledgeRepository.get_bot_knowledge_base(
            db,
            bot_id=bot.id,
            knowledge_base_id=knowledge_base.id,
        )

        if existing:
            return

        await KnowledgeRepository.attach_to_bot(
            db,
            bot_id=bot.id,
            knowledge_base_id=knowledge_base.id,
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="knowledge_base",
            resource_id=knowledge_base.id,
            action="bot.knowledge_base.attached",
            status="success",
        )

        await db.commit()

    # =====================================================
    # DETACH KNOWLEDGE BASE
    # =====================================================

    async def detach_knowledge_base(
        self,
        db: AsyncSession,
        *,
        bot_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> None:

        if is_admin:
            bot = await BotRepository.get_by_id(
                db,
                bot_id=bot_id,
            )
        else:
            bot = await BotRepository.get_owned_bot(
                db,
                bot_id=bot_id,
                user_id=user_id,
            )

        if not bot:
            raise BotNotFoundException()

        if is_admin:
            knowledge_base = await KnowledgeRepository.get_by_id(
                db,
                knowledge_base_id=knowledge_base_id,
            )
        else:
            knowledge_base = await KnowledgeRepository.get_owned(
                db,
                knowledge_base_id=knowledge_base_id,
                user_id=user_id,
            )

        if not knowledge_base:
            raise KnowledgeBaseNotFoundException()

        await KnowledgeRepository.detach_from_bot(
            db,
            bot_id=bot.id,
            knowledge_base_id=knowledge_base.id,
        )

        await AuditRepository.create(
            db,
            user_id=user_id,
            bot_id=bot.id,
            resource_type="knowledge_base",
            resource_id=knowledge_base.id,
            action="bot.knowledge_base.detached",
            status="success",
        )

        await db.commit()

    # =====================================================
    # CREATE WITH DOCUMENTS
    # =====================================================

    async def create_with_documents(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        name: str,
        slug: str | None = None,
        description: str | None = None,
        system_instruction: str | None = None,
        welcome_message: str | None = None,
        conversation_starters: list[str] | None = None,
        visibility: str = "private",
        avatar_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        knowledge_base_id: uuid.UUID | None = None,
        knowledge_base_name: str | None = None,
        files: list[UploadFile] | None = None,
    ) -> dict[str, Any]:

        valid_files = [f for f in (files or []) if f.filename]
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        # Validate file extensions & sizes upfront
        for f in valid_files:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in TextExtractionService.SUPPORTED_EXTENSIONS:
                raise ValidationException(
                    f"Unsupported file format '{f.filename}'. Supported formats: {', '.join(sorted(TextExtractionService.SUPPORTED_EXTENSIONS))}"
                )
            if f.size is not None and f.size > max_bytes:
                raise ValidationException(
                    f"File '{f.filename}' exceeds the maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB."
                )

        # 1. Create Bot
        bot = await self.create(
            db,
            user_id=user_id,
            name=name,
            slug=slug,
            description=description,
            system_instruction=system_instruction,
            welcome_message=welcome_message,
            conversation_starters=conversation_starters,
            visibility=visibility,
            avatar_url=avatar_url,
            metadata=metadata,
        )

        # 2. Resolve or Create Knowledge Base
        if knowledge_base_id:
            kb = await KnowledgeRepository.get_owned(
                db,
                knowledge_base_id=knowledge_base_id,
                user_id=user_id,
            )
            if not kb:
                raise KnowledgeBaseNotFoundException(
                    "Specified knowledge base was not found."
                )
        else:
            kb_title = knowledge_base_name or f"{bot.name} Knowledge Base"
            kb_desc = f"Auto-created knowledge base for bot {bot.name}"
            kb = await KnowledgeRepository.create(
                db,
                user_id=user_id,
                name=kb_title,
                description=kb_desc,
            )
            await db.commit()
            await db.refresh(kb)

        # 3. Attach Knowledge Base to Bot
        await self.attach_knowledge_base(
            db,
            bot_id=bot.id,
            knowledge_base_id=kb.id,
            user_id=user_id,
        )

        # 4. Upload, Chunk, and Embed files synchronously
        document_service = DocumentService()
        uploaded_documents = []
        processed_count = 0
        failed_count = 0
        total_chunks = 0

        for file in valid_files:
            try:
                # Upload document
                doc = await document_service.upload(
                    db,
                    user_id=user_id,
                    knowledge_base_id=kb.id,
                    file=file,
                )
                await db.commit()
                await db.refresh(doc)

                # Process: extract, chunk, embed, and store in vector DB
                processed_doc = await document_service.process_document(
                    db,
                    document_id=doc.id,
                )
                uploaded_documents.append(processed_doc)

                if processed_doc.status == "ready":
                    processed_count += 1
                    total_chunks += (processed_doc.chunk_count or 0)
                else:
                    failed_count += 1

            except Exception as exc:
                logger.error("Failed processing file %s: %s", file.filename, exc)
                failed_count += 1
                uploaded_documents.append(
                    {
                        "id": uuid.uuid4(),
                        "original_name": file.filename or "unknown",
                        "file_size": file.size or 0,
                        "status": "failed",
                        "chunk_count": 0,
                        "error_message": str(exc),
                        "created_at": datetime.now(timezone.utc),
                    }
                )

        # 5. Fetch updated bot with versions
        bot_with_versions = await BotRepository.get_with_versions(
            db,
            bot_id=bot.id,
        )
        final_bot = bot_with_versions or bot

        return {
            "id": final_bot.id,
            "user_id": final_bot.user_id,
            "name": final_bot.name,
            "slug": final_bot.slug,
            "description": final_bot.description,
            "status": final_bot.status,
            "visibility": final_bot.visibility,
            "avatar_url": final_bot.avatar_url,
            "is_api_enabled": final_bot.is_api_enabled,
            "metadata_": final_bot.metadata_,
            "created_at": final_bot.created_at,
            "updated_at": final_bot.updated_at,
            "versions": getattr(final_bot, "versions", []),
            "knowledge_base_id": kb.id,
            "knowledge_base_name": kb.name,
            "documents": uploaded_documents,
            "summary": {
                "total_files": len(valid_files),
                "processed_files": processed_count,
                "failed_files": failed_count,
                "total_chunks": total_chunks,
            },
        }