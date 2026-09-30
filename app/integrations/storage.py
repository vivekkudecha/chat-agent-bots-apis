from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
import uuid

from fastapi import UploadFile

from app.config import settings


class StorageProvider(ABC):

    @abstractmethod
    async def save_upload(
        self,
        *,
        file: UploadFile,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
    ) -> str:
        """
        Save uploaded file.

        Returns storage_key.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        storage_key: str,
    ) -> None:
        pass

    @abstractmethod
    async def exists(
        self,
        storage_key: str,
    ) -> bool:
        pass

    @abstractmethod
    async def get_local_path(
        self,
        storage_key: str,
    ) -> Path:
        """
        Return a local filesystem path.

        For remote storage implementations this may
        download the object to a temporary location.
        """
        pass


# =========================================================
# LOCAL STORAGE
# =========================================================

class LocalStorageProvider(StorageProvider):

    def __init__(
        self,
        base_path: str | Path,
    ):

        self.base_path = Path(
            base_path
        ).resolve()

        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    async def save_upload(
        self,
        *,
        file: UploadFile,
        user_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
    ) -> str:

        extension = (
            Path(file.filename or "")
            .suffix
            .lower()
        )

        file_id = uuid.uuid4()

        relative_path = Path(
            str(user_id),
            str(knowledge_base_id),
            f"{file_id}{extension}",
        )

        destination = (
            self.base_path
            / relative_path
        ).resolve()

        # Prevent path traversal.
        if self.base_path not in destination.parents:
            raise ValueError(
                "Invalid storage path."
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            await asyncio.to_thread(
                self._write_upload,
                file.file,
                destination,
            )

        finally:
            await file.seek(0)

        return str(relative_path)

    @staticmethod
    def _write_upload(
        source,
        destination: Path,
    ) -> None:

        source.seek(0)

        with destination.open(
            "wb"
        ) as output:

            while True:

                chunk = source.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(chunk)

    # -----------------------------------------------------
    # Resolve
    # -----------------------------------------------------

    def _resolve_key(
        self,
        storage_key: str,
    ) -> Path:

        path = (
            self.base_path
            / storage_key
        ).resolve()

        if (
            path != self.base_path
            and self.base_path
            not in path.parents
        ):
            raise ValueError(
                "Invalid storage key."
            )

        return path

    # -----------------------------------------------------
    # Exists
    # -----------------------------------------------------

    async def exists(
        self,
        storage_key: str,
    ) -> bool:

        path = self._resolve_key(
            storage_key
        )

        return await asyncio.to_thread(
            path.exists
        )

    # -----------------------------------------------------
    # Local Path
    # -----------------------------------------------------

    async def get_local_path(
        self,
        storage_key: str,
    ) -> Path:

        path = self._resolve_key(
            storage_key
        )

        exists = await asyncio.to_thread(
            path.exists
        )

        if not exists:
            raise FileNotFoundError(
                storage_key
            )

        return path

    # -----------------------------------------------------
    # Delete
    # -----------------------------------------------------

    async def delete(
        self,
        storage_key: str,
    ) -> None:

        path = self._resolve_key(
            storage_key
        )

        if await asyncio.to_thread(
            path.exists
        ):
            await asyncio.to_thread(
                path.unlink
            )


# =========================================================
# FACTORY
# =========================================================

_storage_provider: StorageProvider | None = None


def get_storage_provider() -> StorageProvider:

    global _storage_provider

    if _storage_provider is not None:
        return _storage_provider

    provider = (
        settings.STORAGE_PROVIDER
        .lower()
        .strip()
    )

    if provider == "local":

        _storage_provider = (
            LocalStorageProvider(
                settings.LOCAL_STORAGE_PATH
            )
        )

        return _storage_provider

    raise RuntimeError(
        f"Unsupported storage provider: {provider}"
    )