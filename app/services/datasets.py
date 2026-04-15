import re
from pathlib import Path
from uuid import UUID, uuid4

import structlog
from fastapi import UploadFile

from app.consts import UPLOAD_CHUNK_BYTES
from app.db.datasets import Dataset, DatasetsRepo
from app.settings import Settings

logger = structlog.get_logger(__name__)


class DatasetService:
    def __init__(self, settings: Settings, datasets_repo: DatasetsRepo) -> None:
        self._settings = settings
        self._datasets_repo = datasets_repo

    @staticmethod
    def _safe_filename(name: str) -> str:
        base = Path(name).name
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._")

        if not cleaned:
            return "upload.bin"

        return cleaned[:200]

    async def save_upload(self, upload: UploadFile) -> Dataset:
        self._settings.storage_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = uuid4()
        safe_name = self._safe_filename(upload.filename or "upload.csv")
        relative_dir = Path(str(dataset_id))
        target_dir = (self._settings.storage_dir / relative_dir).resolve()

        if not str(target_dir).startswith(str(self._settings.storage_dir.resolve())):
            msg = "Некорректный путь сохранения файла"
            raise ValueError(msg)

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        max_bytes = self._settings.max_upload_size_mb * 1024 * 1024
        total = 0

        logger.info("Начало загрузки файла", filename=safe_name, dataset_id=str(dataset_id))

        with target_path.open(mode="wb") as file_handle:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)

                if not chunk:
                    break

                total += len(chunk)

                if total > max_bytes:
                    target_path.unlink(missing_ok=True)
                    msg = f"Файл превышает лимит {self._settings.max_upload_size_mb} МБ"
                    raise ValueError(msg)

                file_handle.write(chunk)

        dataset = await self._datasets_repo.create(
            dataset_id=dataset_id,
            original_filename=upload.filename or safe_name,
            stored_path=str(target_path),
            size_bytes=total,
            content_type=upload.content_type,
        )

        logger.info(
            "Файл сохранён",
            dataset_id=str(dataset.id),
            size_bytes=total,
            path=str(target_path),
        )

        return dataset

    async def list_datasets(self) -> list[Dataset]:
        return await self._datasets_repo.list_all()

    async def get_dataset(self, dataset_id: UUID) -> Dataset | None:
        return await self._datasets_repo.get_by_id(dataset_id)
