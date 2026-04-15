from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.enums import JobStatus


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    stored_path: str
    size_bytes: int
    content_type: str | None
    created_at: datetime


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_id: UUID
    status: JobStatus
    target_column: str
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    metrics: dict[str, Any] | None
    model_artifact_path: str | None
    created_at: datetime
