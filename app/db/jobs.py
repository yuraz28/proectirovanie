from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, String
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel, col, select

from app.consts import utc_now_naive
from app.db.base import BaseRepo
from app.enums import JobStatus


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    dataset_id: UUID = Field(foreign_key="datasets.id", index=True)
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        sa_column=Column(String(32), nullable=False),
    )
    target_column: str = Field(default="label", max_length=256)
    started_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    error_message: str | None = Field(default=None, max_length=8192)
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    model_artifact_path: str | None = Field(default=None, max_length=2048)
    created_at: datetime = Field(
        default_factory=utc_now_naive,
        sa_column_kwargs={"nullable": False},
    )


class JobsRepo(BaseRepo):
    async def create(self, *, dataset_id: UUID, target_column: str) -> Job:
        job = Job(dataset_id=dataset_id, target_column=target_column, status=JobStatus.PENDING)

        async with self._session() as session:
            session.add(job)
            await session.flush()
            await session.refresh(job)

        return job

    async def get_by_id(self, job_id: UUID) -> Job | None:
        async with self._session() as session:
            return await session.get(Job, job_id)

    async def update_job(self, job: Job) -> Job:
        async with self._session() as session:
            session.add(job)
            await session.flush()
            await session.refresh(job)

        return job

    async def list_for_dataset(self, dataset_id: UUID) -> list[Job]:
        async with self._session() as session:
            result = await session.exec(
                select(Job)
                .where(Job.dataset_id == dataset_id)
                .order_by(col(Job.created_at).desc()),
            )

            return list(result.all())
