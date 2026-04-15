import asyncio
from pathlib import Path
from uuid import UUID

import structlog

from app.consts import utc_now_naive
from app.db.datasets import DatasetsRepo
from app.db.jobs import Job, JobsRepo
from app.enums import JobStatus
from app.services.sklearn_pipeline import run_sklearn_training_pipeline
from app.settings import Settings

logger = structlog.get_logger(__name__)


class TrainingService:
    def __init__(
        self,
        settings: Settings,
        jobs_repo: JobsRepo,
        datasets_repo: DatasetsRepo,
    ) -> None:
        self._settings = settings
        self._jobs_repo = jobs_repo
        self._datasets_repo = datasets_repo

    async def start_training(self, *, dataset_id: UUID, target_column: str) -> Job:
        dataset = await self._datasets_repo.get_by_id(dataset_id)

        if dataset is None:
            msg = "Датасет не найден"
            raise ValueError(msg)

        job = await self._jobs_repo.create(dataset_id=dataset_id, target_column=target_column)

        logger.info(
            "Создана задача обучения",
            job_id=str(job.id),
            dataset_id=str(dataset_id),
            target_column=target_column,
        )

        return job

    async def run_job(self, job_id: UUID) -> None:
        job = await self._jobs_repo.get_by_id(job_id)

        if job is None:
            logger.warning("Задача не найдена", job_id=str(job_id))

            return

        dataset = await self._datasets_repo.get_by_id(job.dataset_id)

        if dataset is None:
            job.status = JobStatus.FAILED
            job.error_message = "Датасет для задачи не найден"
            job.finished_at = utc_now_naive()
            await self._jobs_repo.update_job(job)

            return

        job.status = JobStatus.RUNNING
        job.started_at = utc_now_naive()
        job.error_message = None
        await self._jobs_repo.update_job(job)

        model_dir = self._settings.storage_dir / "models" / str(job.id)

        try:
            metrics = await asyncio.to_thread(
                run_sklearn_training_pipeline,
                dataset_path=Path(dataset.stored_path),
                target_column=job.target_column,
                model_dir=model_dir,
            )
            job.metrics = metrics
            job.model_artifact_path = str(model_dir)
            job.status = JobStatus.DONE
            job.finished_at = utc_now_naive()

            logger.info(
                "Обучение завершено",
                job_id=str(job.id),
                task_type=metrics.get("task_type"),
                test_metrics=metrics.get("test_metrics"),
            )
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = utc_now_naive()

            logger.exception("Ошибка обучения", job_id=str(job.id))

        await self._jobs_repo.update_job(job)

    async def get_job(self, job_id: UUID) -> Job | None:
        return await self._jobs_repo.get_by_id(job_id)
