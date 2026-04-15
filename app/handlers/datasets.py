from typing import Annotated
from uuid import UUID

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from app.container import Container
from app.schemas.responses import DatasetResponse, JobResponse
from app.services.datasets import DatasetService
from app.services.training import TrainingService

datasets_router = APIRouter(prefix="/api/datasets", tags=["Датасеты"])


async def _run_training_job(service: TrainingService, job_id: UUID) -> None:
    await service.run_job(job_id)


@datasets_router.post(
    path="/upload",
    status_code=HTTP_201_CREATED,
    summary="Загрузить датасет",
)
@inject
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    service: Annotated[DatasetService, Depends(Provide[Container.dataset_service])],
) -> DatasetResponse:
    try:
        dataset = await service.save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DatasetResponse.model_validate(dataset)


@datasets_router.get(path="/", summary="Список датасетов")
@inject
async def list_datasets(
    service: Annotated[DatasetService, Depends(Provide[Container.dataset_service])],
) -> list[DatasetResponse]:
    datasets = await service.list_datasets()

    return [DatasetResponse.model_validate(item) for item in datasets]


@datasets_router.post(
    path="/{dataset_id}/train",
    summary="Запустить обучение модели",
)
@inject
async def start_training(
    dataset_id: UUID,
    background_tasks: BackgroundTasks,
    training_service: Annotated[TrainingService, Depends(Provide[Container.training_service])],
    target_column: str = "label",
) -> JobResponse:
    try:
        job = await training_service.start_training(
            dataset_id=dataset_id,
            target_column=target_column,
        )
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    background_tasks.add_task(_run_training_job, training_service, job.id)

    return JobResponse.model_validate(job)


@datasets_router.get(
    path="/{dataset_id}",
    summary="Получить датасет",
)
@inject
async def get_dataset(
    dataset_id: UUID,
    service: Annotated[DatasetService, Depends(Provide[Container.dataset_service])],
) -> DatasetResponse:
    dataset = await service.get_dataset(dataset_id)

    if dataset is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Датасет не найден")

    return DatasetResponse.model_validate(dataset)
