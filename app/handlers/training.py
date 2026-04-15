from typing import Annotated
from uuid import UUID

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_404_NOT_FOUND

from app.container import Container
from app.schemas.responses import JobResponse
from app.services.training import TrainingService

jobs_router = APIRouter(prefix="/api/jobs", tags=["Обучение"])


@jobs_router.get(
    path="/{job_id}",
    summary="Статус задачи обучения",
)
@inject
async def get_job(
    job_id: UUID,
    service: Annotated[TrainingService, Depends(Provide[Container.training_service])],
) -> JobResponse:
    job = await service.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Задача не найдена")

    return JobResponse.model_validate(job)
