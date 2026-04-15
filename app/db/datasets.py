from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel, col, select

from app.consts import utc_now_naive
from app.db.base import BaseRepo


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    original_filename: str = Field(max_length=512)
    stored_path: str = Field(max_length=2048)
    size_bytes: int = Field(ge=0)
    content_type: str | None = Field(default=None, max_length=128)
    created_at: datetime = Field(
        default_factory=utc_now_naive,
        sa_column_kwargs={"nullable": False},
    )


class DatasetsRepo(BaseRepo):
    async def create(
        self,
        *,
        dataset_id: UUID | None = None,
        original_filename: str,
        stored_path: str,
        size_bytes: int,
        content_type: str | None,
    ) -> Dataset:
        dataset = Dataset(
            id=dataset_id or uuid4(),
            original_filename=original_filename,
            stored_path=stored_path,
            size_bytes=size_bytes,
            content_type=content_type,
        )

        async with self._session() as session:
            session.add(dataset)
            await session.flush()
            await session.refresh(dataset)

        return dataset

    async def list_all(self) -> list[Dataset]:
        async with self._session() as session:
            result = await session.exec(select(Dataset).order_by(col(Dataset.created_at).desc()))

            return list(result.all())

    async def get_by_id(self, dataset_id: UUID) -> Dataset | None:
        async with self._session() as session:
            return await session.get(Dataset, dataset_id)
