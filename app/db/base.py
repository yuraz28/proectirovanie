from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession


class BaseRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self._session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        session = self._session_factory()

        try:
            yield session
        except SQLAlchemyError:
            await session.rollback()

            raise
        else:
            await session.commit()
        finally:
            await session.close()
