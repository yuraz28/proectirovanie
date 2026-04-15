TARGET_DIRS := app migrations

lint:
	@uv run ruff check $(TARGET_DIRS)
	@uv run mypy --strict $(TARGET_DIRS)

format:
	@uv run ruff check --fix $(TARGET_DIRS)

migrate:
	@uv run alembic upgrade head

test:
	@docker compose up -d --wait postgres
	@uv run alembic upgrade head
	@uv run pytest --cov

run:
	@docker compose up -d --wait postgres
	@uv run alembic upgrade head
	@uv run uvicorn --host 0.0.0.0 --port 8000 --no-access-log app.main:app
