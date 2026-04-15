# BigData MVP

Веб-приложение для загрузки CSV-файлов, обучения модели (pandas + scikit-learn) и просмотра метрик и графиков в браузере.

## Требования

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Docker / Docker Compose (для PostgreSQL)

## Быстрый старт

```bash
cp .env.example .env
docker compose up -d --wait postgres
uv sync --all-groups
uv run alembic upgrade head
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

Или одной командой (поднимает Postgres и API):

```bash
make run
```

Откройте в браузере: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — веб-интерфейс.

Документация API (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## PostgreSQL в Docker

Сервис описан в [docker-compose.yaml](docker-compose.yaml): образ `postgres:17-alpine`, порт **5432**, БД и пользователь **bigdata** / пароль **bigdata** (только для локальной разработки).

Параметры подключения задаются переменными `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (см. [.env.example](.env.example)). Приложение на хосте подключается к `localhost:5432`.

### Postgres не стартует: `No space left on device`

Сообщение `initdb: ... No space left on device` значит, что **закончилось место** на диске образа Docker или на диске Mac. См. `docker system df`, `docker system prune -a --volumes`, настройки **Disk image size** в Docker Desktop, затем `docker compose down -v && docker compose up -d --wait postgres`.

## Миграции (Alembic)

Схема БД задаётся миграциями в [migrations/versions/](migrations/versions/), конфигурация — [alembic.ini](alembic.ini) и [migrations/env.py](migrations/env.py) (async, тот же DSN, что у приложения).

```bash
# применить все миграции
make migrate
# или: uv run alembic upgrade head
```

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `DB_HOST` | Хост PostgreSQL (для Docker на хосте — `localhost`) |
| `DB_PORT` | Порт (по умолчанию `5432`) |
| `DB_NAME` | Имя базы |
| `DB_USER` / `DB_PASSWORD` | Учётные данные |
| `STORAGE_DIR` | Каталог для загруженных файлов и артефактов моделей |
| `MAX_UPLOAD_SIZE_MB` | Лимит размера загрузки |

## Сценарий использования (UI)

1. Загрузите файл **CSV** (с заголовком) с числовыми признаками и колонкой таргета.
2. По умолчанию таргет — колонка `label`. При необходимости укажите другое имя в поле «Колонка таргета».
3. Нажмите «Запустить обучение», затем «Обновить» для просмотра статуса и метрик.
4. После статуса `done`: для классификации — матрица ошибок и корреляции; для регрессии — график «факт vs предсказание», столбчатая диаграмма RMSE/MAE/R², корреляции и важность признаков RandomForest.

Тип задачи (**классификация** или **регрессия**) выбирается автоматически по виду числового таргета.

## API (кратко)

- `POST /api/datasets/upload` — multipart-загрузка файла.
- `GET /api/datasets/` — список датасетов.
- `GET /api/datasets/{dataset_id}` — метаданные датасета.
- `POST /api/datasets/{dataset_id}/train?target_column=label` — постановка задачи обучения (выполняется в фоне).
- `GET /api/jobs/{job_id}` — статус задачи, метрики и агрегаты для графиков.

## Разработка

```bash
make lint     # ruff + mypy (включая migrations/)
make format   # ruff --fix
make migrate  # alembic upgrade head
make test     # Postgres, миграции, pytest
```
