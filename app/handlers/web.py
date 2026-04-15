from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

web_router = APIRouter(tags=["Интерфейс"])


@web_router.get(path="/", summary="Веб-интерфейс", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={})
