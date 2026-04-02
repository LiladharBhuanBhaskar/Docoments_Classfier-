from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import process_document, process_pdf

DEFAULT_OLLAMA_MODEL = "mistral:latest"
DEFAULT_CHAT_PROMPT_TEMPLATE = """You are a helpful assistant.

User message:
{message}
"""

logger = logging.getLogger(__name__)

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_SUPPORTED_UPLOAD_SUFFIXES = _SUPPORTED_IMAGE_SUFFIXES | {".pdf"}
_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Document OCR + Parser API")

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model: str = Field(default=DEFAULT_OLLAMA_MODEL)
    prompt_template: str = Field(default=DEFAULT_CHAT_PROMPT_TEMPLATE)
    system_prompt: str | None = None


class DocumentQuestionRequest(BaseModel):
    message: str = Field(..., min_length=1)
    document_json: dict[str, Any]
    model: str = Field(default=DEFAULT_OLLAMA_MODEL)


def _require_ollama_helpers() -> tuple[Any, Any]:
    try:
        from parser.ollama import answer_question_from_document, chat_with_ollama
    except Exception as exc:
        raise RuntimeError(
            "Ollama integration is unavailable in this environment. Install the 'ollama' Python package to enable chat."
        ) from exc
    return chat_with_ollama, answer_question_from_document


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def landing() -> FileResponse:
    if not _STATIC_DIR.exists():
        raise HTTPException(status_code=500, detail="Static directory is missing")
    return FileResponse(_STATIC_DIR / "landing.html")


@app.get("/app")
def app_ui() -> FileResponse:
    if not _STATIC_DIR.exists():
        raise HTTPException(status_code=500, detail="Static directory is missing")
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/chat")
async def chat(payload: ChatRequest) -> dict[str, str]:
    try:
        chat_with_ollama, _ = _require_ollama_helpers()
        return await run_in_threadpool(
            chat_with_ollama,
            payload.message,
            model=payload.model,
            prompt_template=payload.prompt_template,
            system_prompt=payload.system_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Chat failed") from exc


@app.post("/document-chat")
async def document_chat(payload: DocumentQuestionRequest) -> dict[str, str]:
    try:
        _, answer_question_from_document = _require_ollama_helpers()
        return await run_in_threadpool(
            answer_question_from_document,
            payload.message,
            payload.document_json,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document chat failed") from exc


@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    doc_type: str | None = Form(default=None),
) -> dict[str, Any]:
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix or '(no extension)'}",
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty upload")

        Path(tmp_path).write_bytes(content)

        if suffix == ".pdf":
            result = await run_in_threadpool(process_pdf, tmp_path, doc_type)
        else:
            result = await run_in_threadpool(process_document, tmp_path, doc_type)

        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail="Processing failed") from exc
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
