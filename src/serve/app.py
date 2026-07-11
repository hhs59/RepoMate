from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import get_settings
from src.serve.vllm_server import is_ready
from src.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="repomate API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "out"
if _frontend_dir.exists():
    app.mount("/_next", StaticFiles(directory=_frontend_dir / "_next"), name="next-static")


def _resolve_slug(provided: str | None) -> str:
    if provided:
        return provided
    import os
    env_slug = os.environ.get("REPOMATE_DEFAULT_REPO")
    if env_slug:
        return env_slug
    raise HTTPException(status_code=400, detail="No repo slug provided")


class ChatRequest(BaseModel):
    repo_slug: Optional[str] = None
    query: str
    mode: str = "rag"


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    elapsed_ms: int
    mode: str = "rag"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "vllm_ready": is_ready()}


@app.get("/repos")
def list_repos() -> dict:
    settings = get_settings()
    slugs = []
    if settings.data_dir.exists():
        for d in sorted(settings.data_dir.iterdir()):
            if d.is_dir() and (d / "chunks.jsonl").exists():
                slugs.append(d.name)
    return {"repos": slugs}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    slug = _resolve_slug(req.repo_slug)

    from src.serve.hybrid import answer as hybrid_answer

    mode = req.mode
    if mode == "hybrid":
        from src.serve.vllm_lora import adapter_exists
        if not adapter_exists(slug):
            mode = "rag"

    result = hybrid_answer(slug, req.query, mode=mode)
    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        elapsed_ms=result["elapsed_ms"],
        mode=result["mode"],
    )


@app.get("/")
def serve_frontend():
    if _frontend_dir.exists():
        return FileResponse(_frontend_dir / "index.html")
    return {"status": "ok", "message": "Frontend not built. Run: cd frontend && npm run build"}


@app.post("/train")
def train_endpoint(req: dict) -> dict:
    import subprocess
    import sys
    import os

    slug = req.get("repo_slug")
    if not slug:
        raise HTTPException(status_code=400, detail="repo_slug required")

    epochs = req.get("epochs", 1)
    max_chunks = req.get("max_chunks", 200)
    skip_data = req.get("skip_data", False)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "data", slug, "train_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.cli", "train", slug,
        "--epochs", str(epochs),
        "--max-chunks", str(max_chunks),
    ]
    if skip_data:
        cmd.append("--skip-data")

    subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    return {"status": "started", "log": log_path}


@app.get("/train/status")
def train_status(slug: str) -> dict:
    settings = get_settings()
    log_path = settings.data_dir / slug / "train_log.txt"
    adapter_dir = settings.data_dir / slug / "adapters"

    log_content = ""
    if log_path.exists():
        log_content = log_path.read_text()[-2000:]

    has_adapter = adapter_dir.exists() and any(
        f.suffix in (".safetensors", ".bin") for f in adapter_dir.rglob("*")
    )

    return {"log": log_content, "adapter_ready": has_adapter}
