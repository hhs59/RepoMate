from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

KEEP_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hxx", ".cs", ".rb", ".php",
    ".swift", ".scala", ".md", ".vue", ".svelte", ".sql", ".yaml", ".yml",
    ".toml", ".json", ".xml", ".css", ".html", ".sh", ".bash", ".tf",
    ".proto", ".graphql", ".sol", ".dart", ".r", ".lua", ".zig", ".nim",
})

DROP_DIRS = frozenset({
    "node_modules", ".git", "dist", "build", "out", "target", "vendor",
    "venv", ".venv", "__pycache__", ".eggs", ".tox", ".nox", ".pytest_cache",
})

MAX_FILE_SIZE_BYTES = 256 * 1024
MAX_COMMITS = 2000
MAX_FILES = 3000
DIFF_MAX_TOKENS = 2048


class Settings(BaseSettings):
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    bge_model_id: str = "BAAI/bge-small-en-v1.5"
    reranker_model_id: str = "BAAI/bge-reranker-base"
    device: str = ""
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    vllm_port: int = 8000
    api_port: int = 8080
    chunk_max_tokens: int = 1024
    chunk_overlap: int = 128
    synth_pairs_per_chunk: int = 3
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    rerank_pool_size: int = 20

    model_config = SettingsConfigDict(env_file=".env", env_prefix="REPOMATE_")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm_api_key() -> str:
    return get_settings().llm_api_key

def get_llm_base_url() -> str:
    return get_settings().llm_base_url

def get_llm_model() -> str:
    return get_settings().llm_model

def get_base_model_id() -> str:
    return get_settings().base_model_id

def get_device() -> str:
    s = get_settings()
    if s.device:
        return s.device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
