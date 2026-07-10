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

DROP_PATTERNS = frozenset({
    "*.min.js", "*.min.css", "*.lock", "package-lock.json", "yarn.lock",
    "go.sum", "Cargo.lock", "Pipfile.lock", "poetry.lock", "Gemfile.lock",
})

BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".so", ".dll", ".dylib", ".exe", ".bin", ".parquet", ".npy",
    ".npz", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".onnx",
    ".pt", ".pth", ".ckpt", ".safetensors", ".whl", ".egg", ".o", ".a",
    ".mp3", ".mp4", ".avi", ".wav", ".mov", ".mkv", ".flac",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
})

MAX_FILE_SIZE_BYTES = 256 * 1024
MAX_COMMITS = 2000
MAX_FILES = 3000
DIFF_MAX_TOKENS = 2048


class Settings(BaseSettings):
    fireworks_api_key: str
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_synth_model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    gemma_model_id: str = "google/gemma-2-9b-it"
    bge_model_id: str = "BAAI/bge-m3"
    amd_device: str = "cuda:0"
    data_dir: Path = Path("./data")
    vllm_port: int = 8000
    api_port: int = 8080
    ui_port: int = 8501
    chunk_max_tokens: int = 1024
    chunk_overlap: int = 128
    synth_pairs_per_chunk: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CIYC_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
