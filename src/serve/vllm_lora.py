from __future__ import annotations

from src.config import get_settings

FT_MODEL_NAME = "repomate-ft"


def adapter_exists(slug: str) -> bool:
    settings = get_settings()
    adapter_dir = settings.data_dir / slug / "adapters"
    return adapter_dir.exists() and any(
        f.suffix in (".safetensors", ".bin") for f in adapter_dir.rglob("*")
    )
