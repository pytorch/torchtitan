from __future__ import annotations

import os
from pathlib import Path

_CACHE_ENV_VARS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
_CACHE_MARKERS = ("models--", "datasets--", "spaces--", "blobs", "refs")


def is_hf_offline() -> bool:
    for key in _OFFLINE_ENV_VARS:
        val = os.environ.get(key)
        if val is None:
            continue
        if str(val).lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def resolve_hf_cache_dir() -> str | None:
    candidates: list[Path] = []
    for key in _CACHE_ENV_VARS:
        val = os.environ.get(key)
        if val:
            candidates.append(Path(val))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")

    # Default HF cache location
    try:
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    except Exception:
        pass

    repo_root = _find_repo_root()
    if repo_root is not None:
        candidates.append(repo_root / "assets" / "hf" / "hub")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            path = path.expanduser()
        except Exception:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    for path in unique:
        if _has_cache_contents(path):
            return str(path)

    if not is_hf_offline():
        for path in unique:
            if path.exists():
                return str(path)

    return str(unique[0]) if unique else None


def _has_cache_contents(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for marker in _CACHE_MARKERS:
        if (path / marker).exists():
            return True
    return False


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "torchtitan").exists():
            return parent
    return None
