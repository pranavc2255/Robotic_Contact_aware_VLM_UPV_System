from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_repo_path(repo_root: str | Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def load_json_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
