from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from upv_vlm_v2.config.loader import snapshot_yaml
from upv_vlm_v2.config.defaults import DEFAULT_STAGE_DIRS
from upv_vlm_v2.data.models import json_ready


class ArtifactWriter:
    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "artifacts").mkdir(exist_ok=True)
        (self.session_dir / "tables").mkdir(exist_ok=True)

    @staticmethod
    def make_session_dir(output_root: str | Path) -> Path:
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("session_%Y%m%d_%H%M%S")
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        idx = 2
        candidate = root / stamp
        while True:
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                candidate = root / f"{stamp}_{idx:02d}"
                idx += 1

    def stage_dir(self, stage_name: str) -> Path:
        rel = DEFAULT_STAGE_DIRS.get(stage_name, stage_name)
        path = self.session_dir / "artifacts" / rel
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_all_stage_dirs(self) -> dict[str, str]:
        return {name: str(self.stage_dir(name)) for name in DEFAULT_STAGE_DIRS}

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        path = self.session_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
        return path

    def write_stage_json(self, stage_name: str, filename: str, payload: Any) -> Path:
        path = self.stage_dir(stage_name) / filename
        path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
        return path

    def write_config_snapshot(self, config: dict[str, Any]) -> Path:
        return snapshot_yaml(config, self.session_dir / "config_snapshot.yaml")

    def relative(self, path: str | Path | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        try:
            return str(p.relative_to(self.session_dir))
        except ValueError:
            return str(p)
