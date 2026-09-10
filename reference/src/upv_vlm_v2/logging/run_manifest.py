from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from upv_vlm_v2.logging.artifact_writer import ArtifactWriter


def write_run_manifest(writer: ArtifactWriter, *, mode: str, requested_material: str, axis_mode: str, config_path: str | None) -> Path:
    payload: dict[str, Any] = {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "mode": mode,
        "requested_material": requested_material,
        "axis_mode": axis_mode,
        "config_path": config_path,
        "session_dir": str(writer.session_dir),
        "stage_dirs": writer.ensure_all_stage_dirs(),
    }
    return writer.write_json("run_manifest.json", payload)

