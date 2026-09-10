from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from upv_vlm_v2.logging.artifact_writer import ArtifactWriter


CALIBRATION_KEYS = [
    ("tool", "camera_to_tcp_transform_file"),
    ("tool", "upv_tool_geometry_file"),
    ("tool", "robot_base_file"),
    ("planning", "workspace_limits_file"),
]


PERCEPTION_ONLY_MODES = {
    "dry_run",
    "snapshot_only",
    "target_only",
    "geometry_only",
    "anchor_only",
    "path_length_only",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_calibration_paths(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for section, key in CALIBRATION_KEYS:
        value = (config.get(section) or {}).get(key)
        if value:
            records.append({"section": section, "key": key, "path": str(value)})
    return records


def validate_calibration_files(config: dict[str, Any], *, required: bool) -> tuple[list[dict[str, Any]], list[str]]:
    records = collect_calibration_paths(config)
    warnings: list[str] = []
    for record in records:
        path = Path(record["path"])
        record["exists"] = path.exists()
        record["sha256"] = sha256_file(path) if path.exists() else None
        if required and not path.exists():
            raise FileNotFoundError(f"Missing required calibration file: {path}")
        if not path.exists():
            warnings.append(f"Missing calibration file allowed for current mode/backend: {path}")
    return records, warnings


def requires_robot_calibration(
    *,
    mode: str,
    execution_backend: str | None = None,
    simulation_enabled: bool = False,
) -> bool:
    """Return whether robot/tool calibration is required for this run.

    Perception and local path-length modes do not need robot/tool transforms.
    `plan_only` can use an explicit simulation fallback. Real execute always
    requires robot/tool calibration, even if planning.simulation_enabled is set.
    """
    normalized_mode = str(mode).strip().lower()
    backend = str(execution_backend or "none").strip().lower()
    if backend == "real":
        return normalized_mode in {"plan_only", "execute"}
    if normalized_mode in PERCEPTION_ONLY_MODES:
        return False
    if normalized_mode == "plan_only" and simulation_enabled:
        return False
    if normalized_mode == "execute" and backend == "simulated":
        return False
    return normalized_mode in {"plan_only", "execute"}


def write_calibration_manifest(writer: ArtifactWriter, config: dict[str, Any], *, required: bool) -> tuple[Path, list[str]]:
    records, warnings = validate_calibration_files(config, required=required)
    payload = {
        "source_files_modified": False,
        "calibration_files": records,
        "warnings": warnings,
        "note": "v2 records paths/checksums only and never overwrites source calibration files.",
    }
    return writer.write_json("calibration_manifest.json", payload), warnings
