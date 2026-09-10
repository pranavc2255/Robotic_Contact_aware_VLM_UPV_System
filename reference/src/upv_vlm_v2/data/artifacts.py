from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactManifest:
    session_dir: str
    config_snapshot: str | None = None
    calibration_manifest: str | None = None
    stage_dirs: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_dir": self.session_dir,
            "config_snapshot": self.config_snapshot,
            "calibration_manifest": self.calibration_manifest,
            "stage_dirs": self.stage_dirs,
            "files": self.files,
            "diagnostics": self.diagnostics,
        }

