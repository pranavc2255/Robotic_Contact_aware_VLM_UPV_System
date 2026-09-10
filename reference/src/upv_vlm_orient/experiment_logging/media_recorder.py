from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
from typing import Any


@dataclass
class VideoRecorder:
    output_path: Path
    fps: float = 10.0
    dry_run: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    def start(self) -> None:
        self.events.append({"event": "video_start", "path": str(self.output_path), "fps": self.fps, "dry_run": self.dry_run, "timestamp": time.time()})

    def stop(self) -> None:
        self.events.append({"event": "video_stop", "path": str(self.output_path), "dry_run": self.dry_run, "timestamp": time.time()})


@dataclass
class RosbagRecorder:
    output_dir: Path
    topics: list[str]
    max_duration_s: int = 180
    dry_run: bool = True
    process: subprocess.Popen[Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events.append(
            {
                "event": "rosbag_start",
                "output_dir": str(self.output_dir),
                "topics": self.topics,
                "max_duration_s": self.max_duration_s,
                "dry_run": self.dry_run,
                "timestamp": time.time(),
            }
        )
        if self.dry_run:
            return
        cmd = ["ros2", "bag", "record", "-o", str(self.output_dir), *self.topics]
        self.process = subprocess.Popen(cmd)

    def stop(self) -> None:
        self.events.append({"event": "rosbag_stop", "output_dir": str(self.output_dir), "dry_run": self.dry_run, "timestamp": time.time()})
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
