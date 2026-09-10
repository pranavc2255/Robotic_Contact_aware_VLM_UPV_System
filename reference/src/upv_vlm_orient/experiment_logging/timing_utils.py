from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Iterator

from upv_vlm_orient.experiment_logging.experiment_schema import NA, TIMING_KEYS


@dataclass
class TimingRecorder:
    timings: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict)

    def start(self, key: str) -> None:
        self._starts[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        start = self._starts.pop(key, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        self.timings[key] = self.timings.get(key, 0.0) + elapsed
        return elapsed

    @contextmanager
    def measure(self, key: str) -> Iterator[None]:
        self.start(key)
        try:
            yield
        finally:
            self.stop(key)

    def summary(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {}
        for key in TIMING_KEYS:
            out[key] = float(self.timings[key]) if key in self.timings else NA
            if key not in self.timings:
                out[f"{key}_NA_reason"] = "not_measured_or_not_available_from_underlying_pipeline"
        return out
