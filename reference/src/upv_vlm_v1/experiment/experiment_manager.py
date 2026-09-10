from __future__ import annotations

from pathlib import Path
from typing import Any

from upv_vlm_v1.experiment.t4_runner_adapter import T4RunnerAdapter


class ExperimentManager:
    def __init__(self, repo_root: str | Path, e1_config_path: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.e1_config_path = Path(e1_config_path)
        if not self.e1_config_path.is_absolute():
            self.e1_config_path = self.repo_root / self.e1_config_path

    def make_t4_adapter(self, t4_config_path: str | Path, default_prompt_config: str | Path | None = None) -> T4RunnerAdapter:
        return T4RunnerAdapter(self.repo_root, t4_config_path, default_prompt_config=default_prompt_config)

    def create_or_resume_run(self, *args: Any, **kwargs: Any) -> Path | None:
        return kwargs.get("run_dir")

    def save_metadata(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_manual_labels(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_upv_results(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_prompt_bundle(self, *args: Any, **kwargs: Any) -> None:
        return None

    def run_perception_only(self, adapter: T4RunnerAdapter, **kwargs: Any) -> dict[str, Any]:
        return adapter.run_oneshot("perception_only", **kwargs)

    def run_plan_only(self, adapter: T4RunnerAdapter, **kwargs: Any) -> dict[str, Any]:
        return adapter.run_oneshot("plan_only", **kwargs)

    def run_clamp_only(self, adapter: T4RunnerAdapter, confirmation: str, manual_opening_mm: float | None = None) -> dict[str, Any]:
        return adapter.clamp_only(confirmation, manual_opening_mm=manual_opening_mm)

    def run_full_autonomous(self, adapter: T4RunnerAdapter, confirmation: str, **kwargs: Any) -> dict[str, Any]:
        return adapter.full_final(confirmation, **kwargs)

    def run_full_autonomous_home(self, adapter: T4RunnerAdapter, confirmation: str, **kwargs: Any) -> dict[str, Any]:
        return adapter.full_final_home(confirmation, **kwargs)

    def home_robot(self, adapter: T4RunnerAdapter, confirmation: str) -> dict[str, Any]:
        return adapter.home(confirmation)

    def emergency_release(self, adapter: T4RunnerAdapter, confirmation: str) -> dict[str, Any]:
        return adapter.emergency_release(confirmation)

    def collect_outputs(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update_metrics(self, *args: Any, **kwargs: Any) -> None:
        return None
