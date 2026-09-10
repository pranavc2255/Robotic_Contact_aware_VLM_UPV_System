#!/usr/bin/env python3
"""Write reproducibility metadata for a paper experiment session."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from upv_vlm_v2.experiments.run_all_paper_experiments import DEFAULT_SESSION_PROVENANCE


DEFAULT_PAPER_CONFIGS = [
    "configs/v2/experiments/paper/e1_target_selection.yaml",
    "configs/v2/experiments/paper/e2_geometry_path_length.yaml",
    "configs/v2/experiments/paper/e3_anchor_selection_proposed_main_pipeline.yaml",
    "configs/v2/experiments/paper/e3_baseline_contact_crop_grid.yaml",
    "configs/v2/experiments/paper/paper_experiment_manifest.yaml",
]


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def torch_snapshot() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        return {
            "torch_version": getattr(torch, "__version__", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        return {"torch_import_error": repr(exc)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def ensure_session_provenance(paper: Path, server_url: str) -> dict[str, Any]:
    path = paper / "00_run_manifest" / "session_provenance.json"
    provenance = read_json(path)
    if not provenance:
        provenance = {
            "session_paths": DEFAULT_SESSION_PROVENANCE,
            "config_paths": {
                "e1_target_selection": DEFAULT_PAPER_CONFIGS[0],
                "e2_geometry_path_length": DEFAULT_PAPER_CONFIGS[1],
                "e3_anchor_selection_proposed_main_pipeline": DEFAULT_PAPER_CONFIGS[2],
                "e3_baseline_contact_crop_grid": DEFAULT_PAPER_CONFIGS[3],
                "paper_experiment_manifest": DEFAULT_PAPER_CONFIGS[4],
            },
            "qwen_server_url": server_url,
            "created_by": "write_reproducibility_report fallback",
        }
        write_json(path, provenance)
    return provenance


def run(args: argparse.Namespace) -> None:
    paper = Path(args.paper_session)
    out = paper / "07_audit_report"
    out.mkdir(parents=True, exist_ok=True)
    provenance = ensure_session_provenance(paper, args.server_url)
    provenance_configs = list((provenance.get("config_paths") or {}).values())
    provenance_sessions = list((provenance.get("session_paths") or {}).values())
    config_paths = list(dict.fromkeys([*args.config, *provenance_configs]))
    session_paths = list(dict.fromkeys([*args.session_path, *provenance_sessions]))
    tags = _git(["tag", "--points-at", "HEAD"])
    status = _git(["status", "--short"])
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_branch": _git(["branch", "--show-current"]),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_tags_at_head": tags.splitlines() if tags else [],
        "git_status_short": status,
        "repo_dirty": bool(status.strip()),
        "qwen_server_url": args.server_url,
        "torch": torch_snapshot(),
        "config_paths": config_paths,
        "dataset_or_session_paths": session_paths,
        "session_provenance_path": str(paper / "00_run_manifest" / "session_provenance.json"),
        "session_provenance": provenance,
    }
    write_json(out / "environment_snapshot.json", env)
    inventory = subprocess.check_output(["find", str(paper), "-maxdepth", "4", "-type", "f"], text=True)
    (out / "file_inventory.txt").write_text(inventory, encoding="utf-8")
    report = [
        "# UPV_VLM_v2 Paper Experiment Reproducibility Report",
        "",
        f"- Git branch: `{env['git_branch']}`",
        f"- Git commit: `{env['git_commit']}`",
        f"- Git tags at HEAD: `{', '.join(env['git_tags_at_head']) or 'none'}`",
        f"- Repo dirty: `{env['repo_dirty']}`",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Qwen server URL: `{args.server_url}`",
        "",
        "## Warnings",
        "",
    ]
    if env["repo_dirty"]:
        report.append("- Repository has uncommitted changes at report time.")
    else:
        report.append("- No uncommitted changes reported by Git.")
    report.extend(["", "## Referenced Configs", ""])
    for cfg in config_paths:
        report.append(f"- `{cfg}`")
    report.extend(["", "## Referenced Dataset/Sessions", ""])
    for session in session_paths:
        report.append(f"- `{session}`")
    (out / "reproducibility_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-session", required=True)
    p.add_argument("--server-url", default="http://127.0.0.1:8899")
    p.add_argument("--config", action="append", default=[])
    p.add_argument("--session-path", action="append", default=[])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
