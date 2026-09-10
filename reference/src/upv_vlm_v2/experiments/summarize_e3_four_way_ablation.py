#!/usr/bin/env python3
"""Summarize E3 four-way anchor-selection ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def selected_distribution_from_csv(csv_path: Path, selected_col: str = "selected_anchor") -> dict[str, int]:
    if not csv_path.exists():
        return {}
    out: dict[str, int] = {}
    for row in read_csv(csv_path):
        anchor = str(row.get(selected_col, "")).strip() or "UNKNOWN"
        out[anchor] = out.get(anchor, 0) + 1
    return out


def _row(
    *,
    method: str,
    input_representation: str,
    decision_mode: str,
    session: Path,
    n_cases: Any,
    parse_success_rate: Any,
    top1_contact_usable_rate: Any,
    contact_unsuitable_selected_rate: Any,
    selected_anchor_distribution: dict[str, int],
) -> dict[str, Any]:
    return {
        "method": method,
        "input_representation": input_representation,
        "decision_mode": decision_mode,
        "n_cases": n_cases,
        "parse_success_rate": parse_success_rate,
        "top1_contact_usable_rate": top1_contact_usable_rate,
        "contact_unsuitable_selected_rate": contact_unsuitable_selected_rate,
        "selected_anchor_distribution": selected_anchor_distribution,
        "session": str(session),
    }


def summarize_direct(session: Path) -> dict[str, Any]:
    summary = read_json(session / "summary_overall.json")
    return _row(
        method="Full-object direct-choice baseline",
        input_representation="full RGB/mask geometry overview with A1-A5 markers",
        decision_mode="choose one anchor directly",
        session=session,
        n_cases=summary.get("n_cases"),
        parse_success_rate=summary.get("parse_success_rate"),
        top1_contact_usable_rate=summary.get("top1_contact_usable_rate"),
        contact_unsuitable_selected_rate=summary.get("contact_unsuitable_selected_rate"),
        selected_anchor_distribution=selected_distribution_from_csv(session / "master_full_overlay_baseline_results.csv"),
    )


def summarize_per_anchor(session: Path) -> dict[str, Any]:
    summary = read_json(session / "summary_overall.json")
    dist = summary.get("selected_anchor_distribution")
    if not isinstance(dist, dict):
        dist = selected_distribution_from_csv(session / "master_full_overlay_per_anchor_scoring_results.csv")
    return _row(
        method="Full-object per-anchor scoring baseline",
        input_representation="full RGB/mask geometry overview with A1-A5 markers",
        decision_mode="score A1-A5 independently from full overview",
        session=session,
        n_cases=summary.get("n_cases"),
        parse_success_rate=summary.get("anchor_parse_success_rate"),
        top1_contact_usable_rate=summary.get("top1_contact_usable_rate"),
        contact_unsuitable_selected_rate=summary.get("contact_unsuitable_selected_rate"),
        selected_anchor_distribution={str(k): int(v) for k, v in dist.items()},
    )


def summarize_grid(session: Path) -> dict[str, Any]:
    summary = read_json(session / "summary_overall.json")
    dist = summary.get("selected_anchor_distribution")
    if not isinstance(dist, dict):
        dist = selected_distribution_from_csv(session / "master_contact_crop_grid_baseline_results.csv")
    return _row(
        method="Contact-crop grid baseline",
        input_representation="single image containing all A1-A5 contact-crop tiles",
        decision_mode="choose one anchor from all crop tiles in one Qwen call",
        session=session,
        n_cases=summary.get("n_cases"),
        parse_success_rate=summary.get("parse_success_rate"),
        top1_contact_usable_rate=summary.get("top1_contact_usable_rate"),
        contact_unsuitable_selected_rate=summary.get("contact_unsuitable_selected_rate"),
        selected_anchor_distribution={str(k): int(v) for k, v in dist.items()},
    )


def summarize_proposed(session: Path) -> dict[str, Any]:
    summary = read_json(session / "summary_overall.json")
    return _row(
        method="Proposed contact-crop scoring",
        input_representation="single-anchor top/bottom contact crops",
        decision_mode="score A1-A5 independently from local contact crops",
        session=session,
        n_cases=summary.get("n_cases"),
        parse_success_rate=summary.get("anchor_parse_success_rate"),
        top1_contact_usable_rate=summary.get("top1_usable_rate"),
        contact_unsuitable_selected_rate=summary.get("bad_anchor_selected_rate"),
        selected_anchor_distribution=selected_distribution_from_csv(session / "master_single_anchor_results.csv"),
    )


def run(args: argparse.Namespace) -> Path:
    sessions = [
        args.direct_baseline_session,
        args.per_anchor_baseline_session,
        args.grid_baseline_session,
        args.proposed_session,
    ]
    for session in sessions:
        if not session.exists():
            raise FileNotFoundError(session)

    out_dir = args.out_dir or (args.grid_baseline_session / "four_way_ablation")
    rows = [
        summarize_direct(args.direct_baseline_session),
        summarize_per_anchor(args.per_anchor_baseline_session),
        summarize_grid(args.grid_baseline_session),
        summarize_proposed(args.proposed_session),
    ]
    fields = [
        "method",
        "input_representation",
        "decision_mode",
        "n_cases",
        "parse_success_rate",
        "top1_contact_usable_rate",
        "contact_unsuitable_selected_rate",
        "selected_anchor_distribution",
        "session",
    ]
    csv_rows = []
    for row in rows:
        rr = dict(row)
        rr["selected_anchor_distribution"] = json.dumps(rr["selected_anchor_distribution"], sort_keys=True)
        csv_rows.append(rr)
    table_path = out_dir / "four_way_comparison_table.csv"
    write_csv(table_path, csv_rows, fields)
    write_json(out_dir / "four_way_comparison_summary.json", {"rows": rows, "table": str(table_path)})
    print(table_path)
    print(",".join(fields))
    for row in csv_rows:
        print(",".join(str(row.get(field, "")) for field in fields))
    return table_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct-baseline-session", required=True, type=Path)
    p.add_argument("--per-anchor-baseline-session", required=True, type=Path)
    p.add_argument("--grid-baseline-session", required=True, type=Path)
    p.add_argument("--proposed-session", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
