#!/usr/bin/env python3
"""
Summarize E3 three-way ablation:

1. Full-object direct-choice baseline
2. Full-object per-anchor scoring baseline
3. Proposed contact-crop scoring

This script only reads completed output sessions and writes comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def pct(x: float) -> float:
    return round(100.0 * float(x), 3)


def selected_distribution_from_csv(csv_path: Path, selected_col: str = "selected_anchor") -> dict[str, int]:
    if not csv_path.exists():
        return {}
    rows = read_csv(csv_path)
    out: dict[str, int] = {}
    for r in rows:
        a = str(r.get(selected_col, "")).strip() or "UNKNOWN"
        out[a] = out.get(a, 0) + 1
    return out


def summarize_direct_baseline(session: Path) -> dict[str, Any]:
    s = read_json(session / "summary_overall.json")
    dist = selected_distribution_from_csv(session / "master_full_overlay_baseline_results.csv")

    return {
        "method": "Full-object direct-choice baseline",
        "input_representation": "Full RGB/mask overview with A1-A5 markers",
        "decision_mode": "choose one anchor directly",
        "session": str(session),
        "n_cases": s.get("n_cases", 0),
        "parse_success_rate": s.get("parse_success_rate", 0.0),
        "parse_success_percent": pct(s.get("parse_success_rate", 0.0)),
        "top1_contact_usable_rate": s.get("top1_contact_usable_rate", 0.0),
        "top1_contact_usable_percent": pct(s.get("top1_contact_usable_rate", 0.0)),
        "contact_unsuitable_selected_rate": s.get("contact_unsuitable_selected_rate", 0.0),
        "contact_unsuitable_selected_percent": pct(s.get("contact_unsuitable_selected_rate", 0.0)),
        "selected_anchor_distribution": dist,
    }


def summarize_per_anchor_baseline(session: Path) -> dict[str, Any]:
    s = read_json(session / "summary_overall.json")
    dist = s.get("selected_anchor_distribution")
    if not isinstance(dist, dict):
        dist = selected_distribution_from_csv(session / "master_full_overlay_per_anchor_scoring_results.csv")

    return {
        "method": "Full-object per-anchor scoring baseline",
        "input_representation": "Full RGB/mask overview with A1-A5 markers",
        "decision_mode": "score A1-A5 independently from full overview",
        "session": str(session),
        "n_cases": s.get("n_cases", 0),
        "parse_success_rate": s.get("anchor_parse_success_rate", 0.0),
        "parse_success_percent": pct(s.get("anchor_parse_success_rate", 0.0)),
        "top1_contact_usable_rate": s.get("top1_contact_usable_rate", 0.0),
        "top1_contact_usable_percent": pct(s.get("top1_contact_usable_rate", 0.0)),
        "contact_unsuitable_selected_rate": s.get("contact_unsuitable_selected_rate", 0.0),
        "contact_unsuitable_selected_percent": pct(s.get("contact_unsuitable_selected_rate", 0.0)),
        "selected_anchor_distribution": dist,
    }


def summarize_proposed(session: Path) -> dict[str, Any]:
    s = read_json(session / "summary_overall.json")
    dist = selected_distribution_from_csv(session / "master_single_anchor_results.csv")

    return {
        "method": "Proposed contact-crop scoring",
        "input_representation": "Single-anchor paired top/bottom contact crops",
        "decision_mode": "score A1-A5 independently from local contact crops",
        "session": str(session),
        "n_cases": s.get("n_cases", 0),
        "parse_success_rate": s.get("anchor_parse_success_rate", 0.0),
        "parse_success_percent": pct(s.get("anchor_parse_success_rate", 0.0)),
        "top1_contact_usable_rate": s.get("top1_usable_rate", 0.0),
        "top1_contact_usable_percent": pct(s.get("top1_usable_rate", 0.0)),
        "contact_unsuitable_selected_rate": s.get("bad_anchor_selected_rate", 0.0),
        "contact_unsuitable_selected_percent": pct(s.get("bad_anchor_selected_rate", 0.0)),
        "selected_anchor_distribution": dist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-baseline-session", required=True, type=Path)
    ap.add_argument("--per-anchor-baseline-session", required=True, type=Path)
    ap.add_argument("--proposed-session", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    for p in [args.direct_baseline_session, args.per_anchor_baseline_session, args.proposed_session]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir = args.out_dir or (args.per_anchor_baseline_session / "three_way_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        summarize_direct_baseline(args.direct_baseline_session),
        summarize_per_anchor_baseline(args.per_anchor_baseline_session),
        summarize_proposed(args.proposed_session),
    ]

    fields = [
        "method",
        "input_representation",
        "decision_mode",
        "n_cases",
        "parse_success_rate",
        "parse_success_percent",
        "top1_contact_usable_rate",
        "top1_contact_usable_percent",
        "contact_unsuitable_selected_rate",
        "contact_unsuitable_selected_percent",
        "selected_anchor_distribution",
        "session",
    ]

    csv_rows = []
    for r in rows:
        rr = dict(r)
        rr["selected_anchor_distribution"] = json.dumps(rr["selected_anchor_distribution"], sort_keys=True)
        csv_rows.append(rr)

    write_csv(out_dir / "three_way_comparison_table.csv", csv_rows, fields)
    write_json(out_dir / "three_way_comparison_summary.json", {"rows": rows})

    print("Wrote:")
    print(out_dir / "three_way_comparison_table.csv")
    print(out_dir / "three_way_comparison_summary.json")
    print()
    print("Three-way comparison:")
    for r in rows:
        print(
            f"- {r['method']}: "
            f"parse={r['parse_success_percent']}%, "
            f"usable={r['top1_contact_usable_percent']}%, "
            f"unsuitable={r['contact_unsuitable_selected_percent']}%, "
            f"dist={r['selected_anchor_distribution']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
