#!/usr/bin/env python3
"""Compare E3 full-overlay baseline against proposed contact-crop scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def first_number(summary: dict[str, Any], keys: list[str]) -> float | int | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except Exception:
                pass
    return None


def proposed_parse_rate(summary: dict[str, Any]) -> float | None:
    value = first_number(summary, ["parse_success_rate", "case_parse_success_rate"])
    if value is not None:
        return float(value)
    n_cases = first_number(summary, ["n_cases"])
    cases_any = first_number(summary, ["cases_with_any_parse_success"])
    if n_cases:
        return float(cases_any or 0) / float(n_cases)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_rows(baseline_summary: dict[str, Any], proposed_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method": "Full-object anchor-overlay baseline",
            "input_representation": "full RGB/mask geometry overview with A1-A5 markers",
            "n_cases": baseline_summary.get("n_cases"),
            "parse_success_rate": baseline_summary.get("parse_success_rate"),
            "top1_contact_usable_rate": baseline_summary.get("top1_contact_usable_rate"),
            "contact_unsuitable_selected_rate": baseline_summary.get("contact_unsuitable_selected_rate"),
        },
        {
            "method": "Proposed contact-crop scoring",
            "input_representation": "single-anchor top/bottom contact crops",
            "n_cases": proposed_summary.get("n_cases"),
            "parse_success_rate": proposed_parse_rate(proposed_summary),
            "top1_contact_usable_rate": first_number(
                proposed_summary, ["top1_contact_usable_rate", "top1_usable_rate"]
            ),
            "contact_unsuitable_selected_rate": first_number(
                proposed_summary, ["contact_unsuitable_selected_rate", "bad_anchor_selected_rate"]
            ),
        },
    ]


def run(args: argparse.Namespace) -> Path:
    baseline_session = Path(args.baseline_session)
    proposed_session = Path(args.proposed_session)
    baseline_summary = load_json(baseline_session / "summary_overall.json")
    proposed_summary = load_json(proposed_session / "summary_overall.json")

    rows = build_rows(baseline_summary, proposed_summary)
    fields = [
        "method",
        "input_representation",
        "n_cases",
        "parse_success_rate",
        "top1_contact_usable_rate",
        "contact_unsuitable_selected_rate",
    ]

    comparison_csv = baseline_session / "comparison_table.csv"
    write_csv(comparison_csv, rows, fields)

    comparison_json = baseline_session / "comparison_summary.json"
    comparison_json.write_text(
        json.dumps(
            {
                "baseline_session": str(baseline_session),
                "proposed_session": str(proposed_session),
                "comparison_table": str(comparison_csv),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(",".join(fields))
    for row in rows:
        print(",".join(str(row.get(f, "")) for f in fields))
    print(f"Wrote {comparison_csv}")
    return comparison_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-session", required=True)
    p.add_argument("--proposed-session", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
