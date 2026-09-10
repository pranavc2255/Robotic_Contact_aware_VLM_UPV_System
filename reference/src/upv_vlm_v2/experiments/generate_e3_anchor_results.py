"""Generate E3 contact-anchor selection tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any


METHODS = {
    "geometry_only": "geometry_only_selected_anchor_id",
    "proposed_contact_aware": "proposed_selected_anchor_id",
}


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in cols})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_md_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _labels_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), {})[str(row["anchor_id"])] = row
    return grouped


def _manual_best(case_labels: dict[str, dict[str, Any]]) -> str | None:
    for anchor_id, row in case_labels.items():
        if str(row.get("is_manual_best")).lower() == "true" or row.get("is_manual_best") is True:
            return anchor_id
    return None


def _usable(label: str | None) -> bool:
    return str(label or "").lower() in {"good", "acceptable"}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _method_decisions(master: list[dict[str, Any]], labels: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in master:
        case_labels = labels.get(str(row.get("case_id")), {})
        best = _manual_best(case_labels)
        for method, key in METHODS.items():
            selected = str(row.get(key) or "")
            label_row = case_labels.get(selected, {})
            manual_label = label_row.get("manual_label")
            decisions.append({
                "case_index": row.get("case_index"),
                "case_id": row.get("case_id"),
                "material": row.get("material"),
                "condition_type": row.get("condition_type"),
                "method": method,
                "selected_anchor_id": selected,
                "manual_label": manual_label,
                "selected_is_usable": _usable(manual_label),
                "selected_is_manual_best": selected == best,
                "manual_best_anchor_id": best,
                "failure_reason": row.get("failure_reason"),
            })
    return decisions


def _summarize_method(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        subset = [row for row in decisions if row["method"] == method]
        n = len(subset)
        usable = sum(bool(row["selected_is_usable"]) for row in subset)
        best = sum(bool(row["selected_is_manual_best"]) for row in subset)
        not_bad = sum(str(row.get("manual_label")).lower() != "bad" for row in subset if row.get("manual_label"))
        rows.append({
            "method": method,
            "n_scenes": n,
            "top1_usable_anchor_rate": _rate(usable, n),
            "manual_best_anchor_agreement": _rate(best, n),
            "bad_anchor_avoidance_rate": _rate(not_bad, n),
        })
    return rows


def _summarize_condition(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    conditions = sorted({row.get("condition_type") for row in decisions if row.get("condition_type")})
    for condition in conditions:
        for method in METHODS:
            subset = [row for row in decisions if row.get("condition_type") == condition and row["method"] == method]
            n = len(subset)
            usable = sum(bool(row["selected_is_usable"]) for row in subset)
            rows.append({"condition_type": condition, "method": method, "n_scenes": n, "top1_usable_anchor_rate": _rate(usable, n)})
    return rows


def _failure_breakdown(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for row in decisions:
        if str(row.get("manual_label")).lower() != "bad":
            continue
        condition = str(row.get("condition_type") or "")
        if "chipped" in condition:
            category = "chipped_jagged_selected"
        elif "debris" in condition:
            category = "debris_obstruction_selected"
        else:
            category = "other_bad_selection"
        key = (row["method"], category)
        counts[key] = counts.get(key, 0) + 1
    return [{"method": method, "failure_type": category, "count": count} for (method, category), count in sorted(counts.items())]


def _dataset_protocol(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in master:
        rows.append({
            "case_index": row.get("case_index"),
            "case_id": row.get("case_id"),
            "material": row.get("material"),
            "condition_type": row.get("condition_type"),
            "condition_notes": row.get("condition_notes"),
        })
    return rows


def _all_anchor_labels(label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return label_rows


def _corrections(decisions: list[dict[str, Any]]) -> list[str]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in decisions:
        by_case.setdefault(str(row["case_id"]), {})[row["method"]] = row
    out = []
    for case_id, methods in by_case.items():
        geom = methods.get("geometry_only", {})
        proposed = methods.get("proposed_contact_aware", {})
        if str(geom.get("manual_label")).lower() == "bad" and bool(proposed.get("selected_is_usable")):
            out.append(case_id)
    return sorted(out)


def _figures(out_dir: Path, method_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], session_dir: Path, decisions: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = [row["method"] for row in method_rows]
    metrics = ["top1_usable_anchor_rate", "manual_best_anchor_agreement", "bad_anchor_avoidance_rate"]
    x = range(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 4))
    for idx, row in enumerate(method_rows):
        vals = [float(row.get(metric) or 0.0) for metric in metrics]
        plt.bar([p + idx * width for p in x], vals, width=width, label=row["method"])
    plt.xticks([p + width / 2 for p in x], ["usable", "best agreement", "bad avoidance"], rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_e3_method_comparison_bar.png", dpi=180)
    plt.close()

    conditions = sorted({row["condition_type"] for row in condition_rows})
    plt.figure(figsize=(8, 4))
    for idx, method in enumerate(methods):
        vals = [float(next((row.get("top1_usable_anchor_rate") for row in condition_rows if row["condition_type"] == condition and row["method"] == method), 0.0) or 0.0) for condition in conditions]
        plt.bar([p + idx * width for p in range(len(conditions))], vals, width=width, label=method)
    plt.xticks([p + width / 2 for p in range(len(conditions))], conditions, rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_e3_condition_wise_usable_rate.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(methods, [float(row.get("manual_best_anchor_agreement") or 0.0) for row in method_rows])
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_e3_best_anchor_agreement_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    labels = [f"{row['method']}\n{row['failure_type']}" for row in failure_rows] or ["none"]
    values = [int(row["count"]) for row in failure_rows] or [0]
    plt.bar(labels, values)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_e3_failure_breakdown.png", dpi=180)
    plt.close()

    _example_panel(fig_dir / "fig_e3_example_imperfection_avoidance_panel.png", session_dir, decisions)


def _example_panel(output: Path, session_dir: Path, decisions: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    correction_ids = _corrections(decisions)
    case_id = correction_ids[0] if correction_ids else (decisions[0]["case_id"] if decisions else None)
    master_rows = _read_rows(session_dir / "master_anchor_results.csv")
    row = next((item for item in master_rows if item.get("case_id") == case_id), {})
    keys = [
        ("rgb_path", "raw RGB"),
        ("wide_contact_grid_path", "candidate grid"),
        ("geometry_only_selected_anchor_overlay_path", "geometry-only"),
        ("selected_anchor_overlay_path", "proposed"),
    ]
    panels = []
    for key, label in keys:
        path = row.get(key)
        if path and Path(path).exists():
            img = Image.open(path).convert("RGB").resize((300, 220))
        else:
            img = Image.new("RGB", (300, 220), (238, 238, 238))
            ImageDraw.Draw(img).text((20, 95), label, fill=(30, 30, 30))
        panels.append((img, label))
    canvas = Image.new("RGB", (600, 500), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(panels):
        x = (idx % 2) * 300
        y = (idx // 2) * 250 + 25
        draw.text((x + 8, y - 20), label, fill=(0, 0, 0))
        canvas.paste(img, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def generate_results(*, session: str | Path) -> Path:
    session_dir = Path(session)
    out_dir = session_dir / "paper_ready_e3_anchor_results"
    table_dir = out_dir / "tables"
    master = _read_rows(session_dir / "master_anchor_results.csv")
    labels = _read_rows(session_dir / "manual_labels" / "manual_anchor_labels.csv")
    label_map = _labels_by_case(labels)
    decisions = _method_decisions(master, label_map)
    method_rows = _summarize_method(decisions)
    condition_rows = _summarize_condition(decisions)
    failure_rows = _failure_breakdown(decisions)
    correction_ids = _corrections(decisions)

    protocol_rows = _dataset_protocol(master)
    tables = {
        "table_e3_dataset_protocol": protocol_rows,
        "table_e3_method_summary": method_rows,
        "table_e3_condition_wise_results": condition_rows,
        "table_e3_failure_breakdown": failure_rows,
        "table_e3_all_anchor_labels": _all_anchor_labels(labels),
        "table_e3_all_scene_decisions": decisions,
    }
    for name, rows in tables.items():
        _write_rows(table_dir / f"{name}.csv", rows)
        _write_md_table(table_dir / f"{name}.md", rows)
    summary = {
        "total_scenes": len(master),
        "total_anchor_candidates_labeled": len(labels),
        "method_summaries": method_rows,
        "condition_summaries": condition_rows,
        "correction_count": len(correction_ids),
        "correction_case_ids": correction_ids,
        "failure_case_ids_by_method": {
            method: [row["case_id"] for row in decisions if row["method"] == method and str(row.get("manual_label")).lower() == "bad"]
            for method in METHODS
        },
        "imperfect_only_condition_types": ["chipped_jagged", "debris_obstruction"],
    }
    _write_json(out_dir / "summary_e3_anchor_results.json", summary)
    _figures(out_dir, method_rows, condition_rows, failure_rows, session_dir, decisions)
    example_dir = out_dir / "example_panels"
    example_dir.mkdir(parents=True, exist_ok=True)
    for path in (out_dir / "figures").glob("fig_e3_example_imperfection_avoidance_panel.png"):
        shutil.copy2(path, example_dir / path.name)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready E3 anchor-selection results.")
    parser.add_argument("--session", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = generate_results(session=args.session)
    print(f"e3_results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
