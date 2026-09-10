"""Interactive manual labeling for E3 anchor candidates."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

from upv_vlm_v2.experiments.e3_anchor_crop_utils import (
    build_clean_anchor_review_grid,
    render_clean_anchor_inputs_for_artifact_dir,
)

LABEL_COLUMNS = [
    "case_index",
    "case_id",
    "material",
    "condition_type",
    "axis_mode",
    "anchor_id",
    "manual_label",
    "is_usable",
    "is_manual_best",
    "manual_reason",
    "geometry_only_selected_anchor_id",
    "proposed_selected_anchor_id",
    "label_timestamp",
]


def _timestamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def parse_anchor_id_list(text: str, candidate_ids: list[str], *, allow_empty: bool = True) -> list[str]:
    text = (text or "").strip()
    if not text:
        if allow_empty:
            return []
        raise ValueError("empty anchor list")
    valid = {item.upper(): item.upper() for item in candidate_ids}
    out: list[str] = []
    for raw in text.replace(" ", "").split(","):
        if not raw:
            continue
        value = raw.upper()
        if value.isdigit():
            value = f"A{int(value)}"
        if value not in valid:
            raise ValueError(f"invalid anchor ID: {raw}")
        if value in out:
            raise ValueError(f"duplicate anchor ID: {value}")
        out.append(value)
    return out


def validate_case_labels(
    *,
    candidate_ids: list[str],
    good: list[str],
    acceptable: list[str],
    bad: list[str],
    manual_best: str,
) -> None:
    valid = {item.upper() for item in candidate_ids}
    groups = [item.upper() for item in good + acceptable + bad]
    if len(groups) != len(set(groups)):
        raise ValueError("anchor IDs may appear in only one label group")
    missing = sorted(valid - set(groups))
    extra = sorted(set(groups) - valid)
    if missing:
        raise ValueError(f"missing labels for: {','.join(missing)}")
    if extra:
        raise ValueError(f"invalid labeled anchors: {','.join(extra)}")
    if manual_best.upper() not in valid:
        raise ValueError(f"manual best anchor must be one of: {','.join(sorted(valid))}")


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    values = [part.strip().upper() for part in str(row.get("candidate_anchor_ids") or "").split(",") if part.strip()]
    if values:
        return values
    count = int(row.get("candidate_anchor_count") or 0)
    return [f"A{i}" for i in range(1, count + 1)]


def _existing_labels(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = _read_rows(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("case_id")), []).append(row)
    return grouped


def _open_image(path: str | None) -> None:
    if not path or not Path(path).exists():
        return
    try:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _case_dir(session: Path, row: dict[str, Any]) -> Path:
    return session / "cases" / f"case_{int(row['case_index']):03d}_{row['case_id']}"


def _latest_anchor_artifact_dir(case_dir: Path) -> Path | None:
    dirs = sorted(case_dir.glob("pipeline_session/session_*/artifacts/04_anchor_selection"))
    return dirs[-1] if dirs else None


def _ensure_clean_review_images(case_dir: Path) -> tuple[str | None, str | None]:
    clean_dir = case_dir / "clean_single_anchor_inputs"
    grid_path = case_dir / "clean_anchor_review_grid.png"
    if grid_path.exists():
        return str(grid_path), str(clean_dir)
    artifact_dir = _latest_anchor_artifact_dir(case_dir)
    if artifact_dir is None:
        return None, str(clean_dir) if clean_dir.exists() else None
    debug = render_clean_anchor_inputs_for_artifact_dir(
        artifact_dir=artifact_dir,
        output_dir=clean_dir,
        config={},
    )
    image_paths = {
        aid: meta.get("rendered_input_path")
        for aid, meta in (debug.get("anchors") or {}).items()
        if isinstance(meta, dict) and meta.get("rendered_input_path")
    }
    if image_paths:
        return build_clean_anchor_review_grid(image_paths=image_paths, output_path=grid_path), str(clean_dir)
    return None, str(clean_dir)


def label_session(*, session: str | Path, case_id: str | None = None, overwrite: bool = False, open_images: bool = True) -> Path:
    session_dir = Path(session)
    master = _read_rows(session_dir / "master_anchor_results.csv")
    if case_id:
        master = [row for row in master if row.get("case_id") == case_id]
    labels_dir = session_dir / "manual_labels"
    labels_path = labels_dir / "manual_anchor_labels.csv"
    existing_by_case = _existing_labels(labels_path)
    all_rows = [row for rows in existing_by_case.values() for row in rows] if not overwrite else []
    for row in master:
        cid = str(row.get("case_id"))
        if not overwrite and cid in existing_by_case:
            continue
        candidate_ids = _candidate_ids(row)
        if not candidate_ids:
            print(f"Skipping {cid}: no candidate anchors")
            continue
        print("=" * 60)
        print(f"E3 manual labels: {cid}")
        print(f"Material: {row.get('material')}  Condition: {row.get('condition_type')}")
        print(f"Candidate IDs: {','.join(candidate_ids)}")
        print(f"Geometry-only selected: {row.get('geometry_only_selected_anchor_id')}")
        print(f"Proposed selected: {row.get('proposed_selected_anchor_id')}")
        case_dir = _case_dir(session_dir, row)
        clean_grid_path = row.get("clean_anchor_review_grid_path")
        clean_dir = row.get("clean_single_anchor_inputs_dir")
        if not clean_grid_path or not Path(str(clean_grid_path)).exists():
            clean_grid_path, clean_dir = _ensure_clean_review_images(case_dir)
        if clean_grid_path:
            print(f"Clean anchor review grid: {clean_grid_path}")
        if clean_dir:
            print(f"Clean per-anchor images: {clean_dir}")
        if open_images:
            _open_image(clean_grid_path or row.get("wide_contact_grid_path") or row.get("selected_anchor_overlay_path"))
        while True:
            try:
                good = parse_anchor_id_list(input("Good anchors: "), candidate_ids)
                acceptable = parse_anchor_id_list(input("Acceptable anchors: "), candidate_ids)
                bad = parse_anchor_id_list(input("Bad anchors: "), candidate_ids)
                best = parse_anchor_id_list(input("Manual best anchor: "), candidate_ids, allow_empty=False)
                if len(best) != 1:
                    raise ValueError("manual best anchor must be exactly one ID")
                validate_case_labels(candidate_ids=candidate_ids, good=good, acceptable=acceptable, bad=bad, manual_best=best[0])
                if best[0] in bad:
                    confirm = input("Manual best is labeled Bad. Type yes to confirm: ").strip().lower()
                    if confirm != "yes":
                        continue
                notes = input("Optional notes: ").strip()
                action = input("Satisfied? [y/n/retry/skip/quit]: ").strip().lower() or "y"
                if action == "quit":
                    _write_rows(labels_path, all_rows, LABEL_COLUMNS)
                    _write_json(labels_dir / "labeling_progress.json", {"last_case_id": cid, "complete": False})
                    return labels_path
                if action in {"n", "retry"}:
                    continue
                if action == "skip":
                    break
                stamp = _timestamp()
                case_rows = []
                label_map = {anchor: "Good" for anchor in good}
                label_map.update({anchor: "Acceptable" for anchor in acceptable})
                label_map.update({anchor: "Bad" for anchor in bad})
                for anchor in candidate_ids:
                    label = label_map[anchor]
                    case_rows.append({
                        "case_index": row.get("case_index"),
                        "case_id": cid,
                        "material": row.get("material"),
                        "condition_type": row.get("condition_type"),
                        "axis_mode": row.get("axis_mode") or "major",
                        "anchor_id": anchor,
                        "manual_label": label,
                        "is_usable": label in {"Good", "Acceptable"},
                        "is_manual_best": anchor == best[0],
                        "manual_reason": notes,
                        "geometry_only_selected_anchor_id": row.get("geometry_only_selected_anchor_id"),
                        "proposed_selected_anchor_id": row.get("proposed_selected_anchor_id"),
                        "label_timestamp": stamp,
                    })
                all_rows = [existing for existing in all_rows if existing.get("case_id") != cid]
                all_rows.extend(case_rows)
                _write_rows(case_dir / "manual_anchor_labels.csv", case_rows, LABEL_COLUMNS)
                _write_json(case_dir / "manual_anchor_labels.json", {"case_id": cid, "anchors": case_rows})
                break
            except ValueError as exc:
                print(f"Invalid labels: {exc}")
    all_rows.sort(key=lambda item: (int(item.get("case_index") or 0), str(item.get("anchor_id"))))
    _write_rows(labels_path, all_rows, LABEL_COLUMNS)
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["case_id"], []).append(row)
    _write_json(labels_dir / "manual_anchor_labels.json", grouped)
    _write_json(labels_dir / "labeling_progress.json", {"complete": True, "case_count": len(grouped)})
    return labels_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually label E3 anchor candidates as Good, Acceptable, or Bad.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-open-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = label_session(session=args.session, case_id=args.case_id, overwrite=args.overwrite, open_images=not args.no_open_images)
    print(f"manual_labels: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
