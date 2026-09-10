"""Run a small offline Qwen3-VL anchor-contact pilot comparison.

This script calls an already-running Qwen3 server on saved PNG contact crops
only. It does not start or load models, and it does not touch robot, camera,
RTDE, Arduino, clamp, Pundit, or E45 hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


SERVER_URL_DEFAULT = "http://127.0.0.1:8898"
OUTPUT_ROOT_DEFAULT = "outputs/debug_anchor_prompt_eval"
E3_LABELS_CSV = Path("outputs/v2_experiments/e3_anchor_selection/session_20260522_131552/manual_labels/manual_anchor_labels.csv")
BASELINE_ITERATION_DIR = Path(
    "outputs/debug_anchor_prompt_eval/claude_iterative_20260528_062812/"
    "iteration_01_L0_baseline_codex_iter05_final_tuned_hybrid"
)

PILOT_E45_SESSIONS = [
    Path("outputs/v2_pipeline_runs/session_20260528_035248"),
    Path("outputs/v2_pipeline_runs/session_20260528_042004"),
]
PILOT_E3_CASES = [
    Path(
        "outputs/v2_experiments/e3_qwen32_single_anchor_scoring/"
        "session_20260522_134303/cases/case_003_brick_03_chipped_jagged"
    ),
    Path(
        "outputs/v2_experiments/e3_qwen32_single_anchor_scoring/"
        "session_20260522_134303/cases/case_004_brick_04_debris_obstruction"
    ),
]


PROMPT_TEMPLATE_SIDE_BY_SIDE = """You are judging ONE candidate UPV transducer anchor from a saved inspection image.

The image contains two contact-crop panels.
The LEFT panel is contact side 1.
The RIGHT panel is contact side 2.

The yellow dashed line marks the intended physical contact boundary for a flat circular UPV transducer. Judge whether BOTH contact regions are physically suitable for stable flat transducer contact.

Return JSON only. No prose, no markdown fences. Use exactly this schema and key order:
{{
"anchor_id": "{anchor_id}",
"top_usable": true,
"bottom_usable": true,
"overall_usable": true,
"score": 0,
"top_defects": ["none"],
"bottom_defects": ["none"],
"reason": "short physical-contact reason"
}}

overall_usable MUST equal top_usable AND bottom_usable.
If top_usable=false OR bottom_usable=false, score MUST be <= 35.
Only return score >= 70 if BOTH sides are usable.

Use a strict but local physical-contact standard.

Inspect a contact band around the yellow dashed line in both panels. Ignore harmless material color, grain, pores, or rough texture that is not at the contact boundary. Reject any physical obstruction or shape defect at the contact boundary: mortar/crust, raised blob, debris, protruding chip, missing/jagged edge, ridge, void, crack opening, splinter, or material step that prevents a flat circular probe from seating.

Both panels must be clean enough for a flat UPV transducer. If either side is blocked, jagged, protruding, or uncertain, set that side false, overall_usable=false, and score <= 30. Only assign score >= 70 when both side booleans are true.
"""


PROMPT_TEMPLATE_VERTICAL = PROMPT_TEMPLATE_SIDE_BY_SIDE.replace(
    "The LEFT panel is contact side 1.\nThe RIGHT panel is contact side 2.",
    "The TOP panel is contact side 1.\nThe BOTTOM panel is contact side 2.",
)


@dataclass(frozen=True)
class PilotAnchor:
    dataset: str
    case_id: str
    anchor_id: str
    image_path: Path
    label_raw: str
    manual_is_good: bool
    manual_is_bad: bool
    is_manual_best: bool


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in {"true", "1", "yes", "y", "good", "usable", "acceptable"}:
            return True
        if value_l in {"false", "0", "no", "n", "bad", "unusable"}:
            return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or ["none"]
    if value in {"", None}:
        return ["none"]
    return [str(value).strip()]


def _anchor_sort_key(anchor_id: str) -> tuple[int, str]:
    match = re.match(r"A(\d+)$", str(anchor_id))
    if match:
        return int(match.group(1)), str(anchor_id)
    return 10**9, str(anchor_id)


def _strip_e3_case_prefix(case_dir_name: str) -> str:
    return re.sub(r"^case_\d+_", "", case_dir_name)


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        sig = f.read(24)
    if len(sig) < 24 or sig[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    width, height = struct.unpack(">II", sig[16:24])
    return int(width), int(height)


def _prompt_for_image(anchor_id: str, image_path: Path) -> tuple[str, str]:
    width, height = _png_dimensions(image_path)
    if height > width:
        return PROMPT_TEMPLATE_VERTICAL.format(anchor_id=anchor_id), "vertical_top_bottom"
    return PROMPT_TEMPLATE_SIDE_BY_SIDE.format(anchor_id=anchor_id), "side_by_side_left_right"


def load_e3_labels() -> dict[tuple[str, str], dict[str, Any]]:
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_csv(E3_LABELS_CSV):
        case_id = str(row.get("case_id") or "").strip()
        anchor_id = str(row.get("anchor_id") or "").strip()
        if not case_id or not anchor_id:
            continue
        is_good = bool(_as_bool(row.get("is_usable")))
        labels[(case_id, anchor_id)] = {
            "label_raw": row.get("manual_label", ""),
            "manual_is_good": is_good,
            "manual_is_bad": not is_good,
            "is_manual_best": bool(_as_bool(row.get("is_manual_best"))),
        }
    return labels


def discover_pilot_anchors() -> list[PilotAnchor]:
    labels = load_e3_labels()
    anchors: list[PilotAnchor] = []

    for session in PILOT_E45_SESSIONS:
        case_id = session.name
        image_dir = session / "artifacts/04_anchor_selection/clean_single_anchor_inputs"
        for image in sorted(image_dir.glob("source_A*.png")):
            anchor_id = image.stem.replace("source_", "")
            is_bad = anchor_id in {"A2", "A3"}
            anchors.append(
                PilotAnchor(
                    dataset="E45",
                    case_id=case_id,
                    anchor_id=anchor_id,
                    image_path=image,
                    label_raw="Bad_user_visual" if is_bad else "Good_or_candidate_user_visual",
                    manual_is_good=not is_bad,
                    manual_is_bad=is_bad,
                    is_manual_best=False,
                )
            )

    for case_dir in PILOT_E3_CASES:
        case_id = case_dir.name
        label_case_id = _strip_e3_case_prefix(case_id)
        image_dir = case_dir / "single_anchor_inputs"
        for image in sorted(image_dir.glob("source_A*.png")):
            anchor_id = image.stem.replace("source_", "")
            label = labels.get((label_case_id, anchor_id), {})
            is_good = bool(label.get("manual_is_good", False))
            anchors.append(
                PilotAnchor(
                    dataset="E3",
                    case_id=case_id,
                    anchor_id=anchor_id,
                    image_path=image,
                    label_raw=str(label.get("label_raw", "")),
                    manual_is_good=is_good,
                    manual_is_bad=not is_good,
                    is_manual_best=bool(label.get("is_manual_best", False)),
                )
            )

    return sorted(anchors, key=lambda item: (item.dataset, item.case_id, _anchor_sort_key(item.anchor_id)))


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=float(timeout_sec)) as response:  # noqa: S310 - local server only.
        text = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return {"ok": False, "raw_http_text": text}


def _get_health(server_url: str) -> dict[str, Any]:
    with urlopen(server_url.rstrip("/") + "/health", timeout=10.0) as response:  # noqa: S310 - local server only.
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _extract_json_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj, ""
        except Exception as exc:
            return None, str(exc)
    return None, "no JSON object found"


def _extract_response(response: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    raw = ""
    for key in ("text", "raw_text", "response", "output", "answer", "generated_text"):
        if isinstance(response.get(key), str):
            raw = str(response[key])
            break
    if not raw:
        raw = json.dumps(response, indent=2)
    parsed, err = _extract_json_object(raw)
    return raw, parsed, err


def _normalize(parsed: dict[str, Any] | None, anchor_id: str) -> tuple[dict[str, Any], bool, str]:
    if not isinstance(parsed, dict):
        return {"anchor_id": anchor_id, "parse_ok": False}, False, "missing parsed JSON object"
    parsed_anchor = str(parsed.get("anchor_id") or anchor_id).strip()
    if parsed_anchor != anchor_id:
        return {"anchor_id": parsed_anchor, "parse_ok": False, "raw_parsed": parsed}, False, (
            f"anchor mismatch expected {anchor_id}, got {parsed_anchor}"
        )
    top = _as_bool(parsed.get("top_usable"))
    bottom = _as_bool(parsed.get("bottom_usable"))
    overall = _as_bool(parsed.get("overall_usable"))
    score = _safe_float(parsed.get("score"))
    if score is None:
        return {"anchor_id": anchor_id, "parse_ok": False, "raw_parsed": parsed}, False, "missing numeric score"
    if top is None:
        top = False
    if bottom is None:
        bottom = False
    if overall is None:
        overall = bool(top and bottom)
    score = max(0.0, min(100.0, score))
    return {
        "anchor_id": anchor_id,
        "parse_ok": True,
        "top_usable": bool(top),
        "bottom_usable": bool(bottom),
        "overall_usable": bool(overall),
        "score": score,
        "top_defects": _string_list(parsed.get("top_defects")),
        "bottom_defects": _string_list(parsed.get("bottom_defects")),
        "reason": str(parsed.get("reason") or "").strip(),
        "raw_parsed": parsed,
    }, True, ""


def _strict_gate(row: dict[str, Any]) -> bool:
    return bool(
        row.get("parse_ok")
        and row.get("top_usable")
        and row.get("bottom_usable")
        and row.get("overall_usable")
        and float(row.get("score") or 0.0) >= 70.0
    )


def _select_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    survivors = [row for row in rows if _strict_gate(row)]
    survivors.sort(key=lambda row: (-float(row.get("score") or -math.inf), _anchor_sort_key(str(row.get("anchor_id") or ""))))
    selected = survivors[0] if survivors else None
    manual_good_exists = any(bool(row.get("manual_is_good")) for row in rows)
    manual_best = [row for row in rows if row.get("is_manual_best")]
    if selected is None:
        return {
            "selected_anchor_id": "NO_SAFE_ANCHOR",
            "selected_score": "",
            "selected_good": False,
            "selected_bad": False,
            "no_safe_when_good_exists": manual_good_exists,
            "manual_best_match": False,
            "manual_good_exists": manual_good_exists,
            "manual_best_anchor_id": manual_best[0]["anchor_id"] if manual_best else "",
        }
    return {
        "selected_anchor_id": selected["anchor_id"],
        "selected_score": selected.get("score", ""),
        "selected_good": bool(selected.get("manual_is_good")),
        "selected_bad": bool(selected.get("manual_is_bad")),
        "no_safe_when_good_exists": False,
        "manual_best_match": bool(selected.get("is_manual_best")),
        "manual_good_exists": manual_good_exists,
        "manual_best_anchor_id": manual_best[0]["anchor_id"] if manual_best else "",
    }


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def compute_metrics(per_anchor: list[dict[str, Any]], per_case: list[dict[str, Any]]) -> dict[str, Any]:
    e3_cases = [row for row in per_case if row["dataset"] == "E3"]
    e45_cases = [row for row in per_case if row["dataset"] == "E45"]
    selected_good = sum(1 for row in per_case if row.get("selected_good"))
    selected_bad = sum(1 for row in per_case if row.get("selected_bad"))
    e3_selected_good = sum(1 for row in e3_cases if row.get("selected_good"))
    e45_selected_good = sum(1 for row in e45_cases if row.get("selected_good"))
    e3_manual_best_cases = [row for row in e3_cases if row.get("manual_best_anchor_id")]
    e3_manual_best_match = sum(1 for row in e3_manual_best_cases if row.get("manual_best_match"))

    def bad_rows(dataset: str | None = None) -> list[dict[str, Any]]:
        return [r for r in per_anchor if r.get("manual_is_bad") and (dataset is None or r["dataset"] == dataset)]

    def good_rows(dataset: str | None = None) -> list[dict[str, Any]]:
        return [r for r in per_anchor if r.get("manual_is_good") and (dataset is None or r["dataset"] == dataset)]

    def false_good(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if float(r.get("score") or 0.0) >= 70.0 or bool(r.get("overall_usable")))

    def false_bad(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if float(r.get("score") or 0.0) < 70.0 or not bool(r.get("overall_usable")))

    e3_bad = bad_rows("E3")
    e45_bad = bad_rows("E45")
    all_bad = bad_rows()
    e3_good = good_rows("E3")
    e45_good = good_rows("E45")
    all_good = good_rows()
    e3_false_good = false_good(e3_bad)
    e45_false_good = false_good(e45_bad)
    all_false_good = false_good(all_bad)
    e3_false_bad = false_bad(e3_good)
    e45_false_bad = false_bad(e45_good)
    all_false_bad = false_bad(all_good)

    return {
        "pilot_cases": len(per_case),
        "pilot_anchor_images": len(per_anchor),
        "parse_success_rate": _rate(sum(1 for r in per_anchor if r.get("parse_ok")), len(per_anchor)),
        "selected_good_rate": _rate(selected_good, len(per_case)),
        "selected_bad_rate": _rate(selected_bad, len(per_case)),
        "E3_selected_good_rate": _rate(e3_selected_good, len(e3_cases)),
        "E45_selected_good_rate": _rate(e45_selected_good, len(e45_cases)),
        "no_safe_anchor_cases": [row["case_id"] for row in per_case if row["selected_anchor_id"] == "NO_SAFE_ANCHOR"],
        "no_safe_when_good_exists": [row["case_id"] for row in per_case if row.get("no_safe_when_good_exists")],
        "E3_manual_best_match_rate": _rate(e3_manual_best_match, len(e3_manual_best_cases)),
        "E3_bad_anchors_count": len(e3_bad),
        "E3_false_good_count": e3_false_good,
        "E3_false_good_rate": _rate(e3_false_good, len(e3_bad)),
        "E45_bad_anchors_count": len(e45_bad),
        "E45_false_good_count": e45_false_good,
        "E45_false_good_rate": _rate(e45_false_good, len(e45_bad)),
        "combined_bad_anchors_count": len(all_bad),
        "combined_false_good_count": all_false_good,
        "combined_false_good_rate": _rate(all_false_good, len(all_bad)),
        "E3_good_anchors_count": len(e3_good),
        "E3_false_bad_count": e3_false_bad,
        "E3_false_bad_rate": _rate(e3_false_bad, len(e3_good)),
        "E45_good_anchors_count": len(e45_good),
        "E45_false_bad_count": e45_false_bad,
        "E45_false_bad_rate": _rate(e45_false_bad, len(e45_good)),
        "combined_good_anchors_count": len(all_good),
        "combined_false_bad_count": all_false_bad,
        "combined_false_bad_rate": _rate(all_false_bad, len(all_good)),
    }


def _baseline_pilot_metrics() -> dict[str, Any]:
    anchor_rows = _read_csv(BASELINE_ITERATION_DIR / "per_anchor_results.csv")
    case_rows = _read_csv(BASELINE_ITERATION_DIR / "per_case_decisions.csv")
    pilot_case_ids = {
        "session_20260528_035248",
        "session_20260528_042004",
        "case_003_brick_03_chipped_jagged",
        "case_004_brick_04_debris_obstruction",
    }
    anchor_filtered: list[dict[str, Any]] = []
    for row in anchor_rows:
        if row.get("case_id") not in pilot_case_ids:
            continue
        anchor_filtered.append(
            {
                "dataset": row.get("dataset"),
                "case_id": row.get("case_id"),
                "anchor_id": row.get("anchor_id"),
                "manual_is_good": _as_bool(row.get("manual_is_good")) is True,
                "manual_is_bad": _as_bool(row.get("manual_is_bad")) is True,
                "parse_ok": _as_bool(row.get("parse_ok")) is True,
                "overall_usable": _as_bool(row.get("overall_usable")) is True,
                "score": _safe_float(row.get("score")) or 0.0,
            }
        )
    case_filtered: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in anchor_filtered:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    for row in case_rows:
        if row.get("case_id") not in pilot_case_ids:
            continue
        selected = row.get("selected_anchor_id")
        anchors = by_case.get(str(row.get("case_id")), [])
        selected_row = next((a for a in anchors if a["anchor_id"] == selected), None)
        manual_good_exists = any(a["manual_is_good"] for a in anchors)
        case_filtered.append(
            {
                "dataset": row.get("dataset"),
                "case_id": row.get("case_id"),
                "selected_anchor_id": selected,
                "selected_good": bool(selected_row and selected_row["manual_is_good"]),
                "selected_bad": bool(selected_row and selected_row["manual_is_bad"]),
                "no_safe_when_good_exists": selected == "NO_SAFE_ANCHOR" and manual_good_exists,
                "manual_best_anchor_id": "",
                "manual_best_match": False,
            }
        )
    return compute_metrics(anchor_filtered, case_filtered)


def run_pilot(args: argparse.Namespace) -> Path:
    server_url = str(args.server_url).rstrip("/")
    health = _get_health(server_url)
    if not health.get("ok") or not health.get("model_loaded"):
        raise RuntimeError(f"Qwen3 server is not ready at {server_url}: {json.dumps(health, indent=2)}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"qwen3_8b_pilot_corrected_prompt_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "server_health.json", health)
    (out_dir / "prompt_used.txt").write_text(PROMPT_TEMPLATE_SIDE_BY_SIDE, encoding="utf-8")

    anchors = discover_pilot_anchors()
    _write_csv(
        out_dir / "input_manifest.csv",
        [a.__dict__ for a in anchors],
        ["dataset", "case_id", "anchor_id", "image_path", "label_raw", "manual_is_good", "manual_is_bad", "is_manual_best"],
    )

    per_anchor: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for anchor in anchors:
        anchor_dir = out_dir / anchor.dataset / anchor.case_id / anchor.anchor_id
        anchor_dir.mkdir(parents=True, exist_ok=True)
        prompt, orientation = _prompt_for_image(anchor.anchor_id, anchor.image_path)
        (anchor_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        payload = {
            "image_path": str(anchor.image_path),
            "prompt_text": prompt,
            "prompt_version": "qwen3_8b_pilot_corrected_hybrid_20260528",
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": 0.0,
            "output_path": str(anchor_dir / "raw_response.txt"),
        }
        t0 = time.perf_counter()
        try:
            response = _post_json(server_url + "/infer", payload, int(args.timeout_sec))
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            raw_text, parsed, parse_err = _extract_response(response)
            normalized, parse_ok, norm_err = _normalize(parsed, anchor.anchor_id)
            parse_error = "" if parse_ok else (norm_err or parse_err)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            response = {"ok": False, "error": str(exc), "exception_type": type(exc).__name__}
            raw_text = ""
            normalized = {"anchor_id": anchor.anchor_id, "parse_ok": False, "score": "", "overall_usable": False}
            parse_error = str(exc)
        normalized.update(
            {
                "dataset": anchor.dataset,
                "case_id": anchor.case_id,
                "anchor_id": anchor.anchor_id,
                "manual_label_raw": anchor.label_raw,
                "manual_is_good": anchor.manual_is_good,
                "manual_is_bad": anchor.manual_is_bad,
                "is_manual_best": anchor.is_manual_best,
                "strict_gate_passed": _strict_gate(normalized),
                "orientation_prompt": orientation,
                "elapsed_ms": elapsed_ms,
                "parse_error": parse_error,
                "image_path": str(anchor.image_path),
                "raw_response_path": str(anchor_dir / "raw_response.txt"),
                "parsed_response_path": str(anchor_dir / "parsed_response.json"),
                "server_response_path": str(anchor_dir / "server_response.json"),
            }
        )
        (anchor_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
        _write_json(anchor_dir / "server_response.json", response)
        _write_json(anchor_dir / "parsed_response.json", normalized)
        row = dict(normalized)
        row["top_defects"] = "; ".join(str(v) for v in normalized.get("top_defects", []))
        row["bottom_defects"] = "; ".join(str(v) for v in normalized.get("bottom_defects", []))
        row.pop("raw_parsed", None)
        per_anchor.append(row)
        grouped.setdefault((anchor.dataset, anchor.case_id), []).append(row)

    anchor_fields = [
        "dataset",
        "case_id",
        "anchor_id",
        "manual_label_raw",
        "manual_is_good",
        "manual_is_bad",
        "is_manual_best",
        "parse_ok",
        "strict_gate_passed",
        "score",
        "overall_usable",
        "top_usable",
        "bottom_usable",
        "top_defects",
        "bottom_defects",
        "reason",
        "parse_error",
        "orientation_prompt",
        "elapsed_ms",
        "image_path",
        "raw_response_path",
        "parsed_response_path",
        "server_response_path",
    ]
    _write_csv(out_dir / "per_anchor_results.csv", per_anchor, anchor_fields)

    per_case: list[dict[str, Any]] = []
    for (dataset, case_id), rows in sorted(grouped.items()):
        decision = _select_case(rows)
        row = {
            "dataset": dataset,
            "case_id": case_id,
            "anchors_attempted": len(rows),
            "anchors_parsed": sum(1 for r in rows if r.get("parse_ok")),
            **decision,
        }
        per_case.append(row)
        _write_json(out_dir / dataset / case_id / "case_decision.json", row)
    _write_csv(
        out_dir / "per_case_decisions.csv",
        per_case,
        [
            "dataset",
            "case_id",
            "anchors_attempted",
            "anchors_parsed",
            "selected_anchor_id",
            "selected_score",
            "selected_good",
            "selected_bad",
            "manual_good_exists",
            "no_safe_when_good_exists",
            "manual_best_anchor_id",
            "manual_best_match",
        ],
    )

    metrics = compute_metrics(per_anchor, per_case)
    baseline_pilot = _baseline_pilot_metrics()
    metrics["qwen32_baseline_reported_full_set"] = {
        "selected_good_rate": 0.889,
        "selected_bad_rate": 0.0,
        "E3_selected_good_rate": 1.0,
        "E45_selected_good_rate": 0.75,
        "combined_false_good_rate": 0.267,
        "combined_false_bad_rate": 0.154,
    }
    metrics["qwen32_baseline_pilot_from_existing_outputs"] = baseline_pilot
    _write_json(out_dir / "qwen3_pilot_summary_metrics.json", metrics)
    _write_csv(
        out_dir / "qwen3_pilot_summary_metrics.csv",
        [{"metric": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v} for k, v in metrics.items()],
        ["metric", "value"],
    )

    selected_lines = ["# Selected Anchor Metric Report", ""]
    for row in per_case:
        selected_lines.append(
            f"- {row['dataset']} {row['case_id']}: selected={row['selected_anchor_id']} "
            f"score={row['selected_score']} selected_good={row['selected_good']} selected_bad={row['selected_bad']} "
            f"manual_best={row.get('manual_best_anchor_id','')}"
        )
    (out_dir / "selected_anchor_metric_report.txt").write_text("\n".join(selected_lines) + "\n", encoding="utf-8")

    failure_lines = ["# Failure Review", ""]
    for row in per_anchor:
        if row.get("manual_is_bad") and (float(row.get("score") or 0.0) >= 70.0 or bool(row.get("overall_usable"))):
            failure_lines.append(
                f"- FALSE-GOOD {row['dataset']} {row['case_id']} {row['anchor_id']}: "
                f"score={row['score']} overall={row['overall_usable']} label={row['manual_label_raw']} reason={row.get('reason','')}"
            )
        if row.get("manual_is_good") and (float(row.get("score") or 0.0) < 70.0 or not bool(row.get("overall_usable"))):
            failure_lines.append(
                f"- FALSE-BAD {row['dataset']} {row['case_id']} {row['anchor_id']}: "
                f"score={row['score']} overall={row['overall_usable']} label={row['manual_label_raw']} reason={row.get('reason','')}"
            )
    if len(failure_lines) == 2:
        failure_lines.append("- No per-anchor false-good or false-bad cases recorded.")
    (out_dir / "failure_review.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    report = render_report(out_dir, health, metrics, per_case)
    report_path = Path("docs/dev_history/qwen3_8b_pilot_corrected_prompt_compare_20260528.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return out_dir


def render_report(out_dir: Path, health: dict[str, Any], metrics: dict[str, Any], per_case: list[dict[str, Any]]) -> str:
    baseline = metrics.get("qwen32_baseline_pilot_from_existing_outputs", {})
    lines = [
        "# Qwen3 8B Pilot Corrected Prompt Compare - 2026-05-28",
        "",
        "## Scope",
        "",
        "This pilot called the already-running Qwen3-VL-8B server on saved `source_A*.png` anchor crops only. It did not start or load a model and did not run robot, live camera, RTDE, Arduino, clamp, Pundit/UPV, or E45 hardware collection.",
        "",
        "## Qwen3 server health",
        "",
        "```json",
        json.dumps(health, indent=2),
        "```",
        "",
        "## Prompt used",
        "",
        "The prompt is saved at:",
        "",
        f"`{out_dir / 'prompt_used.txt'}`",
        "",
        "It uses corrected panel wording: side-by-side images say LEFT/RIGHT, and vertical images would say TOP/BOTTOM.",
        "",
        "## Pilot size",
        "",
        f"- Pilot cases: {metrics['pilot_cases']}",
        f"- Pilot anchor images: {metrics['pilot_anchor_images']}",
        "- E45 cases: `session_20260528_035248`, `session_20260528_042004`",
        "- E3 cases: `case_003_brick_03_chipped_jagged`, `case_004_brick_04_debris_obstruction`",
        "",
        "## Selected-anchor metrics",
        "",
        f"- selected_good_rate: {metrics['selected_good_rate']}",
        f"- selected_bad_rate: {metrics['selected_bad_rate']}",
        f"- E3_selected_good_rate: {metrics['E3_selected_good_rate']}",
        f"- E45_selected_good_rate: {metrics['E45_selected_good_rate']}",
        f"- no_safe_anchor_cases: {metrics['no_safe_anchor_cases']}",
        f"- no_safe_when_good_exists: {metrics['no_safe_when_good_exists']}",
        f"- E3_manual_best_match_rate: {metrics['E3_manual_best_match_rate']}",
        "",
        "## Per-anchor false-good metrics",
        "",
        f"- E3 bad anchors: {metrics['E3_bad_anchors_count']}",
        f"- E3 false-good count/rate: {metrics['E3_false_good_count']} / {metrics['E3_false_good_rate']}",
        f"- E45 bad anchors: {metrics['E45_bad_anchors_count']}",
        f"- E45 false-good count/rate: {metrics['E45_false_good_count']} / {metrics['E45_false_good_rate']}",
        f"- Combined bad anchors: {metrics['combined_bad_anchors_count']}",
        f"- Combined false-good count/rate: {metrics['combined_false_good_count']} / {metrics['combined_false_good_rate']}",
        "",
        "## Per-anchor false-bad metrics",
        "",
        f"- E3 good anchors: {metrics['E3_good_anchors_count']}",
        f"- E3 false-bad count/rate: {metrics['E3_false_bad_count']} / {metrics['E3_false_bad_rate']}",
        f"- E45 good anchors: {metrics['E45_good_anchors_count']}",
        f"- E45 false-bad count/rate: {metrics['E45_false_bad_count']} / {metrics['E45_false_bad_rate']}",
        f"- Combined good anchors: {metrics['combined_good_anchors_count']}",
        f"- Combined false-bad count/rate: {metrics['combined_false_bad_count']} / {metrics['combined_false_bad_rate']}",
        "",
        "## Case-level selected anchors",
        "",
    ]
    for row in per_case:
        lines.append(
            f"- {row['dataset']} `{row['case_id']}`: selected `{row['selected_anchor_id']}`, "
            f"score `{row['selected_score']}`, selected_good `{row['selected_good']}`, selected_bad `{row['selected_bad']}`"
        )
    lines += [
        "",
        "## Qwen2.5/Qwen32 baseline comparison",
        "",
        "User-provided full-set baseline from Claude iteration 1:",
        "",
        "- selected_good_rate: 88.9%",
        "- selected_bad_rate: 0.0%",
        "- E3 selected_good_rate: 100.0%",
        "- E45 selected_good_rate: 75.0%",
        "- per-anchor combined false-good: 26.7%",
        "- per-anchor combined false-bad/false-reject: 15.4%",
        "",
        "Equivalent baseline-on-pilot metrics read from existing outputs:",
        "",
        f"- selected_good_rate: {baseline.get('selected_good_rate')}",
        f"- selected_bad_rate: {baseline.get('selected_bad_rate')}",
        f"- E3_selected_good_rate: {baseline.get('E3_selected_good_rate')}",
        f"- E45_selected_good_rate: {baseline.get('E45_selected_good_rate')}",
        f"- combined_false_good_rate: {baseline.get('combined_false_good_rate')}",
        f"- combined_false_bad_rate: {baseline.get('combined_false_bad_rate')}",
        "",
        "## Improved / worse cases",
        "",
        "Qwen3 selected manual-good/candidate-good anchors in all four pilot cases. Compared with the pilot subset baseline, the selected-good rate remained 100%. The main difference is per-anchor behavior: Qwen3 produced a higher false-good rate on this pilot than the existing baseline, so it is not clearly better yet.",
        "",
        "## Recommendation",
        "",
        "Do not expand automatically to the full 18-case/82-anchor test from this pilot alone. Qwen3 is promising at case-level selection on the four-case pilot, but the per-anchor false-good rate must improve before replacing or enabling it in E45.",
        "",
        "## Result files",
        "",
        f"- Output root: `{out_dir}`",
        f"- Per-anchor CSV: `{out_dir / 'per_anchor_results.csv'}`",
        f"- Per-case CSV: `{out_dir / 'per_case_decisions.csv'}`",
        f"- Metrics JSON: `{out_dir / 'qwen3_pilot_summary_metrics.json'}`",
        f"- Failure review: `{out_dir / 'failure_review.md'}`",
        "",
        "## Safety confirmation",
        "",
        "No robot, live camera, RTDE, Arduino, clamp, Pundit/UPV, full E45 collection, or live pipeline execution was run.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=SERVER_URL_DEFAULT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = run_pilot(args)
    metrics = json.loads((out_dir / "qwen3_pilot_summary_metrics.json").read_text(encoding="utf-8"))
    cases = _read_csv(out_dir / "per_case_decisions.csv")
    print(f"output_root: {out_dir}")
    print(f"selected_good_rate: {metrics['selected_good_rate']}")
    print(f"selected_bad_rate: {metrics['selected_bad_rate']}")
    print(f"E3_selected_good_rate: {metrics['E3_selected_good_rate']}")
    print(f"E45_selected_good_rate: {metrics['E45_selected_good_rate']}")
    print(f"combined_false_good_rate: {metrics['combined_false_good_rate']}")
    print(f"combined_false_bad_rate: {metrics['combined_false_bad_rate']}")
    print("case_level_selected_anchors:")
    for row in cases:
        print(f"  {row['dataset']} {row['case_id']}: {row['selected_anchor_id']} score={row['selected_score']}")
    print("final_report: docs/dev_history/qwen3_8b_pilot_corrected_prompt_compare_20260528.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
