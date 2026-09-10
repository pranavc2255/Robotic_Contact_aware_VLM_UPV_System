"""Run offline iterative Qwen32 anchor-contact prompt trials on saved crops.

This runner only reads saved source_A*.png images and calls the local Qwen
server. It never touches robot, camera, RTDE, Arduino, clamp, or UPV hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from upv_vlm_v2.experiments.analyze_qwen_anchor_prompt_trial import analyze_session


E3_SCORING_SESSION = Path("outputs/v2_experiments/e3_qwen32_single_anchor_scoring/session_20260522_134303")
E3_LABEL_SESSION = Path("outputs/v2_experiments/e3_anchor_selection/session_20260522_131552")
E45_SESSIONS = [
    Path("outputs/v2_pipeline_runs/session_20260528_035248"),
    Path("outputs/v2_pipeline_runs/session_20260528_035433"),
    Path("outputs/v2_pipeline_runs/session_20260528_035620"),
    Path("outputs/v2_pipeline_runs/session_20260528_035801"),
    Path("outputs/v2_pipeline_runs/session_20260528_035949"),
    Path("outputs/v2_pipeline_runs/session_20260528_041815"),
    Path("outputs/v2_pipeline_runs/session_20260528_042004"),
    Path("outputs/v2_pipeline_runs/session_20260528_042147"),
]

STRICT_GATE = {
    "parse_ok": True,
    "overall_usable": True,
    "top_usable": True,
    "bottom_usable": True,
    "min_score": 70.0,
}


PROMPT_PREAMBLE = """You are judging ONE candidate UPV transducer anchor from a saved inspection image.

The image contains two contact-crop panels:
- left/top panel: one side of the material edge
- right/bottom panel: the opposite side of the material edge

The yellow dashed line marks the intended physical contact boundary for a flat circular UPV transducer. Judge whether BOTH contact regions are physically suitable for stable flat transducer contact.

Return JSON only. Use exactly these keys:
{{
  "anchor_id": "{anchor_id}",
  "top_defects": ["none"],
  "bottom_defects": ["none"],
  "top_usable": true,
  "bottom_usable": true,
  "overall_usable": true,
  "score": 0,
  "reason": "short physical-contact reason"
}}
"""


PROMPT_VARIANTS: list[tuple[str, str, str]] = [
    (
        "strict_contact_line_binary",
        "saved_clean_source_crop",
        PROMPT_PREAMBLE
        + """
Focus only near the yellow dashed contact line and the immediately adjacent contact band.

Reject the anchor if either panel has raised material, mortar, crust, chip ridge, protrusion, rough ridge, void, debris, jagged break, or non-flat obstruction touching, crossing, or close enough to interfere with the dashed line. Natural color or texture variation away from the dashed line is acceptable.

BOTH sides must be usable. If either side is unusable, set overall_usable=false and score <= 30. If uncertain, mark unusable. Score >= 80 only when both sides are clearly clean, flat, continuous, open, and physically stable.
""",
    ),
    (
        "locality_less_over_strict",
        "saved_clean_source_crop",
        PROMPT_PREAMBLE
        + """
Look in a narrow band around the yellow dashed line. Ignore background outside the object and harmless color/texture variation far from the contact line.

Reject physical height or edge irregularity, not ordinary material color. Reject mortar blobs, protruding material, chips, broken edge geometry, loose debris, or obstructions that would prevent a flat circular probe from seating on either side.

Set top_usable and bottom_usable independently. overall_usable is true only when both are true. Score 70-100 only for anchors physically usable on both sides; score <= 30 if either side is blocked or non-flat.
""",
    ),
    (
        "comparative_policy_no_least_bad",
        "saved_clean_source_crop",
        PROMPT_PREAMBLE
        + """
This image is one anchor from a case that may contain several anchors. Do not choose a least-bad anchor. This anchor must pass an absolute physical-contact standard.

If a flat circular UPV transducer cannot contact BOTH visible edge regions cleanly, mark this anchor unusable even if it might be better than other anchors. If both sides are bad, return overall_usable=false. If only one side is bad, return overall_usable=false.

Use score 90-100 for clean, flat, uninterrupted contact lines; 70-89 for usable with minor harmless texture; 40-69 for uncertain or marginal support; 0-39 for any obstruction, debris, chip, mortar, ridge, jagged edge, or one unusable side.
""",
    ),
    (
        "two_stage_side_first",
        "saved_clean_source_crop",
        PROMPT_PREAMBLE
        + """
Apply this two-stage reasoning before writing JSON:
1. Decide whether the top/left panel permits flat_contact_possible at the dashed line.
2. Decide whether the bottom/right panel permits flat_contact_possible at the dashed line.
3. Set top_usable and bottom_usable from those two decisions.
4. Set overall_usable=true only if top_usable=true AND bottom_usable=true.

One bad side makes the whole anchor unusable. If overall_usable=false, score must be below 40. Mention the visible defect evidence in reason and in top_defects/bottom_defects.
""",
    ),
    (
        "final_tuned_hybrid_strict_gate",
        "saved_clean_source_crop",
        PROMPT_PREAMBLE
        + """
Use a strict but local physical-contact standard.

Inspect a contact band around the yellow dashed line in both panels. Ignore harmless material color, grain, pores, or rough texture that is not at the contact boundary. Reject any physical obstruction or shape defect at the contact boundary: mortar/crust, raised blob, debris, protruding chip, missing/jagged edge, ridge, void, crack opening, splinter, or material step that prevents a flat circular probe from seating.

Both panels must be clean enough for a flat UPV transducer. If either side is blocked, jagged, protruding, or uncertain, set that side false, overall_usable=false, and score <= 30. Only assign score >= 70 when both side booleans are true. Use NO generosity for a least-bad anchor; unusable anchors must remain low-scored.
""",
    ),
]


@dataclass(frozen=True)
class AnchorImage:
    dataset: str
    case_id: str
    anchor_id: str
    image_path: Path
    source_session: Path
    old_selected_anchor_id: str = ""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=float(timeout_sec)) as response:  # noqa: S310 - local server only.
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Qwen HTTPError {exc.code}: {body}") from exc
    try:
        return json.loads(text)
    except Exception:
        return {"ok": False, "raw_http_text": text}


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
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
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


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "usable", "good", "acceptable"}:
            return True
        if s in {"false", "0", "no", "n", "bad", "unusable"}:
            return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or ["none"]
    if value is None or value == "":
        return ["none"]
    return [str(value).strip()]


def _nested_bool(parsed: dict[str, Any], key: str, nested_keys: list[str]) -> bool | None:
    direct = _as_bool(parsed.get(key))
    if direct is not None:
        return direct
    for parent in nested_keys:
        child = parsed.get(parent)
        if isinstance(child, dict):
            for ck in ("flat_contact_possible", "usable", "contact_usable", "is_usable"):
                val = _as_bool(child.get(ck))
                if val is not None:
                    return val
    return None


def _normalize(parsed: dict[str, Any] | None, anchor_id: str) -> tuple[dict[str, Any], bool, str]:
    if not isinstance(parsed, dict):
        return {"anchor_id": anchor_id, "parse_ok": False}, False, "missing parsed JSON object"
    parsed_anchor = str(parsed.get("anchor_id") or anchor_id).strip()
    if parsed_anchor != anchor_id:
        return {
            "anchor_id": parsed_anchor,
            "expected_anchor_id": anchor_id,
            "parse_ok": False,
            "raw_parsed": parsed,
        }, False, f"anchor mismatch expected {anchor_id}, got {parsed_anchor}"
    score = _safe_float(parsed.get("score") or parsed.get("overall_score") or parsed.get("contact_score"))
    if score is None:
        return {"anchor_id": anchor_id, "parse_ok": False, "raw_parsed": parsed}, False, "missing score"
    score = max(0.0, min(100.0, score))
    top = _nested_bool(parsed, "top_usable", ["top", "top_side", "top_contact"])
    bottom = _nested_bool(parsed, "bottom_usable", ["bottom", "bottom_side", "bottom_contact"])
    overall = _as_bool(parsed.get("overall_usable") or parsed.get("usable") or parsed.get("contact_usable"))
    if top is None:
        top = False
    if bottom is None:
        bottom = False
    if overall is None:
        overall = bool(top and bottom and score >= 70.0)
    norm = {
        "anchor_id": anchor_id,
        "parse_ok": True,
        "score": score,
        "top_usable": bool(top),
        "bottom_usable": bool(bottom),
        "overall_usable": bool(overall),
        "top_defects": _string_list(parsed.get("top_defects") or parsed.get("top_evidence")),
        "bottom_defects": _string_list(parsed.get("bottom_defects") or parsed.get("bottom_evidence")),
        "reason": str(parsed.get("reason") or parsed.get("explanation") or "").strip(),
        "raw_parsed": parsed,
    }
    return norm, True, ""


def _anchor_sort_key(anchor_id: str) -> tuple[int, str]:
    match = re.match(r"A(\d+)$", anchor_id)
    return (int(match.group(1)), anchor_id) if match else (10**9, anchor_id)


def _strict_ok(result: dict[str, Any]) -> bool:
    return bool(
        result.get("parse_ok")
        and result.get("overall_usable")
        and result.get("top_usable")
        and result.get("bottom_usable")
        and float(result.get("score") or 0.0) >= float(STRICT_GATE["min_score"])
    )


def _load_old_selected_anchor(session: Path) -> str:
    candidates = [
        session / "artifacts/04_anchor_selection/qwen32_single_anchor_decision.json",
        session / "artifacts/04_anchor_selection/final_anchor_decision_wide_context.json",
        session / "pipeline_result.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("selected_anchor_id", "selected_anchor", "final_selected_anchor_id"):
            if data.get(key):
                return str(data[key])
        anchor = data.get("anchor_selection") if isinstance(data.get("anchor_selection"), dict) else {}
        for key in ("selected_anchor_id", "selected_anchor"):
            if anchor.get(key):
                return str(anchor[key])
    return ""


def discover_images() -> list[AnchorImage]:
    images: list[AnchorImage] = []
    for case_dir in sorted((E3_SCORING_SESSION / "cases").glob("case_*")):
        clean_dir = case_dir / "single_anchor_inputs"
        if not clean_dir.exists():
            continue
        case_id = case_dir.name
        for image in sorted(clean_dir.glob("source_A*.png")):
            anchor_id = image.stem.replace("source_", "")
            images.append(AnchorImage("E3", case_id, anchor_id, image, E3_SCORING_SESSION))

    for session in E45_SESSIONS:
        clean_dir = session / "artifacts/04_anchor_selection/clean_single_anchor_inputs"
        if not clean_dir.exists():
            continue
        old_selected = _load_old_selected_anchor(session)
        for image in sorted(clean_dir.glob("source_A*.png")):
            anchor_id = image.stem.replace("source_", "")
            images.append(AnchorImage("E45", session.name, anchor_id, image, session, old_selected))
    return images


def _case_sort_key(item: AnchorImage) -> tuple[str, str, tuple[int, str]]:
    return item.dataset, item.case_id, _anchor_sort_key(item.anchor_id)


def _select_case(anchor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    survivors = [r for r in anchor_rows if _strict_ok(r)]
    survivors.sort(key=lambda r: (-float(r.get("score") or -math.inf), _anchor_sort_key(str(r.get("anchor_id") or ""))))
    selected = survivors[0] if survivors else None
    ranked = sorted(anchor_rows, key=lambda r: (-float(r.get("score") or -math.inf), _anchor_sort_key(str(r.get("anchor_id") or ""))))
    if not selected:
        return {
            "selected_anchor_id": "NO_SAFE_ANCHOR",
            "selected_score": "",
            "selected_strict_usable": False,
            "ranked_anchors": ",".join(str(r.get("anchor_id")) for r in ranked),
            "rejected_anchor_count": len(anchor_rows),
        }
    return {
        "selected_anchor_id": selected.get("anchor_id"),
        "selected_score": selected.get("score"),
        "selected_strict_usable": True,
        "ranked_anchors": ",".join(str(r.get("anchor_id")) for r in ranked),
        "rejected_anchor_count": len(anchor_rows) - len(survivors),
    }


def _run_iteration(
    *,
    iter_dir: Path,
    iteration_index: int,
    prompt_variant: str,
    image_variant: str,
    prompt_template: str,
    images: list[AnchorImage],
    infer_url: str,
    timeout_sec: int,
    max_new_tokens: int,
) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_template
    (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(
        iter_dir / "prompt_info.json",
        {
            "iteration": iter_dir.name,
            "prompt_variant": prompt_variant,
            "image_variant": image_variant,
            "strict_gate": STRICT_GATE,
            "notes": "",
        },
    )

    per_anchor_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in images:
        case_out = iter_dir / item.dataset / item.case_id / item.anchor_id
        case_out.mkdir(parents=True, exist_ok=True)
        rendered_prompt = prompt.format(anchor_id=item.anchor_id)
        (case_out / "prompt.txt").write_text(rendered_prompt, encoding="utf-8")
        payload = {
            "image_path": str(item.image_path),
            "prompt_text": rendered_prompt,
            "prompt_version": f"{iter_dir.name}_{prompt_variant}",
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "output_path": str(case_out / "raw_response.txt"),
        }
        t0 = time.perf_counter()
        try:
            server_response = _post_json(infer_url, payload, timeout_sec)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            raw_text, parsed, parse_error = _extract_response(server_response)
            normalized, parse_ok, norm_error = _normalize(parsed, item.anchor_id)
            final_error = "" if parse_ok else (norm_error or parse_error)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            server_response = {"ok": False, "error": str(exc)}
            raw_text = ""
            normalized = {"anchor_id": item.anchor_id, "parse_ok": False, "score": "", "overall_usable": False}
            final_error = str(exc)
        normalized.update(
            {
                "dataset": item.dataset,
                "case_id": item.case_id,
                "source_session": str(item.source_session),
                "image_path": str(item.image_path),
                "old_selected_anchor_id": item.old_selected_anchor_id,
                "elapsed_ms": elapsed_ms,
                "parse_error": final_error,
                "strict_gate_passed": _strict_ok(normalized),
                "raw_response_path": str(case_out / "raw_response.txt"),
                "parsed_response_path": str(case_out / "parsed_response.json"),
                "server_response_path": str(case_out / "server_response.json"),
            }
        )
        (case_out / "raw_response.txt").write_text(raw_text or "", encoding="utf-8")
        _write_json(case_out / "server_response.json", server_response)
        _write_json(case_out / "parsed_response.json", normalized)
        row = dict(normalized)
        row["top_defects"] = "; ".join(str(v) for v in normalized.get("top_defects", []))
        row["bottom_defects"] = "; ".join(str(v) for v in normalized.get("bottom_defects", []))
        row.pop("raw_parsed", None)
        per_anchor_rows.append(row)
        grouped.setdefault((item.dataset, item.case_id), []).append(row)

    anchor_fields = [
        "dataset",
        "case_id",
        "anchor_id",
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
        "elapsed_ms",
        "image_path",
        "source_session",
        "old_selected_anchor_id",
        "raw_response_path",
        "parsed_response_path",
        "server_response_path",
    ]
    _write_csv(iter_dir / "per_anchor_results.csv", per_anchor_rows, anchor_fields)

    per_case_rows: list[dict[str, Any]] = []
    for (dataset, case_id), rows in sorted(grouped.items()):
        decision = _select_case(rows)
        old_selected = ""
        for row in rows:
            if row.get("old_selected_anchor_id"):
                old_selected = str(row.get("old_selected_anchor_id"))
                break
        out = {
            "dataset": dataset,
            "case_id": case_id,
            "anchors_attempted": len(rows),
            "anchors_parsed": sum(1 for r in rows if r.get("parse_ok")),
            "old_selected_anchor_id": old_selected,
            **decision,
        }
        per_case_rows.append(out)
        _write_json(iter_dir / dataset / case_id / "case_decision.json", out)
    _write_csv(
        iter_dir / "per_case_decisions.csv",
        per_case_rows,
        [
            "dataset",
            "case_id",
            "anchors_attempted",
            "anchors_parsed",
            "selected_anchor_id",
            "selected_score",
            "selected_strict_usable",
            "ranked_anchors",
            "rejected_anchor_count",
            "old_selected_anchor_id",
        ],
    )


def _check_health(server_url: str) -> dict[str, Any]:
    req = Request(server_url.rstrip("/") + "/health", method="GET")
    with urlopen(req, timeout=10.0) as response:  # noqa: S310 - local server only.
        return json.loads(response.read().decode("utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-url", default="http://127.0.0.1:8899")
    ap.add_argument("--output-root", default="outputs/debug_anchor_prompt_eval")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--timeout-sec", type=int, default=240)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--limit-images", type=int, default=None, help="Debug-only image limit.")
    args = ap.parse_args()

    server_url = args.server_url.rstrip("/")
    infer_url = server_url + "/infer"
    try:
        health = _check_health(server_url)
    except (URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Qwen server is not reachable at {server_url}: {exc}") from exc
    if not health.get("ok") or not health.get("model_loaded"):
        raise SystemExit(f"Qwen server is not ready at {server_url}: {json.dumps(health, indent=2)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_root) / f"iterative_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(discover_images(), key=_case_sort_key)
    if args.limit_images:
        images = images[: args.limit_images]
    if not images:
        raise SystemExit("No saved source_A*.png images found for E3/E45 prompt evaluation.")

    _write_json(
        session_dir / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "server_url": server_url,
            "health": health,
            "image_count": len(images),
            "e3_scoring_session": str(E3_SCORING_SESSION),
            "e3_label_session": str(E3_LABEL_SESSION),
            "e45_sessions": [str(p) for p in E45_SESSIONS],
            "strict_gate": STRICT_GATE,
            "hardware_safety": {
                "robot": "not used",
                "live_camera": "not used",
                "rtde": "not used",
                "arduino": "not used",
                "clamp": "not used",
                "pundit": "not used",
            },
        },
    )
    _write_csv(
        session_dir / "input_image_manifest.csv",
        [item.__dict__ for item in images],
        ["dataset", "case_id", "anchor_id", "image_path", "source_session", "old_selected_anchor_id"],
    )

    count = max(1, min(int(args.iterations), len(PROMPT_VARIANTS)))
    for idx, (variant_name, image_variant, prompt) in enumerate(PROMPT_VARIANTS[:count], start=1):
        iter_dir = session_dir / f"iteration_{idx:02d}"
        print(f"[ITER {idx}/{count}] {variant_name}: {len(images)} images")
        _run_iteration(
            iter_dir=iter_dir,
            iteration_index=idx,
            prompt_variant=variant_name,
            image_variant=image_variant,
            prompt_template=prompt,
            images=images,
            infer_url=infer_url,
            timeout_sec=int(args.timeout_sec),
            max_new_tokens=int(args.max_new_tokens),
        )
        summary = analyze_session(session_dir).get("iterations", [])[-1]
        print(
            f"[ITER {idx}] parse={summary.get('parse_success_rate')} "
            f"e3_strict={summary.get('strict_selected_usable_rate_E3')} "
            f"bad_selected={summary.get('bad_selected_count')} "
            f"no_safe={summary.get('no_safe_anchor_count')}"
        )

    final = analyze_session(session_dir)
    best = final.get("best") or {}
    candidate_cfg = Path("configs/v2/experiments/paper/qwen_anchor_best_prompt_candidate.yaml")
    if (session_dir / "best_prompt.txt").exists():
        prompt_text = (session_dir / "best_prompt.txt").read_text(encoding="utf-8")
        candidate_cfg.parent.mkdir(parents=True, exist_ok=True)
        candidate_cfg.write_text(
            "prompt_candidate:\n"
            "  name: qwen_anchor_contact_strict_gate_candidate_20260528\n"
            f"  source_session: {session_dir}\n"
            f"  best_iteration: {best.get('iteration', '')}\n"
            "  server_endpoint: /infer\n"
            "  strict_gate:\n"
            "    overall_usable: true\n"
            "    top_usable: true\n"
            "    bottom_usable: true\n"
            "    min_score: 70.0\n"
            "  enabled_in_live_pipeline: false\n"
            "  prompt_text: |\n"
            + "\n".join(f"    {line}" for line in prompt_text.splitlines())
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"session_dir": str(session_dir), "best": best}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
