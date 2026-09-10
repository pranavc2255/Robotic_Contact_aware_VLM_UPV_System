#!/usr/bin/env python3
"""
E3 Qwen32 shuffle-consensus experiment.

This script reuses an existing E3 anchor-selection session, takes the already
generated canonical A1..A5 contact-anchor tile images, creates multiple shuffled
display grids, sends each grid to a Qwen server, maps display IDs back to source
anchor IDs, aggregates votes/ranks/scores, and compares the consensus result
against manual labels.

No robot, RTDE, Arduino, clamp, live camera, or dataset capture is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont


DEFAULT_CONFIG: dict[str, Any] = {
    "output_root": "outputs/v2_experiments/e3_qwen32_shuffle_consensus",
    "num_shuffles": 5,
    "shuffle_seed_start": 240424,
    "server_url": "http://127.0.0.1:8899",
    "server_endpoint": "/infer",
    "model_name": "Qwen2.5-VL-32B-Instruct",
    "model_dir": "local_models/Qwen2.5-VL-32B-Instruct",
    "max_new_tokens": 1800,
    "temperature": 0.0,
    "timeout_sec": 240,
    "grid_padding_px": 14,
    "grid_background_rgb": [242, 243, 239],
    "display_header_cover_px": 52,
    "display_header_rgb": [31, 34, 38],
    "display_header_text_rgb": [255, 255, 255],
    "remove_artificial_guide_lines": True,
    "yellow_guide_inpaint_radius_px": 3,
    "use_deterministic_veto_hybrid": True,
    "copy_canonical_tiles": True,
    "fail_fast": False,
    "case_glob": "case_*",
}


PROMPT_TEXT = """You are performing a physical contact-affordance selection task for UPV transducer placement.

The image contains shuffled candidate tiles labeled A1, A2, A3, A4, and A5. Candidate order is randomized between runs, so do not assume any ID is better than another.

Each tile has a TOP contact strip and a BOTTOM contact strip. These highlighted strips are the exact regions where flat UPV transducers will touch.

Judge the highlighted TOP and BOTTOM contact strips first. Do not focus on the rest of the patch unless it affects the strip.

Prefer anchors where BOTH top and bottom contact strips are continuous, uniform, clean, and physically usable.

Penalize any raised or attached material entering the highlighted strip from the edge.
Penalize a vertical seam, crack, line, or material boundary that enters or crosses the strip.
Penalize an abrupt color or texture change at one end of the strip if it suggests a different surface condition, attached material, mortar start, chipped edge, residue, or nonuniform contact.
Penalize gray/white crust, mortar-like buildup, blobs, protrusions, loose debris, chipped sections, broken/jagged edge zones, or any non-planar contact inside the strip.

Material awareness:
- Brick: normal red/brown/orange/tan/beige/cream/yellowish speckled rough clay is not a defect; attached gray/white mortar, crust, debris, or broken edge is a defect.
- Concrete: normal gray aggregate/pores/rough concrete are not defects; protruding paste, debris, broken edge, or local obstruction is a defect.
- Timber: normal grain/knots/saw marks are not defects; nails, tape, splinters, cracks, protruding chips, debris, or blocked contact are defects.

Ranking rules:
- Both TOP and BOTTOM strips must be usable.
- One compromised strip makes the anchor poor.
- If all candidates are imperfect, choose the least bad physically usable contact pair.
- Do not give identical scores/reasons to all anchors unless they are truly visually indistinguishable.
- Every reason must mention the actual visible strip condition.
- Return NO_SAFE_ANCHOR only if every candidate is physically unusable.

Return ONLY one valid JSON object. Do not use markdown fences. Keep reasons short.
Use this compact schema with actual candidate IDs from the image:
{
  "anchors": [
    {"id":"<candidate_id>","defects":["none"],"usable":true,"score":0,"reason":"short physical-contact reason"}
  ],
  "ranking_best_to_worst":["<candidate_id>"],
  "best_anchor":"<candidate_id>",
  "best_anchor_reason":"short reason",
  "no_safe_anchor":false
}
"""


@dataclass
class CaseInfo:
    case_dir: Path
    case_name: str
    case_id: str
    case_index: int | None
    material: str
    condition: str
    artifact_dir: Path
    tile_dir: Path


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_config(path: Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as dict: {path}")

    cfg.update(data)
    return cfg


def infer_case_id_from_dir(case_dir: Path) -> tuple[int | None, str]:
    name = case_dir.name
    m = re.match(r"case_(\d+)_(.+)$", name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, name


def infer_material_condition(case_id: str) -> tuple[str, str]:
    if case_id.startswith("concrete_block_"):
        parts = case_id.split("_")
        material = "concrete_block"
        condition = "_".join(parts[3:]) if len(parts) > 3 else ""
        return material, condition

    parts = case_id.split("_")
    material = parts[0] if parts else ""
    condition = "_".join(parts[2:]) if len(parts) > 2 else ""
    return material, condition


def find_latest_artifact_dir(case_dir: Path) -> Path | None:
    dirs = sorted(case_dir.glob("pipeline_session/session_*/artifacts/04_anchor_selection"))
    if not dirs:
        return None
    return dirs[-1]


def discover_cases(session_dir: Path, case_glob: str = "case_*") -> list[CaseInfo]:
    cases_root = session_dir / "cases"
    if not cases_root.exists():
        raise FileNotFoundError(f"Missing cases dir: {cases_root}")

    cases: list[CaseInfo] = []

    for case_dir in sorted(cases_root.glob(case_glob)):
        if not case_dir.is_dir():
            continue

        idx, case_id = infer_case_id_from_dir(case_dir)
        material, condition = infer_material_condition(case_id)

        artifact_dir = find_latest_artifact_dir(case_dir)
        if artifact_dir is None:
            print(f"[WARN] Skipping {case_dir.name}: no artifact dir", file=sys.stderr)
            continue

        tile_dir = artifact_dir / "boxed_wide_contact_guided_tiles"
        if not tile_dir.exists():
            print(f"[WARN] Skipping {case_dir.name}: no tile dir {tile_dir}", file=sys.stderr)
            continue

        cases.append(
            CaseInfo(
                case_dir=case_dir,
                case_name=case_dir.name,
                case_id=case_id,
                case_index=idx,
                material=material,
                condition=condition,
                artifact_dir=artifact_dir,
                tile_dir=tile_dir,
            )
        )

    return cases


def load_manual_labels(path: Path | None) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    labels: dict[tuple[str, str], str] = {}
    manual_best: dict[str, str] = {}

    if path is None or not path.exists():
        return labels, manual_best

    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = (r.get("case_id") or "").strip()
            aid = (r.get("anchor_id") or "").strip()
            label = (r.get("manual_label") or r.get("label") or "").strip()

            if cid and aid:
                labels[(cid, aid)] = label

            is_best = str(r.get("is_manual_best") or r.get("manual_best") or "").strip().lower()
            if cid and aid and is_best in {"true", "1", "yes", "y"}:
                manual_best[cid] = aid

    return labels, manual_best


def anchor_sort_key(anchor_id: str) -> tuple[int, str]:
    m = re.match(r"A(\d+)$", anchor_id)
    if m:
        return int(m.group(1)), anchor_id
    return 10**9, anchor_id


def canonical_tile_paths(tile_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    for p in tile_dir.glob("candidate_A*_wide_contact_guided_tile.png"):
        m = re.search(r"candidate_(A\d+)_wide_contact_guided_tile\.png$", p.name)
        if m:
            paths[m.group(1)] = p

    return dict(sorted(paths.items(), key=lambda kv: anchor_sort_key(kv[0])))


def make_shuffle_mapping(source_ids: list[str], seed: int) -> dict[str, Any]:
    display_ids = [f"A{i+1}" for i in range(len(source_ids))]
    shuffled_sources = list(source_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled_sources)

    display_to_source = dict(zip(display_ids, shuffled_sources))
    source_to_display = {src: disp for disp, src in display_to_source.items()}

    return {
        "seed": seed,
        "display_ids": display_ids,
        "source_ids": source_ids,
        "display_to_source": display_to_source,
        "source_to_display": source_to_display,
    }


def get_font(size: int = 22) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]

    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            pass

    return ImageFont.load_default()



def remove_artificial_guide_lines(im: Image.Image, cfg: dict[str, Any]) -> Image.Image:
    """Remove artificial yellow and blue/cyan guide lines from a tile.

    This is only for VLM input grids. Original source artifacts are not changed.
    """
    if not bool(cfg.get("remove_yellow_guide_lines", True)):
        return im

    try:
        import numpy as np  # type: ignore

        arr = np.array(im.convert("RGB"))
        r = arr[:, :, 0].astype("int16")
        g = arr[:, :, 1].astype("int16")
        b = arr[:, :, 2].astype("int16")

        yellow_mask = (
            (r > 170)
            & (g > 130)
            & (b < 130)
            & ((r - b) > 70)
            & ((g - b) > 50)
        )

        blue_mask = (
            (b > 140)
            & (g > 80)
            & (r < 120)
            & ((b - r) > 60)
        )

        cyan_mask = (
            (b > 120)
            & (g > 120)
            & (r < 120)
            & ((b - r) > 40)
            & ((g - r) > 40)
        )

        mask = (yellow_mask | blue_mask | cyan_mask).astype("uint8") * 255

        if mask.sum() == 0:
            return im

        radius = int(cfg.get("yellow_guide_inpaint_radius_px", 3))

        try:
            import cv2  # type: ignore

            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.dilate(mask, kernel, iterations=1)
            cleaned = cv2.inpaint(arr, mask2, radius, cv2.INPAINT_TELEA)
            return Image.fromarray(cleaned)
        except Exception:
            keep = mask == 0
            if keep.any():
                med = np.median(arr[keep], axis=0).astype("uint8")
                arr[mask > 0] = med
            return Image.fromarray(arr)

    except Exception:
        return im


def _guide_mask_rgb(arr: Any) -> Any:
    import numpy as np  # type: ignore

    r = arr[:, :, 0].astype("int16")
    g = arr[:, :, 1].astype("int16")
    b = arr[:, :, 2].astype("int16")

    yellow_mask = (
        (r > 170)
        & (g > 130)
        & (b < 130)
        & ((r - b) > 70)
        & ((g - b) > 50)
    )
    blue_mask = (
        (b > 140)
        & (g > 80)
        & (r < 120)
        & ((b - r) > 60)
    )
    cyan_mask = (
        (b > 120)
        & (g > 120)
        & (r < 120)
        & ((b - r) > 40)
        & ((g - r) > 40)
    )
    return yellow_mask | blue_mask | cyan_mask


def _detect_contact_strip_centers(tile: Image.Image) -> tuple[int, int]:
    import numpy as np  # type: ignore

    arr = np.array(tile.convert("RGB"))
    mask = _guide_mask_rgb(arr)
    h = arr.shape[0]
    mid = h // 2

    def _center(submask: Any, fallback: int) -> int:
        ys = np.where(submask)[0]
        if ys.size:
            return int(round(float(np.median(ys))))
        return fallback

    top_y = _center(mask[:mid], max(0, int(round(h * 0.28))))
    bottom_y = _center(mask[mid:], min(h - 1, int(round(h * 0.72))))
    if bottom_y < mid:
        bottom_y = min(h - 1, int(round(h * 0.72)))
    return top_y, bottom_y


def _overlay_contact_strip_band(im: Image.Image, center_y: int, *, band_rgb: tuple[int, int, int], label: str) -> None:
    band_h = max(24, int(round(im.height * 0.07)))
    half_h = max(10, band_h // 2)
    y0 = max(0, center_y - half_h)
    y1 = min(im.height - 1, center_y + half_h)

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (*band_rgb, 82)
    outline = (*band_rgb, 220)
    draw.rectangle((0, y0, im.width - 1, y1), fill=fill, outline=outline, width=2)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    try:
        text_w = draw.textlength(label, font=font)
    except Exception:
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = float(text_bbox[2] - text_bbox[0])
    text_x = 8
    text_y = max(0, y0 + 2)
    pad_x = 4
    pad_y = 2
    draw.rectangle(
        (text_x - pad_x, text_y - pad_y, min(im.width - 1, int(text_x + text_w + pad_x)), min(im.height - 1, text_y + 14 + pad_y)),
        fill=(0, 0, 0, 110),
        outline=(255, 255, 255, 170),
        width=1,
    )
    draw.text((text_x, text_y), label, fill=(255, 255, 255, 240), font=font)

    merged = Image.alpha_composite(im.convert("RGBA"), overlay)
    im.paste(merged.convert("RGB"))


def relabel_tile_for_display(source_tile_path: Path, display_id: str, cfg: dict[str, Any]) -> Image.Image:
    im = Image.open(source_tile_path).convert("RGB")
    top_y, bottom_y = _detect_contact_strip_centers(im)
    _overlay_contact_strip_band(im, top_y, band_rgb=(255, 193, 7), label="TOP CONTACT STRIP")
    _overlay_contact_strip_band(im, bottom_y, band_rgb=(33, 150, 243), label="BOTTOM CONTACT STRIP")
    draw = ImageDraw.Draw(im)

    cover_h = int(cfg.get("display_header_cover_px", 52))
    header_rgb = tuple(int(x) for x in cfg.get("display_header_rgb", [31, 34, 38]))
    text_rgb = tuple(int(x) for x in cfg.get("display_header_text_rgb", [255, 255, 255]))

    draw.rectangle((0, 0, im.width - 1, min(cover_h, im.height) - 1), fill=header_rgb)
    draw.text((10, 9), display_id, fill=text_rgb, font=get_font(24))

    return im


def build_grid_for_mapping(
    tile_paths: dict[str, Path],
    mapping: dict[str, Any],
    out_path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    padding = int(cfg.get("grid_padding_px", 14))
    bg = tuple(int(x) for x in cfg.get("grid_background_rgb", [242, 243, 239]))

    display_to_source = mapping["display_to_source"]
    display_ids = mapping["display_ids"]

    display_tiles: list[tuple[str, str, Image.Image]] = []

    for display_id in display_ids:
        source_id = display_to_source[display_id]
        im = relabel_tile_for_display(tile_paths[source_id], display_id, cfg)
        display_tiles.append((display_id, source_id, im))

    width = padding + sum(im.width + padding for _, _, im in display_tiles)
    height = max(im.height for _, _, im in display_tiles) + 2 * padding

    canvas = Image.new("RGB", (width, height), bg)

    x = padding
    placements = []

    for display_id, source_id, im in display_tiles:
        y = padding
        canvas.paste(im, (x, y))

        placements.append(
            {
                "display_id": display_id,
                "source_anchor_id": source_id,
                "x": x,
                "y": y,
                "width": im.width,
                "height": im.height,
            }
        )

        x += im.width + padding

    canvas.save(out_path)

    return {
        "grid_image_path": str(out_path),
        "grid_size_px": [width, height],
        "placements": placements,
    }


def post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")

    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    parsed.setdefault("_raw_http_body", body)
                    return parsed
            except Exception:
                pass

            return {"ok": True, "text": body, "_raw_http_body": body}

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"HTTPError {exc.code}", "status": exc.code, "_raw_http_body": body}
    except URLError as exc:
        return {"ok": False, "error": f"URLError {exc}"}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def strip_markdown_json(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.I)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_json_from_text(text: str) -> tuple[dict[str, Any] | None, str]:
    if not text:
        return None, "empty raw text"

    s = strip_markdown_json(text)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, ""
    except Exception as exc:
        direct_err = str(exc)

    m = re.search(r"\{.*\}", s, flags=re.S)

    if not m:
        return None, f"no JSON object found; direct parse error: {direct_err}"

    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj, ""
        return None, "extracted JSON was not object"
    except Exception as exc:
        return None, f"extracted JSON parse error: {exc}"


def extract_response(response: dict[str, Any]) -> tuple[str, dict[str, Any] | None, bool, str]:
    raw = (
        response.get("raw_response")
        or response.get("raw_text")
        or response.get("text")
        or response.get("response")
        or ""
    )

    if bool(response.get("parse_ok")) and isinstance(response.get("parsed_json"), dict):
        return str(raw), response["parsed_json"], True, ""

    if "anchors" in response and ("best_anchor" in response or "ranking_best_to_worst" in response):
        return json.dumps(response, indent=2), response, True, ""

    if isinstance(response.get("parsed_json"), dict):
        return str(raw), response["parsed_json"], True, ""

    if not raw and response.get("_raw_http_body"):
        raw = str(response.get("_raw_http_body"))

    obj, err = parse_json_from_text(str(raw))

    if obj is not None:
        return str(raw), obj, True, ""

    err2 = str(response.get("parse_error") or response.get("error") or err or "parse failed")
    return str(raw), None, False, err2


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def normalize_ranking_from_scores(parsed: dict[str, Any]) -> list[str]:
    anchors = parsed.get("anchors") or []
    scored = []

    if isinstance(anchors, list):
        for a in anchors:
            if not isinstance(a, dict):
                continue

            aid = str(a.get("id") or "").strip()

            if not aid:
                continue

            score = safe_float(a.get("overall_score"))
            if score is None:
                score = safe_float(a.get("score"))
            scored.append((aid, score if score is not None else -math.inf))

    scored.sort(key=lambda x: (-x[1], anchor_sort_key(x[0])))

    return [aid for aid, _ in scored]


def remap_qwen_result(parsed: dict[str, Any] | None, mapping: dict[str, Any]) -> dict[str, Any]:
    if parsed is None:
        return {
            "parse_ok": False,
            "source_best": None,
            "source_ranking": [],
            "source_scores": {},
        }

    display_to_source = mapping["display_to_source"]

    display_best = str(parsed.get("best_anchor") or "").strip()
    source_best = display_to_source.get(display_best)

    ranking = parsed.get("ranking_best_to_worst") or parsed.get("ranked_anchor_ids") or []

    if not isinstance(ranking, list) or not ranking:
        ranking = normalize_ranking_from_scores(parsed)

    display_ranking = [str(x).strip() for x in ranking if str(x).strip()]
    source_ranking = [display_to_source[x] for x in display_ranking if x in display_to_source]

    source_scores: dict[str, float] = {}
    source_anchor_details: dict[str, Any] = {}

    for a in parsed.get("anchors") or []:
        if not isinstance(a, dict):
            continue

        display_id = str(a.get("id") or "").strip()
        source_id = display_to_source.get(display_id)

        if not source_id:
            continue

        score = safe_float(a.get("overall_score"))
        if score is None:
            score = safe_float(a.get("score"))

        if score is not None:
            source_scores[source_id] = score

        source_anchor_details[source_id] = {
            "display_id": display_id,
            "top_score": a.get("top_score"),
            "bottom_score": a.get("bottom_score"),
            "overall_score": a.get("overall_score", a.get("score")),
            "score": a.get("score", a.get("overall_score")),
            "issues": a.get("issues"),
            "defects": a.get("defects"),
            "usable": a.get("usable"),
            "reason": a.get("reason"),
        }

    return {
        "parse_ok": True,
        "display_best": display_best or None,
        "source_best": source_best,
        "display_ranking": display_ranking,
        "source_ranking": source_ranking,
        "source_scores": source_scores,
        "source_anchor_details": source_anchor_details,
        "best_anchor_reason": parsed.get("best_anchor_reason"),
    }


def load_deterministic_features(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    p = artifact_dir / "deterministic_anchor_features_wide_context.json"
    data = read_json(p, default=None)

    features: dict[str, dict[str, Any]] = {}

    anchors = None

    if isinstance(data, dict):
        if isinstance(data.get("anchors"), list):
            anchors = data["anchors"]
        elif isinstance(data.get("anchor_features"), list):
            anchors = data["anchor_features"]
    elif isinstance(data, list):
        anchors = data

    if not isinstance(anchors, list):
        return features

    for a in anchors:
        if not isinstance(a, dict):
            continue

        anchor_id = str(a.get("anchor_id") or a.get("id") or "").strip()

        if not anchor_id:
            continue

        features[anchor_id] = {
            "veto": bool(a.get("veto")),
            "veto_reasons": a.get("veto_reasons") or [],
            "deterministic_score": safe_float(a.get("deterministic_score")),
            "valid": a.get("valid"),
        }

    return features


def aggregate_consensus(
    source_ids: list[str],
    remapped_results: list[dict[str, Any]],
    deterministic_features: dict[str, dict[str, Any]] | None = None,
    use_deterministic_veto_hybrid: bool = True,
) -> dict[str, Any]:
    deterministic_features = deterministic_features or {}

    valid = [r for r in remapped_results if r.get("parse_ok") and r.get("source_ranking")]
    num_valid = len(valid)

    vote_counts: Counter[str] = Counter()
    rank_lists: dict[str, list[float]] = {aid: [] for aid in source_ids}
    score_lists: dict[str, list[float]] = {aid: [] for aid in source_ids}

    for r in valid:
        best = r.get("source_best")

        if best in source_ids:
            vote_counts[best] += 1

        ranking = r.get("source_ranking") or []
        rank_map = {aid: i + 1 for i, aid in enumerate(ranking)}
        missing_rank = len(source_ids) + 1

        for aid in source_ids:
            rank_lists[aid].append(float(rank_map.get(aid, missing_rank)))

        scores = r.get("source_scores") or {}

        for aid, score in scores.items():
            if aid in score_lists:
                sf = safe_float(score)
                if sf is not None:
                    score_lists[aid].append(sf)

    rows = []

    for aid in source_ids:
        votes = vote_counts.get(aid, 0)
        avg_rank = statistics.mean(rank_lists[aid]) if rank_lists[aid] else math.inf
        mean_score = statistics.mean(score_lists[aid]) if score_lists[aid] else -math.inf
        det = deterministic_features.get(aid, {})

        rows.append(
            {
                "anchor_id": aid,
                "vote_count": votes,
                "avg_rank": avg_rank,
                "mean_score": mean_score,
                "deterministic_veto": bool(det.get("veto", False)),
                "deterministic_score": det.get("deterministic_score"),
                "veto_reasons": det.get("veto_reasons") or [],
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple:
        mean_score = float(row["mean_score"]) if math.isfinite(float(row["mean_score"])) else -999.0

        return (
            -int(row["vote_count"]),
            float(row["avg_rank"]) if math.isfinite(float(row["avg_rank"])) else 999.0,
            -mean_score,
            anchor_sort_key(row["anchor_id"]),
        )

    ranked_qwen = sorted(rows, key=sort_key) if num_valid > 0 else []
    qwen_selected = ranked_qwen[0]["anchor_id"] if ranked_qwen else None

    non_veto = [r for r in ranked_qwen if not r.get("deterministic_veto")]

    if num_valid == 0:
        hybrid_selected = None
        hybrid_mode = "no_parse_result"
    elif use_deterministic_veto_hybrid and non_veto:
        hybrid_selected = non_veto[0]["anchor_id"]
        hybrid_mode = "qwen32_consensus_among_non_vetoed"
    elif use_deterministic_veto_hybrid and not non_veto and ranked_qwen:
        hybrid_selected = ranked_qwen[0]["anchor_id"]
        hybrid_mode = "all_vetoed_used_qwen32_consensus_low_confidence"
    else:
        hybrid_selected = qwen_selected
        hybrid_mode = "qwen32_consensus_only"

    top_votes = ranked_qwen[0]["vote_count"] if ranked_qwen else 0
    second_votes = ranked_qwen[1]["vote_count"] if len(ranked_qwen) > 1 else 0

    return {
        "num_valid_shuffles": num_valid,
        "num_total_shuffles": len(remapped_results),
        "vote_table": rows,
        "qwen32_ranked_source_anchors": [r["anchor_id"] for r in ranked_qwen],
        "qwen32_selected_anchor": qwen_selected if qwen_selected is not None else "NO_PARSE_RESULT",
        "hybrid_selected_anchor": hybrid_selected if hybrid_selected is not None else "NO_PARSE_RESULT",
        "hybrid_mode": hybrid_mode,
        "vote_margin": top_votes - second_votes,
        "vote_fraction": top_votes / num_valid if num_valid else 0.0,
        "parse_failure": num_valid == 0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields = []
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in rows:
            writer.writerow(r)


def write_vote_csv(path: Path, vote_table: list[dict[str, Any]]) -> None:
    fields = [
        "anchor_id",
        "vote_count",
        "avg_rank",
        "mean_score",
        "deterministic_veto",
        "deterministic_score",
        "veto_reasons",
    ]

    rows = []

    for r in vote_table:
        rr = dict(r)
        rr["veto_reasons"] = "; ".join(str(x) for x in rr.get("veto_reasons") or [])
        rows.append(rr)

    write_csv(path, rows, fields)


def manual_eval(
    case_id: str,
    selected_anchor: str | None,
    labels: dict[tuple[str, str], str],
    manual_best: dict[str, str],
) -> dict[str, Any]:
    label = labels.get((case_id, selected_anchor or ""), "")
    if selected_anchor in {None, "", "NO_PARSE_RESULT"}:
        label = ""
    label_norm = label.strip().lower()
    best_anchor = manual_best.get(case_id, "")

    return {
        "manual_label": label,
        "manual_best_anchor": best_anchor,
        "is_top1_usable": label_norm in {"good", "acceptable", "accept", "ok"},
        "is_manual_best_match": bool(selected_anchor and best_anchor and selected_anchor == best_anchor),
        "bad_anchor_selected": label_norm == "bad",
    }


def summarize_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    n = len(rows)

    if n == 0:
        return {}

    usable = sum(str(r.get(f"{prefix}_is_top1_usable")).lower() == "true" for r in rows)
    best = sum(str(r.get(f"{prefix}_is_manual_best_match")).lower() == "true" for r in rows)
    bad = sum(str(r.get(f"{prefix}_bad_anchor_selected")).lower() == "true" for r in rows)
    parsed = sum(int(r.get("num_valid_shuffles") or 0) > 0 for r in rows)

    return {
        "n_cases": n,
        "cases_with_any_parse_success": parsed,
        "case_parse_success_rate": parsed / n,
        "top1_usable_count": usable,
        "top1_usable_rate": usable / n,
        "manual_best_match_count": best,
        "manual_best_match_rate": best / n,
        "bad_anchor_selected_count": bad,
        "bad_anchor_selected_rate": bad / n,
    }


def group_summary(rows: list[dict[str, Any]], group_field: str, prefix: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        groups[str(r.get(group_field) or "")].append(r)

    out = []

    for group_name, group_rows in sorted(groups.items()):
        s = summarize_rows(group_rows, prefix)
        s[group_field] = group_name
        out.append(s)

    return out


def run_case(
    case: CaseInfo,
    output_case_dir: Path,
    cfg: dict[str, Any],
    labels: dict[tuple[str, str], str],
    manual_best: dict[str, str],
) -> dict[str, Any]:
    output_case_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = canonical_tile_paths(case.tile_dir)

    if not tile_paths:
        raise RuntimeError(f"No canonical tiles found: {case.tile_dir}")

    source_ids = list(tile_paths.keys())

    if bool(cfg.get("copy_canonical_tiles", True)):
        canonical_dir = output_case_dir / "canonical_tiles"
        canonical_dir.mkdir(parents=True, exist_ok=True)

        for _, p in tile_paths.items():
            shutil.copy2(p, canonical_dir / p.name)

    write_json(
        output_case_dir / "case_input_manifest.json",
        {
            "case_name": case.case_name,
            "case_id": case.case_id,
            "case_index": case.case_index,
            "material": case.material,
            "condition": case.condition,
            "source_artifact_dir": str(case.artifact_dir),
            "source_tile_dir": str(case.tile_dir),
            "source_ids": source_ids,
            "model_name": cfg.get("model_name"),
            "model_dir": cfg.get("model_dir"),
        },
    )

    (output_case_dir / "qwen32_prompt.txt").write_text(PROMPT_TEXT, encoding="utf-8")

    num_shuffles = int(cfg.get("num_shuffles", 5))
    seed_start = int(cfg.get("shuffle_seed_start", 240424))
    case_seed_base = seed_start + (case.case_index or 0) * 1000

    server_url = str(cfg.get("server_url", "http://127.0.0.1:8899")).rstrip("/")
    endpoint = str(cfg.get("server_endpoint", "/infer"))
    infer_url = server_url + endpoint
    timeout_sec = int(cfg.get("timeout_sec", 240))

    remapped_results = []
    shuffle_rows = []

    for shuffle_index in range(1, num_shuffles + 1):
        shuffle_dir = output_case_dir / "shuffled_grids"
        response_dir = output_case_dir / "qwen32_responses"
        shuffle_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)

        seed = case_seed_base + shuffle_index
        mapping = make_shuffle_mapping(source_ids, seed)
        mapping["shuffle_index"] = shuffle_index

        grid_path = shuffle_dir / f"shuffle_{shuffle_index}_grid.png"
        grid_meta = build_grid_for_mapping(tile_paths, mapping, grid_path, cfg)
        mapping.update(grid_meta)

        mapping_path = shuffle_dir / f"shuffle_{shuffle_index}_mapping.json"
        write_json(mapping_path, mapping)

        server_response_path = response_dir / f"shuffle_{shuffle_index}_server_response.json"

        payload = {
            "image_path": str(grid_path),
            "prompt_text": PROMPT_TEXT,
            "prompt_version": "e3_qwen32_shuffle_upv_contact_affordance_compact_v3",
            "max_new_tokens": int(cfg.get("max_new_tokens", 900)),
            "temperature": float(cfg.get("temperature", 0.0)),
            "output_path": str(server_response_path),
        }

        t0 = time.perf_counter()
        response = post_json(infer_url, payload, timeout_sec)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        write_json(server_response_path, response)

        raw_text, parsed, parse_ok, parse_error = extract_response(response)

        raw_path = response_dir / f"shuffle_{shuffle_index}_raw.txt"
        parsed_path = response_dir / f"shuffle_{shuffle_index}_parsed.json"
        remapped_path = response_dir / f"shuffle_{shuffle_index}_remapped.json"

        raw_path.write_text(raw_text or "", encoding="utf-8")

        if parsed is not None:
            write_json(parsed_path, parsed)
        else:
            write_json(parsed_path, {"parse_ok": False, "parse_error": parse_error})

        remapped = remap_qwen_result(parsed, mapping)
        remapped["shuffle_index"] = shuffle_index
        remapped["parse_ok"] = bool(parse_ok and remapped.get("parse_ok"))
        remapped["parse_error"] = "" if remapped["parse_ok"] else parse_error
        remapped["elapsed_ms"] = elapsed_ms
        remapped["grid_image_path"] = str(grid_path)
        remapped["mapping_path"] = str(mapping_path)
        remapped["raw_response_path"] = str(raw_path)
        remapped["parsed_response_path"] = str(parsed_path)

        write_json(remapped_path, remapped)

        remapped_results.append(remapped)

        shuffle_rows.append(
            {
                "case_id": case.case_id,
                "shuffle_index": shuffle_index,
                "parse_ok": remapped["parse_ok"],
                "parse_error": remapped["parse_error"],
                "display_best": remapped.get("display_best"),
                "source_best": remapped.get("source_best"),
                "source_ranking": "|".join(remapped.get("source_ranking") or []),
                "elapsed_ms": f"{elapsed_ms:.3f}",
                "grid_image_path": str(grid_path),
            }
        )

    deterministic_features = load_deterministic_features(case.artifact_dir)

    consensus = aggregate_consensus(
        source_ids,
        remapped_results,
        deterministic_features=deterministic_features,
        use_deterministic_veto_hybrid=bool(cfg.get("use_deterministic_veto_hybrid", True)),
    )

    consensus_dir = output_case_dir / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    write_vote_csv(consensus_dir / "qwen32_vote_table.csv", consensus["vote_table"])

    write_json(
        consensus_dir / "qwen32_consensus_decision.json",
        {
            **consensus,
            "case_id": case.case_id,
            "case_name": case.case_name,
            "material": case.material,
            "condition": case.condition,
            "remapped_results": remapped_results,
        },
    )

    write_csv(output_case_dir / "shuffle_results.csv", shuffle_rows)

    qwen_eval = manual_eval(case.case_id, consensus.get("qwen32_selected_anchor"), labels, manual_best)
    hybrid_eval = manual_eval(case.case_id, consensus.get("hybrid_selected_anchor"), labels, manual_best)

    result = {
        "case_name": case.case_name,
        "case_id": case.case_id,
        "case_index": case.case_index,
        "material": case.material,
        "condition": case.condition,
        "num_source_anchors": len(source_ids),
        "num_total_shuffles": consensus["num_total_shuffles"],
        "num_valid_shuffles": consensus["num_valid_shuffles"],
        "parse_failure": consensus["parse_failure"],
        "vote_margin": consensus["vote_margin"],
        "vote_fraction": consensus["vote_fraction"],
        "qwen32_selected_anchor": consensus.get("qwen32_selected_anchor"),
        "qwen32_ranked_source_anchors": "|".join(consensus.get("qwen32_ranked_source_anchors") or []),
        "qwen32_manual_label": qwen_eval["manual_label"],
        "qwen32_manual_best_anchor": qwen_eval["manual_best_anchor"],
        "qwen32_is_top1_usable": qwen_eval["is_top1_usable"],
        "qwen32_is_manual_best_match": qwen_eval["is_manual_best_match"],
        "qwen32_bad_anchor_selected": qwen_eval["bad_anchor_selected"],
        "hybrid_selected_anchor": consensus.get("hybrid_selected_anchor"),
        "hybrid_mode": consensus.get("hybrid_mode"),
        "hybrid_manual_label": hybrid_eval["manual_label"],
        "hybrid_manual_best_anchor": hybrid_eval["manual_best_anchor"],
        "hybrid_is_top1_usable": hybrid_eval["is_top1_usable"],
        "hybrid_is_manual_best_match": hybrid_eval["is_manual_best_match"],
        "hybrid_bad_anchor_selected": hybrid_eval["bad_anchor_selected"],
        "case_output_dir": str(output_case_dir),
    }

    write_json(output_case_dir / "case_result.json", result)

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="Existing E3 anchor-selection session dir")
    ap.add_argument("--manual-labels", default="", help="manual_anchor_labels.csv")
    ap.add_argument("--config", default="", help="YAML config")
    ap.add_argument("--server-url", default="", help="Override server URL")
    ap.add_argument("--num-shuffles", type=int, default=None)
    ap.add_argument("--output-root", default="")
    ap.add_argument("--case-id", default="", help="Optional comma-separated case IDs")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    if args.server_url:
        cfg["server_url"] = args.server_url
    if args.num_shuffles is not None:
        cfg["num_shuffles"] = args.num_shuffles
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.fail_fast:
        cfg["fail_fast"] = True

    session_dir = Path(args.session)

    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    manual_labels_path = Path(args.manual_labels) if args.manual_labels else None
    labels, manual_best = load_manual_labels(manual_labels_path)

    output_session = Path(str(cfg["output_root"])) / f"session_{now_stamp()}"
    output_session.mkdir(parents=True, exist_ok=False)

    selected_case_ids = {x.strip() for x in args.case_id.split(",") if x.strip()}

    cases = discover_cases(session_dir, str(cfg.get("case_glob", "case_*")))

    if selected_case_ids:
        cases = [c for c in cases if c.case_id in selected_case_ids]

    write_json(
        output_session / "run_manifest.json",
        {
            "phase": "e3_qwen32_shuffle_consensus_from_existing_session",
            "created": datetime.now().isoformat(),
            "input_session": str(session_dir),
            "manual_labels": str(manual_labels_path) if manual_labels_path else None,
            "output_session": str(output_session),
            "config": cfg,
            "num_cases": len(cases),
            "case_ids": [c.case_id for c in cases],
        },
    )

    write_json(output_session / "effective_config.json", cfg)

    print("output_session:", output_session)
    print("num_cases:", len(cases))
    print("server_url:", cfg.get("server_url"))
    print("model_dir:", cfg.get("model_dir"))

    rows = []
    errors = []

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.case_id}")

        output_case_dir = output_session / "cases" / case.case_name

        try:
            row = run_case(case, output_case_dir, cfg, labels, manual_best)
            rows.append(row)

            print(
                "  qwen32=",
                row["qwen32_selected_anchor"],
                row["qwen32_manual_label"],
                "hybrid=",
                row["hybrid_selected_anchor"],
                row["hybrid_manual_label"],
                "valid_shuffles=",
                row["num_valid_shuffles"],
            )

        except Exception as exc:
            err = {
                "case_id": case.case_id,
                "case_name": case.case_name,
                "error": repr(exc),
                "case_output_dir": str(output_case_dir),
            }

            errors.append(err)
            print("[ERROR]", err, file=sys.stderr)

            if bool(cfg.get("fail_fast", False)):
                raise

    master_fields = [
        "case_name",
        "case_id",
        "case_index",
        "material",
        "condition",
        "num_source_anchors",
        "num_total_shuffles",
        "num_valid_shuffles",
        "parse_failure",
        "vote_margin",
        "vote_fraction",
        "qwen32_selected_anchor",
        "qwen32_ranked_source_anchors",
        "qwen32_manual_label",
        "qwen32_manual_best_anchor",
        "qwen32_is_top1_usable",
        "qwen32_is_manual_best_match",
        "qwen32_bad_anchor_selected",
        "hybrid_selected_anchor",
        "hybrid_mode",
        "hybrid_manual_label",
        "hybrid_manual_best_anchor",
        "hybrid_is_top1_usable",
        "hybrid_is_manual_best_match",
        "hybrid_bad_anchor_selected",
        "case_output_dir",
    ]

    write_csv(output_session / "master_qwen32_shuffle_results.csv", rows, master_fields)
    write_csv(output_session / "errors.csv", errors, ["case_id", "case_name", "error", "case_output_dir"])

    summary = {
        "phase": "e3_qwen32_shuffle_consensus_from_existing_session",
        "input_session": str(session_dir),
        "manual_labels": str(manual_labels_path) if manual_labels_path else None,
        "output_session": str(output_session),
        "n_cases": len(rows),
        "n_errors": len(errors),
        "qwen32": summarize_rows(rows, "qwen32"),
        "hybrid": summarize_rows(rows, "hybrid"),
    }

    write_json(output_session / "summary_overall.json", summary)
    write_csv(output_session / "summary_by_material_qwen32.csv", group_summary(rows, "material", "qwen32"))
    write_csv(output_session / "summary_by_condition_qwen32.csv", group_summary(rows, "condition", "qwen32"))
    write_csv(output_session / "summary_by_material_hybrid.csv", group_summary(rows, "material", "hybrid"))
    write_csv(output_session / "summary_by_condition_hybrid.csv", group_summary(rows, "condition", "hybrid"))

    paper = output_session / "paper_ready_e3_qwen32_results"
    paper.mkdir(exist_ok=True)

    for name in [
        "master_qwen32_shuffle_results.csv",
        "summary_overall.json",
        "summary_by_material_qwen32.csv",
        "summary_by_condition_qwen32.csv",
        "summary_by_material_hybrid.csv",
        "summary_by_condition_hybrid.csv",
    ]:
        src = output_session / name
        if src.exists():
            shutil.copy2(src, paper / name)

    print("\nDONE")
    print("output_session:", output_session)
    print("summary:", output_session / "summary_overall.json")
    print("master:", output_session / "master_qwen32_shuffle_results.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
