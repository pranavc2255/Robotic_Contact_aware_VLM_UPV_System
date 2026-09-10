from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from upv_vlm_v2.geometry.mask_geometry import (
    add_dominant_rectangle_geometry,
    compute_mask_geometry,
    load_mask,
    save_dominant_rectangle_overlay,
)
from upv_vlm_v2.geometry.overlays import save_axis_overlay


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _find_first(root: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _resolve_inputs(session: Path) -> tuple[Path, Path, Path | None, Path | None]:
    result = _read_json(session / "pipeline_result.json")
    rgb = result.get("target", {}).get("selected_rgb_path") or result.get("capture", {}).get("rgb_path")
    mask = result.get("target", {}).get("selected_mask_path")
    depth = result.get("capture", {}).get("depth_path")
    camera = result.get("capture", {}).get("camera_info_path")
    rgb_path = Path(rgb) if rgb else (_find_first(session, ["raw_rgb.png", "selected_rgb.png", "rgb.png"]) or Path())
    mask_path = Path(mask) if mask else (_find_first(session, ["selected_mask.png", "mask.png"]) or Path())
    if not rgb_path.is_absolute() and not rgb_path.exists():
        rgb_path = session / rgb_path
    if not mask_path.is_absolute() and not mask_path.exists():
        mask_path = session / mask_path
    depth_path = Path(depth) if depth else None
    camera_path = Path(camera) if camera else None
    if not rgb_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Could not resolve saved rgb/mask under {session}: rgb={rgb_path}, mask={mask_path}")
    return rgb_path, mask_path, depth_path, camera_path


def _raw_vs_dominant_overlay(rgb_path: Path, mask_path: Path, geometry: dict[str, Any], out: Path) -> str:
    image = np.array(Image.open(rgb_path).convert("RGB"))
    mask = load_mask(mask_path)
    overlay = image.copy()
    tint = np.zeros_like(overlay)
    tint[:, :] = [40, 220, 120]
    alpha = ((mask > 0).astype(np.float32) * 0.22)[..., None]
    overlay = (overlay.astype(np.float32) * (1.0 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)
    rect = geometry.get("dominant_rectangle") or {}
    corners = rect.get("body_rect_corners_px") if isinstance(rect, dict) else None
    if isinstance(corners, list) and len(corners) >= 4:
        pts = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, (255, 220, 40), 4)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out)
    return str(out)


def compare_session(session: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb, mask, depth, camera = _resolve_inputs(session)
    raw = compute_mask_geometry(mask, depth_path=depth, camera_info_path=camera)
    dominant = add_dominant_rectangle_geometry(
        dict(raw),
        mask,
        {
            "geometry": {
                "body_geometry_mode": "dominant_rectangle",
                "dominant_rectangle_enabled": True,
                "dominant_rectangle_edge_method": "robust_median",
                "dominant_rectangle_min_column_coverage_px": 6,
                "dominant_rectangle_trim_fraction": 0.10,
                "dominant_rectangle_mad_multiplier": 2.5,
            }
        },
    )
    raw_overlay = save_axis_overlay(rgb_path=rgb, mask_path=mask, geometry=raw, output_path=output_dir / "raw_mask_geometry_overlay.png")
    dominant_overlay = save_dominant_rectangle_overlay(rgb_path=rgb, geometry=dominant, output_path=output_dir / "dominant_rectangle_geometry_overlay.png")
    combined = _raw_vs_dominant_overlay(rgb, mask, dominant, output_dir / "raw_vs_dominant_rectangle_overlay.png")
    payload = {
        "session": str(session),
        "rgb_path": str(rgb),
        "mask_path": str(mask),
        "raw_geometry": raw,
        "dominant_rectangle_geometry": dominant.get("dominant_rectangle"),
        "overlays": {
            "raw_mask_geometry_overlay": raw_overlay,
            "dominant_rectangle_geometry_overlay": dominant_overlay,
            "raw_vs_dominant_rectangle_overlay": combined,
        },
    }
    (output_dir / "dominant_rectangle_compare_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw mask geometry and dominant-rectangle geometry on a saved session.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = compare_session(Path(args.session), Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "overlays": payload["overlays"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
