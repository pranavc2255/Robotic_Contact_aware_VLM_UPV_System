from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


ROBOT_MOTION_BLOCKED_REASON = (
    "This output is perception geometry only. Robot motion requires valid safe anchor, depth projection, "
    "camera-to-tool calibration, tool-to-base transform, workspace validation, and explicit operator confirmation."
)


def _timestamp_now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _point_list(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if _is_number(value.get("u")) and _is_number(value.get("v")):
            return [float(value["u"]), float(value["v"])]
        if _is_number(value.get("x")) and _is_number(value.get("y")):
            return [float(value["x"]), float(value["y"])]
    if isinstance(value, (list, tuple)) and len(value) >= 2 and _is_number(value[0]) and _is_number(value[1]):
        return [float(value[0]), float(value[1])]
    return None


def _anchor_sort_key(anchor_id: str) -> tuple[int, str]:
    if isinstance(anchor_id, str) and len(anchor_id) > 1 and anchor_id[0].upper() == "A" and anchor_id[1:].isdigit():
        return int(anchor_id[1:]), anchor_id
    return 10_000, str(anchor_id)


def _axis_for_mode(geometry: dict[str, Any] | None, axis_mode: str) -> tuple[list[float] | None, float | None]:
    if not isinstance(geometry, dict):
        return None, None
    if axis_mode == "major":
        vector = geometry.get("major_axis_vector")
        length = geometry.get("major_axis_length_px")
    elif axis_mode == "minor":
        vector = geometry.get("minor_axis_vector")
        length = geometry.get("minor_axis_length_px")
    else:
        vector = geometry.get("selected_axis_vector")
        length = geometry.get("selected_axis_length_px")
    point = _point_list(vector)
    if point is None or not _is_number(length):
        return None, None
    norm = math.hypot(point[0], point[1])
    if norm <= 0:
        return None, None
    return [point[0] / norm, point[1] / norm], float(length)


def _target_geometry_from_anchor_result(anchor_result: dict[str, Any] | None, axis_mode: str) -> dict[str, Any]:
    geometry = (anchor_result or {}).get("geometry") if isinstance(anchor_result, dict) else None
    if not isinstance(geometry, dict):
        return {
            "target_mask_available": False,
            "target_centroid_px": None,
            "major_axis_unit_px": None,
            "minor_axis_unit_px": None,
            "major_axis_length_px": None,
            "minor_axis_length_px": None,
            "selected_axis_unit_px": None,
            "selected_axis_name": axis_mode,
        }
    selected_axis, _selected_length = _axis_for_mode(geometry, axis_mode)
    return {
        "target_mask_available": True,
        "target_centroid_px": _point_list(geometry.get("center_px")),
        "major_axis_unit_px": _point_list(geometry.get("major_axis_vector")),
        "minor_axis_unit_px": _point_list(geometry.get("minor_axis_vector")),
        "major_axis_length_px": float(geometry["major_axis_length_px"]) if _is_number(geometry.get("major_axis_length_px")) else None,
        "minor_axis_length_px": float(geometry["minor_axis_length_px"]) if _is_number(geometry.get("minor_axis_length_px")) else None,
        "selected_axis_unit_px": selected_axis,
        "selected_axis_name": axis_mode,
    }


def _target_geometry_from_anchor_summary(anchor_summary: dict[str, Any] | None, axis_mode: str) -> dict[str, Any]:
    geometry = (anchor_summary or {}).get("geometry") if isinstance(anchor_summary, dict) else None
    if not isinstance(geometry, dict):
        return _target_geometry_from_anchor_result(None, axis_mode)
    selected_axis = _point_list(geometry.get("selected_axis_vector"))
    return {
        "target_mask_available": True,
        "target_centroid_px": _point_list(geometry.get("center_px")),
        "major_axis_unit_px": _point_list(geometry.get("major_axis_vector")),
        "minor_axis_unit_px": _point_list(geometry.get("minor_axis_vector")),
        "major_axis_length_px": float(geometry["major_axis_length_px"]) if _is_number(geometry.get("major_axis_length_px")) else None,
        "minor_axis_length_px": float(geometry["minor_axis_length_px"]) if _is_number(geometry.get("minor_axis_length_px")) else None,
        "selected_axis_unit_px": selected_axis,
        "selected_axis_name": axis_mode,
    }


def _axis_metrics(anchor_center: list[float] | None, target_geometry: dict[str, Any]) -> tuple[float | None, float | None]:
    if anchor_center is None:
        return None, None
    center = target_geometry.get("target_centroid_px")
    axis = target_geometry.get("selected_axis_unit_px")
    length = target_geometry.get(f"{target_geometry.get('selected_axis_name')}_axis_length_px")
    if _point_list(center) is None or _point_list(axis) is None or not _is_number(length) or float(length) <= 0:
        return None, None
    s_value = (anchor_center[0] - center[0]) * axis[0] + (anchor_center[1] - center[1]) * axis[1]
    normalized = (s_value + float(length) / 2.0) / float(length)
    return float(s_value), float(normalized)


def _side_quality_summary(feature_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(feature_record, dict):
        return None
    top = feature_record.get("top") if isinstance(feature_record.get("top"), dict) else {}
    bottom = feature_record.get("bottom") if isinstance(feature_record.get("bottom"), dict) else {}
    return {
        "top_side_quality_score": top.get("side_quality_score"),
        "bottom_side_quality_score": bottom.get("side_quality_score"),
        "top_edge_roughness_px": top.get("edge_roughness_px"),
        "bottom_edge_roughness_px": bottom.get("edge_roughness_px"),
        "top_mortar_like_anomaly_score": top.get("mortar_like_anomaly_score"),
        "bottom_mortar_like_anomaly_score": bottom.get("mortar_like_anomaly_score"),
    }


def _anchor_feature_maps(anchor_result: dict[str, Any] | None = None, anchor_summary: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    deterministic = None
    decision_trace = None
    tile_items = None
    if isinstance(anchor_result, dict):
        deterministic = anchor_result.get("deterministic_features")
        decision_trace = anchor_result.get("decision_trace")
        tile_items = (anchor_result.get("anchor_summary") or {}).get("wide_contact_guided_tile_results")
    if isinstance(anchor_summary, dict):
        tile_items = tile_items or anchor_summary.get("wide_contact_guided_tile_results")

    feature_by_id = {
        str(item.get("anchor_id")): item
        for item in (deterministic or {}).get("anchors", [])
        if isinstance(item, dict) and item.get("anchor_id")
    }
    trace_by_id = {
        str(item.get("anchor_id")): item
        for item in (decision_trace or {}).get("anchors", [])
        if isinstance(item, dict) and item.get("anchor_id")
    }
    tile_by_id = {
        str(item.get("source_anchor_id") or item.get("candidate_id")): item
        for item in (tile_items or [])
        if isinstance(item, dict) and (item.get("source_anchor_id") or item.get("candidate_id"))
    }
    return {"feature_by_id": feature_by_id, "trace_by_id": trace_by_id, "tile_by_id": tile_by_id}


def _build_anchor_item(
    anchor_id: str,
    anchor_index: int,
    anchor_center_px: list[float] | None,
    anchor_center_source: str,
    target_geometry: dict[str, Any],
    source_record: dict[str, Any] | None,
    feature_record: dict[str, Any] | None,
    trace_record: dict[str, Any] | None,
    tile_record: dict[str, Any] | None,
) -> dict[str, Any]:
    axis_t_value, normalized_axis_position = _axis_metrics(anchor_center_px, target_geometry)
    merged = trace_record or feature_record or {}
    item = {
        "anchor_id": anchor_id,
        "anchor_index": int(anchor_index),
        "anchor_center_px": anchor_center_px,
        "anchor_center_source": anchor_center_source,
        "axis_t_value": axis_t_value,
        "normalized_axis_position": normalized_axis_position,
        "selected_axis_unit_px": target_geometry.get("selected_axis_unit_px"),
        "opposite_contact_top_px": _point_list((source_record or {}).get("top_hit_xy")),
        "opposite_contact_bottom_px": _point_list((source_record or {}).get("bottom_hit_xy")),
        "opposite_contact_left_px": _point_list((source_record or {}).get("left_hit_xy")),
        "opposite_contact_right_px": _point_list((source_record or {}).get("right_hit_xy")),
        "top_contact_guide_y_px": (tile_record or feature_record or {}).get("top_contact_guide_y_px"),
        "bottom_contact_guide_y_px": (tile_record or feature_record or {}).get("bottom_contact_guide_y_px"),
        "deterministic_score": merged.get("deterministic_score"),
        "paired_consistency_score": merged.get("paired_consistency_score"),
        "valid": bool(merged.get("valid", (source_record or {}).get("valid", anchor_center_px is not None))),
        "veto": trace_record.get("veto") if isinstance(trace_record, dict) and "veto" in trace_record else None,
        "veto_reasons": trace_record.get("veto_reasons") if isinstance(trace_record, dict) else None,
        "side_quality_summary": _side_quality_summary(feature_record or trace_record),
    }
    return item


def _safe_final_anchor_id(final_decision: dict[str, Any]) -> str | None:
    selected = final_decision.get("selected_anchor")
    if isinstance(selected, str) and selected and not selected.startswith("NO_"):
        return selected
    final_selected = final_decision.get("final_selected_anchor")
    if isinstance(final_selected, str) and final_selected and not final_selected.startswith("NO_"):
        return final_selected
    return None


def _choose_robot_candidate(anchors: list[dict[str, Any]], final_decision: dict[str, Any]) -> dict[str, Any]:
    safe_anchor_id = _safe_final_anchor_id(final_decision)
    by_id = {anchor["anchor_id"]: anchor for anchor in anchors}
    if safe_anchor_id and safe_anchor_id in by_id:
        anchor = by_id[safe_anchor_id]
        return {
            "robot_candidate_anchor_id": safe_anchor_id,
            "robot_candidate_anchor_center_px": anchor.get("anchor_center_px"),
            "robot_candidate_selection_mode": "safe_final_anchor",
            "robot_candidate_is_safe": True,
            "robot_candidate_for_motion_allowed": False,
        }

    scored = [
        anchor
        for anchor in anchors
        if anchor.get("valid") and _is_number(anchor.get("deterministic_score")) and anchor.get("anchor_center_px") is not None
    ]
    if scored:
        scored.sort(key=lambda item: (-float(item["deterministic_score"]), _anchor_sort_key(item["anchor_id"])))
        anchor = scored[0]
        return {
            "robot_candidate_anchor_id": anchor["anchor_id"],
            "robot_candidate_anchor_center_px": anchor.get("anchor_center_px"),
            "robot_candidate_selection_mode": "debug_best_non_safe_anchor_by_deterministic_score",
            "robot_candidate_is_safe": False,
            "robot_candidate_for_motion_allowed": False,
        }

    return {
        "robot_candidate_anchor_id": None,
        "robot_candidate_anchor_center_px": None,
        "robot_candidate_selection_mode": None,
        "robot_candidate_is_safe": False,
        "robot_candidate_for_motion_allowed": False,
    }


def _base_payload(
    image_path: str | None,
    requested_class: str | None,
    axis_mode: str,
    target_selected: bool,
    selected_target_id: str | None,
    final_decision: dict[str, Any],
    target_geometry: dict[str, Any],
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    final_anchor_id = _safe_final_anchor_id(final_decision) or final_decision.get("final_selected_anchor") or final_decision.get("final_anchor_status")
    payload = {
        "timestamp": _timestamp_now(),
        "image_path": image_path,
        "requested_class": requested_class,
        "axis_mode": axis_mode,
        "target_selected": bool(target_selected),
        "selected_target_id": selected_target_id,
        "final_anchor_id": final_anchor_id,
        "final_anchor_is_safe": bool(_safe_final_anchor_id(final_decision)),
        "coordinate_frame": "color_image_pixels",
        "note": "Pixel coordinates are in the aligned RealSense color image frame when used with R1B/R2.",
        "target_geometry": target_geometry,
        "anchors": anchors,
        **_choose_robot_candidate(anchors, final_decision),
        "robot_motion_allowed": False,
        "reason_robot_motion_blocked": ROBOT_MOTION_BLOCKED_REASON,
    }
    return payload


def build_robot_anchor_geometry_from_v2a_state(
    image_path: str,
    requested_class: str,
    axis_mode: str,
    target_case: dict[str, Any],
    final_decision: dict[str, Any],
    anchor_result: dict[str, Any] | None,
) -> dict[str, Any]:
    target_selected = target_case.get("selected_status") == "SELECTED_BY_CROP_VERIFY"
    selected_target_id = target_case.get("selected_candidate_id")
    target_geometry = _target_geometry_from_anchor_result(anchor_result, axis_mode)
    maps = _anchor_feature_maps(anchor_result=anchor_result)

    anchors: list[dict[str, Any]] = []
    for idx, source in enumerate((anchor_result or {}).get("source_candidates", []) or [], start=1):
        anchor_id = str(source.get("source_anchor_id") or source.get("candidate_id") or f"A{idx}")
        anchors.append(
            _build_anchor_item(
                anchor_id=anchor_id,
                anchor_index=idx,
                anchor_center_px=_point_list(source.get("anchor_xy")),
                anchor_center_source="v2a_source_candidates.anchor_xy",
                target_geometry=target_geometry,
                source_record=source,
                feature_record=maps["feature_by_id"].get(anchor_id),
                trace_record=maps["trace_by_id"].get(anchor_id),
                tile_record=maps["tile_by_id"].get(anchor_id),
            )
        )

    return _base_payload(
        image_path=image_path,
        requested_class=requested_class,
        axis_mode=axis_mode,
        target_selected=target_selected,
        selected_target_id=selected_target_id,
        final_decision=final_decision,
        target_geometry=target_geometry,
        anchors=anchors,
    )


def _even_samples(s_min: float, s_max: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(s_min + s_max) / 2.0]
    step = (s_max - s_min) / float(count - 1)
    return [s_min + step * idx for idx in range(count)]


def _reconstruct_centers(anchor_summary: dict[str, Any] | None) -> dict[str, list[float]]:
    if not isinstance(anchor_summary, dict):
        return {}
    geometry = anchor_summary.get("geometry")
    if not isinstance(geometry, dict):
        return {}
    center = _point_list(geometry.get("center_px"))
    axis = _point_list(geometry.get("selected_axis_vector"))
    length = geometry.get("selected_axis_length_px")
    count = anchor_summary.get("num_anchors")
    margin_ratio = anchor_summary.get("axis_margin_ratio")
    order = anchor_summary.get("source_candidate_order")
    if center is None or axis is None or not _is_number(length) or not _is_number(count) or not _is_number(margin_ratio):
        return {}
    count = int(count)
    half_extent = float(length) / 2.0
    margin_px = float(length) * float(margin_ratio)
    s_values = _even_samples(-half_extent + margin_px, half_extent - margin_px, count)
    if not isinstance(order, list) or len(order) != count:
        order = [f"A{idx}" for idx in range(1, count + 1)]
    return {
        str(anchor_id): [center[0] + axis[0] * s_value, center[1] + axis[1] * s_value]
        for anchor_id, s_value in zip(order, s_values)
    }


def build_robot_anchor_geometry_from_v2a_output_dir(v2a_run_dir: Path) -> dict[str, Any]:
    final_decision = _load_json(v2a_run_dir / "final_decision.json") or {}
    final_summary = _load_json(v2a_run_dir / "final_summary.json") or {}
    anchor_summary = _load_json(v2a_run_dir / "anchor_selection" / "anchor_stage_summary.json")
    features = _load_json(v2a_run_dir / "anchor_selection" / "deterministic_anchor_features_wide_context.json")
    trace = _load_json(v2a_run_dir / "anchor_selection" / "hybrid_decision_trace_wide_context.json")

    requested_class = final_summary.get("requested_class") or final_decision.get("requested_class")
    axis_mode = final_summary.get("axis_mode") or final_decision.get("axis_mode")
    target = final_summary.get("target_selection", {}) if isinstance(final_summary.get("target_selection"), dict) else {}
    selected_target_id = final_decision.get("selected_target_id") or target.get("selected_candidate_id")
    target_selected = bool(selected_target_id and (final_decision.get("target_selection_status") == "SELECTED_BY_CROP_VERIFY" or target.get("selected_status") == "SELECTED_BY_CROP_VERIFY"))
    target_geometry = _target_geometry_from_anchor_summary(anchor_summary, axis_mode)
    reconstructed_centers = _reconstruct_centers(anchor_summary)
    maps = _anchor_feature_maps(
        anchor_result={"deterministic_features": features, "decision_trace": trace},
        anchor_summary=anchor_summary,
    )

    anchor_ids = set(reconstructed_centers)
    anchor_ids.update(maps["feature_by_id"])
    anchor_ids.update(maps["trace_by_id"])
    anchor_ids.update(maps["tile_by_id"])
    anchors: list[dict[str, Any]] = []
    for idx, anchor_id in enumerate(sorted(anchor_ids, key=_anchor_sort_key), start=1):
        anchors.append(
            _build_anchor_item(
                anchor_id=anchor_id,
                anchor_index=idx,
                anchor_center_px=reconstructed_centers.get(anchor_id),
                anchor_center_source="reconstructed_from_anchor_stage_summary_geometry",
                target_geometry=target_geometry,
                source_record=None,
                feature_record=maps["feature_by_id"].get(anchor_id),
                trace_record=maps["trace_by_id"].get(anchor_id),
                tile_record=maps["tile_by_id"].get(anchor_id),
            )
        )

    return _base_payload(
        image_path=final_summary.get("image_path") or (anchor_summary or {}).get("image_path"),
        requested_class=requested_class,
        axis_mode=axis_mode,
        target_selected=target_selected,
        selected_target_id=selected_target_id,
        final_decision=final_decision,
        target_geometry=target_geometry,
        anchors=anchors,
    )


def write_robot_anchor_geometry(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def build_and_write_robot_anchor_geometry_from_v2a_output_dir(v2a_run_dir: Path) -> Path:
    payload = build_robot_anchor_geometry_from_v2a_output_dir(v2a_run_dir)
    return write_robot_anchor_geometry(v2a_run_dir / "robot_anchor_geometry.json", payload)
