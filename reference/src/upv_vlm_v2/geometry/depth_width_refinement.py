"""Depth-refined path-length wrapper for v2.

Phase 2 exposes local selected-anchor path-length measurement through
`geometry.local_chord.compute_local_chord`. This module keeps a narrow wrapper
name for later migration of the more elaborate edge-bin robust projection code.
"""

from __future__ import annotations

from typing import Any

from upv_vlm_v2.geometry.local_chord import compute_local_chord


def estimate_depth_refined_path_length(**kwargs: Any) -> dict[str, Any]:
    return compute_local_chord(**kwargs)
