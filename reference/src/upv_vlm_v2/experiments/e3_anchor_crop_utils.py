"""Compatibility wrapper for E3 clean anchor crop rendering.

The canonical implementation lives in the main anchor-selection package so
manual labels, Qwen scoring, and main pipeline artifacts share one crop geometry.
"""

from upv_vlm_v2.anchor_selection.partition_contact_crop_builder import *  # noqa: F401,F403

