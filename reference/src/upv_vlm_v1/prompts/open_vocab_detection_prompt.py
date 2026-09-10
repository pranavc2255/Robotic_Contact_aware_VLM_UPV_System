from __future__ import annotations


def build_open_vocab_detection_prompt(material_query: str) -> str:
    """Return the simple text query sent to GroundingDINO/SAM2."""

    return str(material_query or "").strip().lower()

