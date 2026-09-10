from __future__ import annotations

from typing import Any

from upv_vlm_orient.experiment_logging.experiment_schema import default_manual_labels


def merge_manual_labels(values: dict[str, Any] | None) -> dict[str, Any]:
    labels = default_manual_labels()
    if values:
        labels.update({key: value for key, value in values.items() if value not in (None, "")})
    return labels
