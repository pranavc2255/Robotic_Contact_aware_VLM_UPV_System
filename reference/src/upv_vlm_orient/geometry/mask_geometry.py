from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_foreground_points(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    rows, cols = np.nonzero(mask > 0)

    if len(rows) == 0:
        raise ValueError("Mask has no foreground pixels.")

    return np.column_stack((cols.astype(float), rows.astype(float)))


def compute_mask_center(mask: np.ndarray) -> tuple[float, float]:
    points = _get_foreground_points(mask)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def canonicalize_angle_deg(angle_deg: float) -> float:
    return float(angle_deg % 180.0)


def compute_mask_axes(mask: np.ndarray) -> dict:
    points = _get_foreground_points(mask)
    center = points.mean(axis=0)
    centered_points = points - center

    covariance = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    major_axis_vector = eigenvectors[:, 0]
    minor_axis_vector = eigenvectors[:, 1]

    major_projections = centered_points @ major_axis_vector
    minor_projections = centered_points @ minor_axis_vector

    major_axis_length = float(major_projections.max() - major_projections.min())
    minor_axis_length = float(minor_projections.max() - minor_projections.min())
    major_axis_angle = canonicalize_angle_deg(
        float(np.degrees(np.arctan2(major_axis_vector[1], major_axis_vector[0])))
    )

    return {
        "center_px": [float(center[0]), float(center[1])],
        "major_axis_vector": [float(major_axis_vector[0]), float(major_axis_vector[1])],
        "minor_axis_vector": [float(minor_axis_vector[0]), float(minor_axis_vector[1])],
        "major_axis_angle_deg": major_axis_angle,
        "major_axis_length_px": major_axis_length,
        "minor_axis_length_px": minor_axis_length,
    }


def create_synthetic_rotated_rectangle_mask(
    width: int = 400,
    height: int = 300,
    rect_cx: float = 200,
    rect_cy: float = 150,
    rect_w: float = 140,
    rect_h: float = 60,
    angle_deg: float = 25.0,
) -> np.ndarray:
    yy, xx = np.indices((height, width))

    x_shifted = xx - rect_cx
    y_shifted = yy - rect_cy

    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    local_x = x_shifted * cos_a + y_shifted * sin_a
    local_y = -x_shifted * sin_a + y_shifted * cos_a

    inside = (np.abs(local_x) <= rect_w / 2.0) & (np.abs(local_y) <= rect_h / 2.0)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[inside] = 255
    return mask


def save_mask_axes_visualization(mask: np.ndarray, geometry: dict, output_path: str) -> str:
    center_x, center_y = geometry["center_px"]
    major_vector = np.array(geometry["major_axis_vector"], dtype=float)
    minor_vector = np.array(geometry["minor_axis_vector"], dtype=float)
    major_half = geometry["major_axis_length_px"] / 2.0
    minor_half = geometry["minor_axis_length_px"] / 2.0

    major_start = np.array([center_x, center_y]) - major_vector * major_half
    major_end = np.array([center_x, center_y]) + major_vector * major_half
    minor_start = np.array([center_x, center_y]) - minor_vector * minor_half
    minor_end = np.array([center_x, center_y]) + minor_vector * minor_half

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(mask, cmap="gray")
    ax.scatter([center_x], [center_y], color="red", s=40)
    ax.plot(
        [major_start[0], major_end[0]],
        [major_start[1], major_end[1]],
        color="lime",
        linewidth=2,
    )
    ax.plot(
        [minor_start[0], minor_end[0]],
        [minor_start[1], minor_end[1]],
        color="cyan",
        linewidth=2,
    )
    ax.set_title("Synthetic Mask Geometry")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return str(output_file)
