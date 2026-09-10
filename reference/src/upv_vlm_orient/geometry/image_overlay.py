from pathlib import Path

from PIL import ImageDraw


def save_image_geometry_overlay(image, geometry: dict, output_path: str) -> str:
    output = image.copy()
    draw = ImageDraw.Draw(output)

    center_x, center_y = geometry["center_px"]
    major_dx, major_dy = geometry["major_axis_vector"]
    minor_dx, minor_dy = geometry["minor_axis_vector"]
    major_half = geometry["major_axis_length_px"] / 2.0
    minor_half = geometry["minor_axis_length_px"] / 2.0

    marker_size = max(10, min(output.width, output.height) // 25)
    line_width = max(2, min(output.width, output.height) // 150)

    draw.line(
        [(center_x - marker_size, center_y), (center_x + marker_size, center_y)],
        fill="red",
        width=line_width,
    )
    draw.line(
        [(center_x, center_y - marker_size), (center_x, center_y + marker_size)],
        fill="red",
        width=line_width,
    )

    draw.line(
        [
            (center_x - major_dx * major_half, center_y - major_dy * major_half),
            (center_x + major_dx * major_half, center_y + major_dy * major_half),
        ],
        fill="lime",
        width=line_width,
    )
    draw.line(
        [
            (center_x - minor_dx * minor_half, center_y - minor_dy * minor_half),
            (center_x + minor_dx * minor_half, center_y + minor_dy * minor_half),
        ],
        fill="cyan",
        width=line_width,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_file)
    return str(output_file)
