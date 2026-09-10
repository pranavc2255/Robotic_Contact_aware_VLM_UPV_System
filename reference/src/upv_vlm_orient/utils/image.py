from pathlib import Path

from PIL import Image, ImageDraw


def load_rgb_image(path: str) -> Image.Image:
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"RGB image file not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            return image.copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read RGB image file: {image_path}") from exc


def get_image_info(image: Image.Image) -> dict:
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }


def save_center_overlay(image: Image.Image, output_path: str) -> str:
    output = image.copy()
    draw = ImageDraw.Draw(output)

    center_x = output.width // 2
    center_y = output.height // 2
    half_size = max(10, min(output.width, output.height) // 20)
    color = "red"
    line_width = max(2, min(output.width, output.height) // 150)

    draw.line(
        [(center_x - half_size, center_y), (center_x + half_size, center_y)],
        fill=color,
        width=line_width,
    )
    draw.line(
        [(center_x, center_y - half_size), (center_x, center_y + half_size)],
        fill=color,
        width=line_width,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_file)
    return str(output_file)
