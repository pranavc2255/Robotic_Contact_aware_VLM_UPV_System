import numpy as np


def foreground_pixel_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    foreground = mask > 0

    if not np.any(foreground):
        raise ValueError("Mask has no foreground pixels.")

    height, width = foreground.shape
    visited = np.zeros((height, width), dtype=bool)
    largest_component = []

    for start_y, start_x in np.argwhere(foreground):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        component = []
        visited[start_y, start_x] = True

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            y_min = max(0, y - 1)
            y_max = min(height - 1, y + 1)
            x_min = max(0, x - 1)
            x_max = min(width - 1, x + 1)

            for next_y in range(y_min, y_max + 1):
                for next_x in range(x_min, x_max + 1):
                    if visited[next_y, next_x]:
                        continue

                    if not foreground[next_y, next_x]:
                        continue

                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))

        if len(component) > len(largest_component):
            largest_component = component

    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

    for y, x in largest_component:
        cleaned_mask[y, x] = 255

    return cleaned_mask
