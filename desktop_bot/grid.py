"""Coordinate grid annotation for visual localization."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from .capture import image_from_bytes


def add_coordinate_grid(
    image_bytes: bytes,
    cell_size: int = 100,
    line_color: tuple[int, int, int] = (255, 80, 80),
    label_color: tuple[int, int, int] = (255, 255, 0),
) -> bytes:
    """Overlay pixel coordinates and return a JPEG suitable for a VLM.

    Returns the original bytes unchanged if the image cannot be decoded.
    """
    if cell_size < 20:
        raise ValueError("cell_size must be at least 20 pixels")

    try:
        image = image_from_bytes(image_bytes)
    except (ValueError, Exception):
        return image_bytes

    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    font = ImageFont.load_default()
    for x in range(0, width, cell_size):
        draw.line((x, 0, x, height), fill=(*line_color, 150), width=1)
        draw.text((x + 3, 3), str(x), fill=(*label_color, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0, 255))
    for y in range(0, height, cell_size):
        draw.line((0, y, width, y), fill=(*line_color, 150), width=1)
        draw.text((3, y + 3), str(y), fill=(*label_color, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0, 255))

    output = BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue()
