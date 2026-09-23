"""Fast screen capture and VLM image preparation."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image

from .models import ScreenFrame


class ScreenCaptureError(RuntimeError):
    """Raised when the desktop cannot be captured."""


class ScreenCapturer:
    def __init__(self, monitor: int = 1) -> None:
        if monitor < 0:
            raise ValueError("monitor must be zero or greater")
        self.monitor = monitor

    def capture(self, max_width: int = 1920, jpeg_quality: int = 82) -> ScreenFrame:
        if max_width < 1:
            raise ValueError("max_width must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        try:
            import mss

            with mss.mss() as session:
                monitors = session.monitors
                if self.monitor >= len(monitors):
                    raise ScreenCaptureError(
                        f"monitor {self.monitor} is unavailable; found {len(monitors) - 1} monitor(s)"
                    )
                raw = session.grab(monitors[self.monitor])
                image = Image.frombytes("RGB", raw.size, raw.rgb)
        except ScreenCaptureError:
            raise
        except Exception as exc:
            raise ScreenCaptureError(f"screen capture failed: {exc}") from exc

        source_width, source_height = image.size
        scale = min(1.0, max_width / source_width)
        if scale < 1.0:
            image = image.resize(
                (round(source_width * scale), round(source_height * scale)),
                Image.Resampling.LANCZOS,
            )

        output = BytesIO()
        image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        return ScreenFrame(
            image_bytes=output.getvalue(),
            width=image.width,
            height=image.height,
            scale_x=image.width / source_width,
            scale_y=image.height / source_height,
            monitor=self.monitor,
        )


def image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Decode an encoded frame and detach it from its input stream."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("invalid image bytes") from exc
