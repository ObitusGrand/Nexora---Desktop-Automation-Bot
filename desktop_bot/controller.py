"""Safe OS-level action execution for validated VLM commands."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .models import ActionCommand, ActionType, Target


class ActionExecutionError(RuntimeError):
    """Raised when an action cannot be safely executed."""


class InputBackend(Protocol):
    FAILSAFE: bool
    PAUSE: float

    def size(self) -> tuple[int, int]: ...
    def moveTo(self, x: int, y: int, duration: float = 0) -> None: ...
    def click(self, x: int, y: int, clicks: int = 1, interval: float = 0) -> None: ...
    def write(self, text: str, interval: float = 0) -> None: ...
    def hotkey(self, *keys: str, interval: float = 0) -> None: ...
    def scroll(self, clicks: int) -> None: ...


@dataclass(frozen=True)
class ControllerConfig:
    """Timing and safety limits for physical input."""

    movement_duration: float = 0.08
    key_interval: float = 0.025
    action_delay: float = 0.15
    max_text_length: int = 10_000
    max_scroll_clicks: int = 100
    failsafe_margin: int = 8

    def __post_init__(self) -> None:
        if self.movement_duration < 0 or self.key_interval < 0 or self.action_delay < 0:
            raise ValueError("action timing values cannot be negative")
        if self.max_text_length < 1 or self.max_scroll_clicks < 1 or self.failsafe_margin < 1:
            raise ValueError("action limits must be positive")


class DesktopController:
    """Translate one validated action into safe pyautogui input."""

    def __init__(
        self,
        backend: InputBackend | None = None,
        config: ControllerConfig | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if backend is None:
            try:
                import pyautogui
            except ImportError as exc:
                raise ActionExecutionError("pyautogui is required for desktop control") from exc
            backend = pyautogui
        self.backend = backend
        self.config = config or ControllerConfig()
        self._sleep = sleep
        self.backend.FAILSAFE = True
        self.backend.PAUSE = self.config.action_delay

    def execute(self, command: ActionCommand) -> None:
        """Execute exactly one command; reject unsafe or incomplete payloads."""
        try:
            if command.action in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
                self._click(command.target, double=command.action is ActionType.DOUBLE_CLICK)
            elif command.action is ActionType.TYPE:
                self._type(command.text)
            elif command.action is ActionType.HOTKEY:
                self._hotkey(command.keys)
            elif command.action is ActionType.SCROLL:
                self._scroll(command.target.y)
            elif command.action in (ActionType.WAIT, ActionType.DONE):
                self._sleep(self.config.action_delay)
            else:
                raise ActionExecutionError(f"unsupported action: {command.action}")
        except ActionExecutionError:
            raise
        except Exception as exc:
            if type(exc).__name__ == "FailSafeException":
                raise ActionExecutionError(
                    "PyAutoGUI safety stop: move the mouse away from a screen corner, "
                    "then retry. The bot keeps FAILSAFE enabled."
                ) from exc
            raise ActionExecutionError(f"failed to execute {command.action.value}: {exc}") from exc

    def _validate_target(self, target: Target) -> None:
        width, height = self.backend.size()
        if not 0 <= target.x < width or not 0 <= target.y < height:
            raise ActionExecutionError(
                f"target ({target.x}, {target.y}) is outside screen bounds {width}x{height}"
            )
        margin = self.config.failsafe_margin
        near_corner = (
            target.x < margin and target.y < margin
            or target.x < margin and target.y >= height - margin
            or target.x >= width - margin and target.y < margin
            or target.x >= width - margin and target.y >= height - margin
        )
        if near_corner:
            raise ActionExecutionError(
                f"target ({target.x}, {target.y}) is too close to a screen corner; "
                f"click targets must be at least {margin} pixels from corners"
            )

    def _click(self, target: Target, double: bool) -> None:
        self._validate_target(target)
        self.backend.moveTo(target.x, target.y, duration=self.config.movement_duration)
        self.backend.click(target.x, target.y, clicks=2 if double else 1, interval=self.config.key_interval)

    def _type(self, text: str) -> None:
        if not text:
            raise ActionExecutionError("type action requires non-empty text")
        if len(text) > self.config.max_text_length:
            raise ActionExecutionError("text payload exceeds configured limit")
        if text.isascii():
            self.backend.write(text, interval=self.config.key_interval)
        else:
            self._paste_text(text)

    def _paste_text(self, text: str) -> None:
        """Type arbitrary text via clipboard paste (handles Unicode, emoji, etc.)."""
        import subprocess
        # Save current clipboard, set new content, paste, then restore
        try:
            old_clip = subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=3,
            ).stdout.rstrip("\r\n")
        except Exception:
            old_clip = None
        subprocess.run(
            ["powershell", "-Command", f"Set-Clipboard -Value '{text.replace(chr(39), chr(39)+chr(39))}'"],
            capture_output=True, timeout=3,
        )
        self.backend.hotkey("ctrl", "v", interval=self.config.key_interval)
        self._sleep(0.1)
        # Restore previous clipboard content
        if old_clip is not None:
            try:
                subprocess.run(
                    ["powershell", "-Command", f"Set-Clipboard -Value '{old_clip.replace(chr(39), chr(39)+chr(39))}'"],
                    capture_output=True, timeout=3,
                )
            except Exception:
                pass

    def _hotkey(self, keys: list[str]) -> None:
        if not keys or any(not key.strip() for key in keys):
            raise ActionExecutionError("hotkey action requires non-empty keys")
        if len(keys) > 6:
            raise ActionExecutionError("hotkey action accepts at most six keys")
        self.backend.hotkey(*keys, interval=self.config.key_interval)

    def _scroll(self, clicks: int) -> None:
        if clicks == 0 or abs(clicks) > self.config.max_scroll_clicks:
            raise ActionExecutionError("scroll amount is outside the configured safety limit")
        self.backend.scroll(clicks)


class WindowManager:
    """Small pygetwindow wrapper for optional window activation."""

    def __init__(self, window_api: Any | None = None) -> None:
        if window_api is None:
            try:
                import pygetwindow
            except ImportError as exc:
                raise ActionExecutionError("pygetwindow is required for window management") from exc
            window_api = pygetwindow
        self.window_api = window_api

    def activate(self, title: str) -> None:
        if not title.strip():
            raise ValueError("window title cannot be empty")
        windows = self.window_api.getWindowsWithTitle(title)
        if not windows:
            raise ActionExecutionError(f"no window found with title: {title}")
        try:
            windows[0].activate()
        except Exception as exc:
            raise ActionExecutionError(f"could not activate window: {title}") from exc
