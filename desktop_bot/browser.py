"""Optional Playwright hook for browser-only actions."""

from __future__ import annotations

from typing import Any

from .models import ActionCommand, ActionType


class BrowserExecutionError(RuntimeError):
    """Raised when a DOM action cannot be completed."""


def execute_dom_action(page: Any, command: ActionCommand, selector: str) -> None:
    """Execute a browser action by selector instead of screen coordinates.

    Playwright is intentionally optional. Pass a Playwright Page object from the
    caller; this module does not start a browser or import Playwright itself.
    """
    if not selector.strip():
        raise BrowserExecutionError("selector cannot be empty")
    locator = page.locator(selector)
    try:
        if command.action is ActionType.CLICK:
            locator.click()
        elif command.action is ActionType.DOUBLE_CLICK:
            locator.dblclick()
        elif command.action is ActionType.TYPE:
            if not command.text:
                raise BrowserExecutionError("type action requires non-empty text")
            locator.fill(command.text)
        elif command.action is ActionType.WAIT:
            locator.wait_for()
        else:
            raise BrowserExecutionError(f"DOM hook does not support {command.action.value}")
    except BrowserExecutionError:
        raise
    except Exception as exc:
        raise BrowserExecutionError(f"browser action failed for {selector}: {exc}") from exc
