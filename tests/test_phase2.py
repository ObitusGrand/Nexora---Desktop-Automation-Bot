from dataclasses import dataclass

import pytest

from desktop_bot.controller import ActionExecutionError, ControllerConfig, DesktopController
from desktop_bot.models import ActionCommand


@dataclass
class FakePyAutoGUI:
    FAILSAFE: bool = False
    PAUSE: float = 0

    def __post_init__(self):
        self.calls = []

    def size(self):
        return (1920, 1080)

    def moveTo(self, x, y, duration=0):
        self.calls.append(("moveTo", x, y, duration))

    def click(self, x, y, clicks=1, interval=0):
        self.calls.append(("click", x, y, clicks, interval))

    def write(self, text, interval=0):
        self.calls.append(("write", text, interval))

    def hotkey(self, *keys, interval=0):
        self.calls.append(("hotkey", keys, interval))

    def scroll(self, clicks):
        self.calls.append(("scroll", clicks))


def command(action, **kwargs):
    values = {"thought": "test", "action": action, "target": {"x": 100, "y": 200}, "expected_outcome": "done"}
    values.update(kwargs)
    return ActionCommand(**values)


def test_controller_enables_pyautogui_safety_and_executes_actions():
    backend = FakePyAutoGUI()
    controller = DesktopController(backend, sleep=lambda _: None)

    controller.execute(command("click"))
    controller.execute(command("double_click"))
    controller.execute(command("type", text="hello"))
    controller.execute(command("hotkey", keys=["ctrl", "l"]))
    controller.execute(command("scroll", target={"x": 0, "y": -3}))

    assert backend.FAILSAFE is True
    assert backend.PAUSE == 0.15
    assert [call[0] for call in backend.calls] == ["moveTo", "click", "moveTo", "click", "write", "hotkey", "scroll"]
    assert backend.calls[3][3] == 2


def test_controller_rejects_out_of_bounds_and_missing_payloads():
    controller = DesktopController(FakePyAutoGUI(), sleep=lambda _: None)

    with pytest.raises(ActionExecutionError, match="outside screen bounds"):
        controller.execute(command("click", target={"x": 1920, "y": 200}))
    with pytest.raises(ActionExecutionError, match="non-empty text"):
        controller.execute(command("type"))
    with pytest.raises(ActionExecutionError, match="non-empty keys"):
        controller.execute(command("hotkey"))
    with pytest.raises(ActionExecutionError, match="safety limit"):
        controller.execute(command("scroll", target={"x": 0, "y": 101}))


def test_controller_rejects_invalid_timing_configuration():
    with pytest.raises(ValueError):
        ControllerConfig(action_delay=-1)
