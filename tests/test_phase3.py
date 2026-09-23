from dataclasses import dataclass

from PIL import Image
from io import BytesIO

from desktop_bot.loop import LoopStatus, TaskRunner
from desktop_bot.models import ActionCommand, ScreenFrame, VerificationResult


def frame(label: bytes) -> ScreenFrame:
    return ScreenFrame(image_bytes=label, width=100, height=100, scale_x=1, scale_y=1, monitor=1)


def action(name="click") -> ActionCommand:
    return ActionCommand(thought="test", action=name, target={"x": 5, "y": 5}, expected_outcome="dialog opens")


@dataclass
class FakeCapture:
    frames: list[ScreenFrame]

    def capture(self):
        return self.frames.pop(0)


class FakeModel:
    def __init__(self, actions, verifications):
        self.actions = iter(actions)
        self.verifications = iter(verifications)
        self.instructions = []
        self.comparisons = []

    def decide(self, image_bytes, instruction):
        self.instructions.append(instruction)
        return next(self.actions)

    def verify(self, previous_image, current_image, expected_outcome):
        self.comparisons.append((previous_image, current_image, expected_outcome))
        return next(self.verifications)


class FakeController:
    def __init__(self):
        self.actions = []

    def execute(self, command):
        self.actions.append(command)


def test_runner_retries_with_failure_context_then_completes():
    model = FakeModel(
        [action(), action(), action("done")],
        [VerificationResult(verified=False, reason="button did not open"), VerificationResult(verified=True, reason="dialog visible")],
    )
    controller = FakeController()
    runner = TaskRunner(FakeCapture([frame(b"before"), frame(b"failed"), frame(b"success")]), model, controller, sleep=lambda _: None)

    state = runner.run("Open the dialog")

    assert state.status is LoopStatus.COMPLETED
    assert len(state.step_history) == 2
    assert state.retry_count == 0
    assert "button did not open" in model.instructions[1]
    assert model.comparisons[0][0] == b"before"
    assert model.comparisons[0][1] == b"failed"


def test_runner_stops_for_user_intervention_after_retry_limit():
    model = FakeModel([action(), action()], [VerificationResult(verified=False, reason="no change"), VerificationResult(verified=False, reason="still no change")])
    controller = FakeController()
    runner = TaskRunner(FakeCapture([frame(b"one"), frame(b"two"), frame(b"three")]), model, controller, max_retries=1, sleep=lambda _: None)

    state = runner.run("Open the dialog")

    assert state.status is LoopStatus.NEEDS_INTERVENTION
    assert state.retry_count == 2
    assert state.failure_context == "still no change"


def test_runner_returns_failed_state_when_capture_or_action_fails():
    class BrokenCapture:
        def capture(self):
            raise RuntimeError("display unavailable")

    runner = TaskRunner(BrokenCapture(), FakeModel([], []), FakeController(), sleep=lambda _: None)
    state = runner.run("Do something")

    assert state.status is LoopStatus.FAILED
    assert "display unavailable" in state.failure_context
