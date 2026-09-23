from dataclasses import dataclass

from desktop_bot.loop import LoopStatus, TaskRunner
from desktop_bot.memory import WorkflowMatch, WorkflowTrace
from desktop_bot.models import ActionCommand, ScreenFrame


def frame(label: bytes) -> ScreenFrame:
    return ScreenFrame(image_bytes=label, width=100, height=100, scale_x=1, scale_y=1, monitor=1)


def action(name="click", **kwargs) -> ActionCommand:
    values = {"thought": "test", "action": name, "target": {"x": 50, "y": 50}, "expected_outcome": "done"}
    values.update(kwargs)
    return ActionCommand(**values)


@dataclass
class FakeCapture:
    frames: list[ScreenFrame]

    def capture(self):
        return self.frames.pop(0)


class FakeModel:
    def __init__(self, actions, verifications):
        self.actions = iter(actions)
        self.verifications = iter(verifications)
        self.decide_calls = 0

    def decide(self, image_bytes, instruction):
        self.decide_calls += 1
        return next(self.actions)

    def verify(self, previous_image, current_image, expected_outcome):
        return next(self.verifications)


class FakeController:
    def __init__(self):
        self.actions = []

    def execute(self, command):
        self.actions.append(command)


class FakeMemory:
    def __init__(self, match=None):
        self.match = match
        self.saved = []

    def find(self, prompt, threshold):
        return self.match

    def save(self, trace):
        self.saved.append(trace)


def test_runner_prompts_then_replays_matching_workflow_without_vlm():
    recorded = action("click")
    match = WorkflowMatch(
        trace=WorkflowTrace("Open settings", [{"command": recorded.model_dump(mode="json")}]),
        similarity=0.96,
    )
    memory = FakeMemory(match)
    controller = FakeController()
    model = FakeModel([], [])
    runner = TaskRunner(
        FakeCapture([]), model, controller, memory=memory,
        replay_confirmation=lambda found: found.similarity > 0.9,
    )

    state = runner.run("Open settings")

    assert state.status is LoopStatus.COMPLETED
    assert state.replayed is True
    assert model.decide_calls == 0
    assert controller.actions == [recorded]


def test_runner_requires_explicit_hitl_confirmation_for_destructive_action():
    model = FakeModel([action("type", text="submit payment")], [])
    controller = FakeController()
    runner = TaskRunner(FakeCapture([frame(b"before")]), model, controller, sleep=lambda _: None)

    state = runner.run("Submit the payment")

    assert state.status is LoopStatus.NEEDS_INTERVENTION
    assert "confirmation" in state.failure_context
    assert controller.actions == []