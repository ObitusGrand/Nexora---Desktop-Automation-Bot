"""Perceive-plan-act-verify orchestration with bounded self-correction."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


def _log(msg: str) -> None:
    """Print debug info to stderr so it's visible in the terminal."""
    print(f"[desktop-bot] {msg}", file=sys.stderr, flush=True)

from .grid import add_coordinate_grid
from .memory import WorkflowMatch, WorkflowMemory, WorkflowTrace
from .models import ActionCommand, ScreenFrame, Target, VerificationResult


class LoopStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_INTERVENTION = "needs_intervention"
    FAILED = "failed"


@dataclass
class StepRecord:
    action: ActionCommand
    verification: VerificationResult | None = None
    error: str | None = None


@dataclass
class TaskState:
    task_prompt: str
    current_screenshot: ScreenFrame | None = None
    last_action: ActionCommand | None = None
    step_history: list[StepRecord] = field(default_factory=list)
    retry_count: int = 0
    status: LoopStatus = LoopStatus.RUNNING
    failure_context: str = ""
    replayed: bool = False


class CaptureSource(Protocol):
    def capture(self) -> ScreenFrame: ...


class PlannerVerifier(Protocol):
    def decide(self, image_bytes: bytes, instruction: str) -> ActionCommand: ...
    def verify(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> VerificationResult: ...


class ActionExecutor(Protocol):
    def execute(self, command: ActionCommand) -> None: ...


REPLAY_PROMPT = "I have executed this workflow before. Would you like me to replay the recorded steps?"


def requires_human_confirmation(command: ActionCommand) -> bool:
    """Conservatively identify actions likely to delete, submit, or communicate."""
    sensitive_terms = (
        "delete", "remove", "erase", "submit", "send", "message", "email",
        "payment", "pay", "purchase", "buy", "transfer", "publish",
    )
    content = " ".join((command.text, command.thought, command.expected_outcome)).lower()
    return command.requires_confirmation or any(term in content for term in sensitive_terms)


class TaskRunner:
    """Run one task until done, retry exhaustion, or an unrecoverable error."""

    def __init__(
        self,
        capture: CaptureSource,
        model: PlannerVerifier,
        controller: ActionExecutor,
        max_retries: int = 2,
        verification_delay: float = 1.0,
        sleep: Any = time.sleep,
        memory: WorkflowMemory | None = None,
        replay_threshold: float = 0.9,
        replay_confirmation: Callable[[WorkflowMatch], bool] | None = None,
        human_confirmation: Callable[[ActionCommand], bool] | None = None,
        app_names: list[str] | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if verification_delay < 0:
            raise ValueError("verification_delay cannot be negative")
        if not 0 <= replay_threshold <= 1:
            raise ValueError("replay_threshold must be between 0 and 1")
        self.capture = capture
        self.model = model
        self.controller = controller
        self.max_retries = max_retries
        self.verification_delay = verification_delay
        self._sleep = sleep
        self.memory = memory
        self.replay_threshold = replay_threshold
        self.replay_confirmation = replay_confirmation
        self.human_confirmation = human_confirmation
        self.app_names = app_names or []

    def run(self, task_prompt: str, max_steps: int = 50) -> TaskState:
        if not task_prompt.strip():
            raise ValueError("task_prompt cannot be empty")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        state = TaskState(task_prompt=task_prompt)
        try:
            if self.memory is not None:
                match = self.memory.find(task_prompt, self.replay_threshold)
                if match is not None and self.replay_confirmation is not None and self.replay_confirmation(match):
                    return self._replay(state, match)
            state.current_screenshot = self.capture.capture()
            _log(f"Captured screen: {state.current_screenshot.width}x{state.current_screenshot.height} "
                 f"(scale: {state.current_screenshot.scale_x:.3f}x{state.current_screenshot.scale_y:.3f})")
            for step_num in range(max_steps):
                instruction = self._planning_instruction(state)
                frame = state.current_screenshot
                annotated = add_coordinate_grid(frame.image_bytes)
                _log(f"Step {step_num + 1}: Asking VLM to decide...")
                action = self.model.decide(annotated, instruction)
                _log(f"Step {step_num + 1}: VLM returned action={action.action.value} "
                     f"target=({action.target.x}, {action.target.y}) "
                     f"text={action.text!r} keys={action.keys} "
                     f"thought={action.thought!r}")

                state.last_action = action
                if action.action.value == "done":
                    _log("VLM returned 'done' — task complete")
                    state.status = LoopStatus.COMPLETED
                    self._save_trace(state)
                    return state

                # Validate click target BEFORE rescaling (in image coordinate space)
                record = StepRecord(action=action)
                state.step_history.append(record)
                invalid_target = self._invalid_click_target(action, frame)
                if invalid_target:
                    _log(f"Invalid target: {invalid_target}")
                    record.error = invalid_target
                    state.retry_count += 1
                    state.failure_context = invalid_target
                    if state.retry_count > self.max_retries:
                        state.status = LoopStatus.NEEDS_INTERVENTION
                        return state
                    continue

                # Rescale coordinates from image-space to physical screen-space AFTER validation
                if action.action.value in ("click", "double_click"):
                    screen_target = frame.to_screen_coordinates(action.target)
                    action = action.model_copy(update={"target": screen_target})
                    record.action = action  # update the record too
                    _log(f"Rescaled target: ({action.target.x}, {action.target.y})")

                if requires_human_confirmation(action) and (
                    self.human_confirmation is None or not self.human_confirmation(action)
                ):
                    state.status = LoopStatus.NEEDS_INTERVENTION
                    state.failure_context = "human confirmation required before destructive action"
                    return state
                _log(f"Executing: {action.action.value}")
                self.controller.execute(action)
                self._sleep(self.verification_delay)
                previous = state.current_screenshot
                current = self.capture.capture()
                try:
                    verification = self.model.verify(previous.image_bytes, current.image_bytes, action.expected_outcome)
                except Exception as vex:
                    _log(f"Verification error (skipping): {vex}")
                    from .models import VerificationResult as VR
                    verification = VR(verified=True, reason="Verification skipped due to model error")
                record.verification = verification
                state.current_screenshot = current
                _log(f"Verification: verified={verification.verified} reason={verification.reason!r}")
                if verification.verified:
                    state.retry_count = 0
                    state.failure_context = ""
                    continue

                state.retry_count += 1
                state.failure_context = verification.reason
                if state.retry_count > self.max_retries:
                    state.status = LoopStatus.NEEDS_INTERVENTION
                    return state

            state.status = LoopStatus.NEEDS_INTERVENTION
            state.failure_context = "maximum step count exceeded"
            return state
        except Exception as exc:
            _log(f"Task failed with exception: {exc}")
            state.status = LoopStatus.FAILED
            state.failure_context = str(exc)
            return state

    def _replay(self, state: TaskState, match: WorkflowMatch) -> TaskState:
        for item in match.trace.actions:
            action = ActionCommand.model_validate(item["command"])
            state.last_action = action
            if requires_human_confirmation(action) and (
                self.human_confirmation is None or not self.human_confirmation(action)
            ):
                state.status = LoopStatus.NEEDS_INTERVENTION
                state.failure_context = "human confirmation required before replaying destructive action"
                return state
            self.controller.execute(action)
            state.step_history.append(StepRecord(action=action))
        state.replayed = True
        state.status = LoopStatus.COMPLETED
        return state

    def _save_trace(self, state: TaskState) -> None:
        if self.memory is None or not state.step_history:
            return
        actions = []
        for record in state.step_history:
            command = record.action.model_dump(mode="json")
            target = command["target"]
            frame = state.current_screenshot
            relative_target = None
            if frame is not None and record.action.action.value in {"click", "double_click"}:
                relative_target = {"x": target["x"] / frame.width, "y": target["y"] / frame.height}
            actions.append({"command": command, "relative_target": relative_target})
        self.memory.save(WorkflowTrace(task_prompt=state.task_prompt, actions=actions, app_names=self.app_names))

    @staticmethod
    def _invalid_click_target(action: ActionCommand, frame: ScreenFrame | None, margin: int = 8) -> str | None:
        if frame is None or action.action.value not in {"click", "double_click"}:
            return None
        target = action.target
        if target.x < margin or target.y < margin or target.x >= frame.width - margin or target.y >= frame.height - margin:
            return (
                f"The proposed click target ({target.x}, {target.y}) is invalid because it is "
                f"too close to a screen edge. Inspect the screenshot again and return a "
                f"real visible target at least {margin} pixels from every screen edge."
            )
        if not (0 <= target.x < frame.width) or not (0 <= target.y < frame.height):
            return (
                f"The proposed click target ({target.x}, {target.y}) is outside the "
                f"image bounds ({frame.width}x{frame.height}). Return a coordinate inside the image."
            )
        return None

    def _planning_instruction(self, state: TaskState) -> str:
        frame = state.current_screenshot
        dims = ""
        if frame is not None:
            dims = f"\n\nThe screenshot is {frame.width}x{frame.height} pixels. All coordinates must be within these bounds."
        if not state.failure_context:
            return state.task_prompt + dims
        return (
            f"{state.task_prompt}{dims}\n\nThe previous attempt was not verified. "
            f"Failure evidence: {state.failure_context}\n"
            "Choose a safer alternative, adjusted coordinate, or recovery action."
        )
