"""Perceive-plan-act-verify orchestration with bounded self-correction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from .memory import WorkflowMatch, WorkflowMemory, WorkflowTrace
from .models import ActionCommand, ScreenFrame, VerificationResult


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
            for _ in range(max_steps):
                instruction = self._planning_instruction(state)
                action = self.model.decide(state.current_screenshot.image_bytes, instruction)
                state.last_action = action
                if action.action.value == "done":
                    state.status = LoopStatus.COMPLETED
                    self._save_trace(state)
                    return state

                record = StepRecord(action=action)
                state.step_history.append(record)
                if requires_human_confirmation(action) and (
                    self.human_confirmation is None or not self.human_confirmation(action)
                ):
                    state.status = LoopStatus.NEEDS_INTERVENTION
                    state.failure_context = "human confirmation required before destructive action"
                    return state
                self.controller.execute(action)
                self._sleep(self.verification_delay)
                previous = state.current_screenshot
                current = self.capture.capture()
                verification = self.model.verify(previous.image_bytes, current.image_bytes, action.expected_outcome)
                record.verification = verification
                state.current_screenshot = current
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

    def _planning_instruction(self, state: TaskState) -> str:
        if not state.failure_context:
            return state.task_prompt
        return (
            f"{state.task_prompt}\n\nThe previous attempt was not verified. "
            f"Failure evidence: {state.failure_context}\n"
            "Choose a safer alternative, adjusted coordinate, or recovery action."
        )
