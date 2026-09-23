"""Typed contracts shared by perception and planning components."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    WAIT = "wait"
    DONE = "done"


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int


class ActionCommand(BaseModel):
    """The only command shape accepted from a VLM."""

    model_config = ConfigDict(extra="forbid")

    thought: str
    action: ActionType
    target: Target
    text: str = ""
    keys: list[str] = Field(default_factory=list)
    expected_outcome: str
    requires_confirmation: bool = False

    @field_validator("text")
    @classmethod
    def text_is_only_for_typing(cls, value: str) -> str:
        return value


class VerificationResult(BaseModel):
    """Strict result returned after comparing the before and after frames."""

    model_config = ConfigDict(extra="forbid")

    verified: bool
    reason: str


class ScreenFrame(BaseModel):
    """A captured screen and its relationship to the physical display."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_bytes: bytes
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    scale_x: float = Field(gt=0)
    scale_y: float = Field(gt=0)
    monitor: int = Field(ge=0)

    def to_screen_coordinates(self, target: Target) -> Target:
        return Target(x=round(target.x / self.scale_x), y=round(target.y / self.scale_y))
