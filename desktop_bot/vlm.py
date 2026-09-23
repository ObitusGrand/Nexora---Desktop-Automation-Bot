"""VLM clients with strict JSON/action validation."""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .models import ActionCommand, VerificationResult

SYSTEM_PROMPT = """You control a desktop through one action at a time. Inspect the screenshot and return ONLY one valid JSON object with exactly these fields:
{"thought":"brief state reasoning","action":"click|double_click|type|hotkey|scroll|wait|done","target":{"x":0,"y":0},"text":"","keys":[],"expected_outcome":"what the next screenshot should show","requires_confirmation":false}
Coordinates must be pixels in the supplied image. Use target {x:0,y:0} when no coordinate is needed. Never use markdown fences or additional text."""


class VLMError(RuntimeError):
    """Raised when a VLM request or response is invalid."""


class VisionLanguageModel(ABC):
    @abstractmethod
    def decide(self, image_bytes: bytes, instruction: str) -> ActionCommand:
        raise NotImplementedError

    @abstractmethod
    def verify(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> VerificationResult:
        raise NotImplementedError


def _parse_command(content: str) -> ActionCommand:
    try:
        payload: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise VLMError("VLM did not return valid JSON") from exc
    try:
        return ActionCommand.model_validate(payload)
    except Exception as exc:
        raise VLMError(f"VLM JSON failed action schema validation: {exc}") from exc


def _parse_verification(content: str) -> VerificationResult:
    try:
        payload: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise VLMError("VLM did not return valid verification JSON") from exc
    try:
        return VerificationResult.model_validate(payload)
    except Exception as exc:
        raise VLMError(f"VLM verification JSON failed schema validation: {exc}") from exc


VERIFICATION_PROMPT = """Compare the before and after screenshots. Decide whether the stated expected outcome happened because of the action. Return ONLY valid JSON with exactly these fields: {\"verified\":true|false,\"reason\":\"brief evidence\"}. Do not infer success from the action alone."""


class OpenAICompatibleVLM(VisionLanguageModel):
    """Works with OpenAI, compatible gateways, and local OpenAI-compatible servers."""

    def __init__(self, endpoint: str, model: str, api_key: str, timeout: float = 45.0) -> None:
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def decide(self, image_bytes: bytes, instruction: str) -> ActionCommand:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ]},
            ],
        }
        try:
            response = httpx.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise VLMError(f"OpenAI-compatible VLM request failed: {exc}") from exc
        return _parse_command(content)

    def verify(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> VerificationResult:
        content = self._request_comparison(previous_image, current_image, expected_outcome)
        return _parse_verification(content)

    def _request_comparison(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> str:
        images = [base64.b64encode(image).decode("ascii") for image in (previous_image, current_image)]
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": VERIFICATION_PROMPT}, {"role": "user", "content": [
                {"type": "text", "text": f"Expected outcome: {expected_outcome}"},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in images],
            ]}],
        }
        try:
            response = httpx.post(self.endpoint, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise VLMError(f"OpenAI-compatible verification request failed: {exc}") from exc


class OllamaVLM(VisionLanguageModel):
    def __init__(self, model: str, endpoint: str = "http://localhost:11434", timeout: float = 60.0) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/") + "/api/chat"
        self.timeout = timeout

    def decide(self, image_bytes: bytes, instruction: str) -> ActionCommand:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction, "images": [base64.b64encode(image_bytes).decode("ascii")]}],
        }
        try:
            response = httpx.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise VLMError(
                "Ollama is not running at "
                f"{self.endpoint.removesuffix('/api/chat')}. Start Ollama and try again."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise VLMError(
                    f"Ollama model '{self.model}' was not found. Run 'ollama pull {self.model}'."
                ) from exc
            raise VLMError(f"Ollama request failed with HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise VLMError(f"Ollama request failed: {exc}") from exc
        return _parse_command(content)

    def verify(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> VerificationResult:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "system", "content": VERIFICATION_PROMPT}, {"role": "user", "content": f"Expected outcome: {expected_outcome}", "images": [base64.b64encode(image).decode("ascii") for image in (previous_image, current_image)]}],
        }
        try:
            response = httpx.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise VLMError(
                "Ollama is not running at "
                f"{self.endpoint.removesuffix('/api/chat')}. Start Ollama and try again."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise VLMError(f"Ollama verification failed with HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise VLMError(f"Ollama verification request failed: {exc}") from exc
        return _parse_verification(content)
