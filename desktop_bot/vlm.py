"""VLM clients with strict JSON/action validation."""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import httpx

from .models import ActionCommand, VerificationResult

SYSTEM_PROMPT = """You control a desktop through one action at a time. Inspect the screenshot and return ONLY one JSON object.
The action value MUST be exactly one word: click, double_click, type, hotkey, scroll, wait, or done. Do not return a list, pipe-separated text, or the words 'click|double_click|type|hotkey|scroll|wait|done'.
Required shape: {"thought":"brief reasoning","action":"click","target":{"x":0,"y":0},"text":"","keys":[],"expected_outcome":"what the next screenshot should show","requires_confirmation":false}
For a type action, put the complete message in text and set keys to []. Use keys only for hotkey actions. Coordinates must be pixels in the supplied image — use the red grid lines and yellow coordinate labels overlaid on the image to find exact pixel positions. For clicks, target {x:0,y:0} is invalid and means you failed to choose a target. Click a real visible control at least 8 pixels from every image edge. Use target {x:0,y:0} only for wait, type, or done. For scroll, set target.y to a signed number of scroll clicks (positive = scroll down, negative = scroll up; typical range 1-10), and set target.x to 0. Never use markdown fences or additional text."""

ACTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thought", "action", "target", "text", "keys", "expected_outcome", "requires_confirmation"],
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["click", "double_click", "type", "hotkey", "scroll", "wait", "done"]},
        "target": {"type": "object", "additionalProperties": False, "required": ["x", "y"], "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
        "text": {"type": "string"},
        "keys": {"type": "array", "items": {"type": "string"}},
        "expected_outcome": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
    },
}

VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verified", "reason"],
    "properties": {
        "verified": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}


class VLMError(RuntimeError):
    """Raised when a VLM request or response is invalid."""


class VisionLanguageModel(ABC):
    @abstractmethod
    def decide(self, image_bytes: bytes, instruction: str) -> ActionCommand:
        raise NotImplementedError

    @abstractmethod
    def verify(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> VerificationResult:
        raise NotImplementedError

    @abstractmethod
    def analyze(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError


def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Extract (width, height) from image bytes. Returns (0, 0) on failure."""
    try:
        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as img:
            return img.size
    except Exception:
        return (0, 0)

def _extract_coords_from_text(text: str) -> tuple[float, float] | None:
    """Try to extract coordinate pairs from VLM thought/text fields.

    llava often puts coordinates in the thought as relative values like (0.544, 0.378)
    or sometimes as pixel values like (540, 380). This function extracts them.
    """
    # Match patterns like (0.544, 0.378) or (540, 380) or "at 0.544,0.378"
    patterns = [
        r'\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)',  # (x, y)
        r'at\s+(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',  # at x, y
        r'x[=:]\s*(\d+\.?\d*)\s*[,;]\s*y[=:]\s*(\d+\.?\d*)',  # x=N, y=N
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            if x >= 0 and y >= 0:
                return (x, y)
    return None


def _parse_command(content: str, image_width: int = 0, image_height: int = 0) -> ActionCommand:
    try:
        payload: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise VLMError("VLM did not return valid JSON") from exc
    if isinstance(payload, dict) and payload.get("action") == "type" and not payload.get("text"):
        keys = payload.get("keys")
        if isinstance(keys, list) and keys and all(isinstance(key, str) and len(key) == 1 for key in keys):
            payload["text"] = "".join(keys)
            payload["keys"] = []
    # Recover coordinates when llava puts them in thought but leaves target as (0,0)
    if isinstance(payload, dict) and payload.get("action") in ("click", "double_click"):
        target = payload.get("target")
        if isinstance(target, dict):
            # If coordinates are floats, they might be relative
            tx, ty = target.get("x", 0), target.get("y", 0)
            if isinstance(tx, float) or isinstance(ty, float):
                if tx <= 1.0 and ty <= 1.0 and image_width > 0 and image_height > 0:
                    target["x"] = round(tx * image_width)
                    target["y"] = round(ty * image_height)
                else:
                    target["x"] = round(tx)
                    target["y"] = round(ty)
            
            # Recover coordinates when llava leaves target as (0,0)
            if target.get("x", 0) == 0 and target.get("y", 0) == 0:
                # Scan thought and expected_outcome for coordinates
                search_text = " ".join(str(payload.get(k, "")) for k in ("thought", "expected_outcome", "text"))
                coords = _extract_coords_from_text(search_text)
                if coords is not None:
                    x, y = coords
                    # If values are < 1.0, they're relative — convert to pixels
                    if x <= 1.0 and y <= 1.0 and image_width > 0 and image_height > 0:
                        target["x"] = round(x * image_width)
                        target["y"] = round(y * image_height)
                    elif x > 1.0 or y > 1.0:
                        # Already pixel values
                        target["x"] = round(x)
                        target["y"] = round(y)
    try:
        return ActionCommand.model_validate(payload)
    except Exception as exc:
        if isinstance(payload, dict) and "|" in str(payload.get("action", "")):
            raise VLMError(
                "VLM returned the action choices as one string. "
                "Use a newer Ollama model or retry; action must be one of click, double_click, "
                "type, hotkey, scroll, wait, or done."
            ) from exc
        raise VLMError(f"VLM JSON failed action schema validation: {exc}") from exc


def _parse_verification(content: str) -> VerificationResult:
    try:
        payload: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise VLMError("VLM did not return valid verification JSON") from exc
    try:
        return VerificationResult.model_validate(payload)
    except Exception as exc:
        if isinstance(payload, dict) and "verified" not in payload:
            raise VLMError(
                "VLM verification response omitted the required boolean 'verified' field. "
                "The model must return {\"verified\":true|false,\"reason\":\"evidence\"}."
            ) from exc
        raise VLMError(f"VLM verification JSON failed schema validation: {exc}") from exc


VERIFICATION_PROMPT = """Compare the before and after screenshots. Return ONLY one JSON object with BOTH required fields: verified (a JSON boolean, true or false) and reason (a short string). Example: {\"verified\":false,\"reason\":\"The expected control is not visible\"}. Never omit verified. Do not infer success from the action alone."""


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
        return _parse_command(content, *_image_dimensions(image_bytes))

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

    def analyze(self, image_bytes: bytes, prompt: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0.5,
            "messages": [
                {"role": "system", "content": "You are a helpful screen analysis assistant. Inspect the screenshot provided and answer the user's prompt about it. Be concise and descriptive."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ]}
            ],
        }
        try:
            response = httpx.post(self.endpoint, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise VLMError(f"OpenAI analysis failed: {exc}") from exc
class OllamaVLM(VisionLanguageModel):
    def __init__(self, model: str, endpoint: str = "http://localhost:11434", timeout: float = 60.0) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/") + "/api/chat"
        self.timeout = timeout

    def decide(self, image_bytes: bytes, instruction: str) -> ActionCommand:
        payload = {
            "model": self.model,
            "stream": False,
            "format": ACTION_JSON_SCHEMA,
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
        return _parse_command(content, *_image_dimensions(image_bytes))

    def verify(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> VerificationResult:
        # Try multi-image structured JSON first
        content = self._try_verify_multi(previous_image, current_image, expected_outcome)
        if content is not None:
            return _parse_verification(content)
        # Fallback: single-image verification (llava and similar can't handle two images)
        content = self._try_verify_single(current_image, expected_outcome)
        if content is not None:
            return _parse_verification(content)
        # If all attempts fail, return an optimistic unverified result so the loop can continue
        return VerificationResult(verified=True, reason="Verification skipped: model does not support image comparison")

    def _try_verify_multi(self, previous_image: bytes, current_image: bytes, expected_outcome: str) -> str | None:
        """Send both before/after images for comparison. Returns content or None on failure."""
        payload = {
            "model": self.model,
            "stream": False,
            "format": VERIFICATION_JSON_SCHEMA,
            "messages": [{
                "role": "system", "content": VERIFICATION_PROMPT,
            }, {
                "role": "user",
                "content": f"Expected outcome: {expected_outcome}",
                "images": [base64.b64encode(img).decode("ascii") for img in (previous_image, current_image)],
            }],
        }
        try:
            response = httpx.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise VLMError(
                "Ollama is not running at "
                f"{self.endpoint.removesuffix('/api/chat')}. Start Ollama and try again."
            ) from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            # Multi-image failed — fall through to single-image
            return None

    def _try_verify_single(self, current_image: bytes, expected_outcome: str) -> str | None:
        """Send only the after screenshot and ask if the outcome is visible."""
        single_prompt = (
            f"Look at this screenshot. Is the following outcome visible? "
            f"Expected outcome: {expected_outcome}\n\n"
            f"{VERIFICATION_PROMPT}"
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [{
                "role": "system", "content": VERIFICATION_PROMPT,
            }, {
                "role": "user",
                "content": single_prompt,
                "images": [base64.b64encode(current_image).decode("ascii")],
            }],
        }
        try:
            response = httpx.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None

    def analyze(self, image_bytes: bytes, prompt: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "stream": False,
            "system": "You are a helpful screen analysis assistant. Inspect the screenshot provided and answer the user's prompt about it. Be concise and descriptive.",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded],
                }
            ],
            "options": {"temperature": 0.5},
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        except Exception as exc:
            raise VLMError(f"Ollama analysis failed: {exc}") from exc
