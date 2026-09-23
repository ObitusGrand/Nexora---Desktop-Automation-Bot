# Desktop Bot

Phase 1 provides a screen perception pipeline for a computer-using agent:

- `ScreenCapturer` captures a monitor with `mss` and emits resized JPEG frames.
- `add_coordinate_grid` annotates frames with pixel coordinates for visual localization.
- `OpenAICompatibleVLM` supports OpenAI-compatible vision APIs and local gateways.
- `OllamaVLM` supports Ollama multimodal models.
- Pydantic models strictly validate the action JSON returned by either VLM.
- `DesktopController` executes validated actions with `pyautogui.FAILSAFE`, bounds checks, payload limits, and micro-delays.
- `WindowManager` can activate a named Windows window through `pygetwindow`.
- `execute_dom_action` is an optional Playwright hook for selector-based browser actions.
- `TaskRunner` coordinates capture, planning, action execution, visual verification, and bounded retries.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

The capture and controller paths require a graphical Windows session. The Ollama adapter expects a multimodal model already installed, for example `ollama pull llava`.

## Run the application

```powershell
python -m pip install -e ".[dev]"
$env:DESKTOP_BOT_VLM = "ollama"
$env:DESKTOP_BOT_MODEL = "llava"
desktop-bot
```

The native control panel keeps the task prompt, provider, model, endpoint, replay threshold, current status, and readiness checklist in one place. API-compatible providers also need `DESKTOP_BOT_API_KEY`. The UI asks before replaying a matching workflow and before executing any action classified as destructive.

## Application checklist

- [ ] Create and activate the Python 3.11+ virtual environment.
- [ ] Run `python -m pip install -e ".[dev]"` successfully.
- [ ] Confirm `python -m desktop_bot.app` opens the control panel.
- [ ] Confirm the readiness checklist reports an active screen and local memory.
- [ ] For Ollama, run `ollama pull llava` and confirm the endpoint is available.
- [ ] For an OpenAI-compatible provider, set `DESKTOP_BOT_VLM`, `DESKTOP_BOT_ENDPOINT`, `DESKTOP_BOT_MODEL`, and `DESKTOP_BOT_API_KEY`.
- [ ] Run a harmless task such as opening a non-destructive application view.
- [ ] Confirm a repeated task offers replay and does not call the VLM after approval.
- [ ] Confirm a delete, payment, submit, or send action pauses for explicit confirmation.
- [ ] Run `python -m pytest` and `python -m compileall -q desktop_bot` before shipping.

The controller aborts when the mouse reaches a screen corner, rejects click coordinates outside the display, limits text and scroll payloads, and requires non-empty type/hotkey data. Upward scrolling uses a negative `target.y`; click targets are still required to be inside the display bounds.

For the optional browser hook, install the extra with `python -m pip install -e ".[dev,browser]"` and provide an existing Playwright `Page` object.

## Phase 3 loop

Create a `TaskRunner` with a `ScreenCapturer`, a VLM adapter, and a `DesktopController`. Each action is followed by a one-second capture-and-verify step. Failed verification is sent back to the planner with the failure evidence; after the configured retry limit, the state becomes `needs_intervention` instead of continuing to operate blindly.

## Phase 4 secure workflow memory

Install Phase 4 dependencies with `python -m pip install -e "."`. `ChromaWorkflowMemory` uses Chroma's embedded local `all-MiniLM-L6-v2` ONNX model and a persistent Chroma collection. The prompt is embedded locally but is not stored as a plaintext Chroma document; the replay trace and app metadata are encrypted with Fernet.

By default, the encryption key is created at `<memory path>\\memory.key` and permissions are restricted where the platform supports it. Set `DESKTOP_BOT_MEMORY_KEY` to a Fernet key when key management is provided externally. Keep the memory directory private and back it up only through an encrypted mechanism.

Pass a `ChromaWorkflowMemory` instance to `TaskRunner(memory=...)`. When similarity is at least `replay_threshold`, the caller supplies `replay_confirmation`; its UI should display exactly: `I have executed this workflow before. Would you like me to replay the recorded steps?` Replay executes the stored commands without another VLM planning request.

Destructive actions are blocked unless `human_confirmation` explicitly returns `True`. The guard covers planner-marked actions and language associated with deletion, submission, payment, purchases, transfers, publishing, and communications. A missing confirmation callback therefore fails closed into `needs_intervention`.
