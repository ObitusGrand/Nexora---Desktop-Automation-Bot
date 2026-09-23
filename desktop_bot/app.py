"""Small native control panel for running the desktop assistant."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

from .capture import ScreenCapturer
from .controller import DesktopController
from .loop import REPLAY_PROMPT, LoopStatus, TaskRunner
from .memory import ChromaWorkflowMemory, WorkflowMatch
from .models import ActionCommand
from .vlm import OllamaVLM, OpenAICompatibleVLM


@dataclass(frozen=True)
class AppConfig:
    provider: str
    model: str
    endpoint: str
    api_key: str
    memory_path: str


def direct_type_text(prompt: str) -> str | None:
    """Return text for an explicit focused-control typing command."""
    normalized = prompt.strip()
    prefixes = ("type:", "type ", "enter:", "enter ")
    for prefix in prefixes:
        if normalized.lower().startswith(prefix):
            text = normalized[len(prefix):].strip()
            return text or None
    return None


def load_config(environ: dict[str, str] | None = None) -> AppConfig:
    values = environ or os.environ
    provider = values.get("DESKTOP_BOT_VLM", "ollama").lower()
    return AppConfig(
        provider=provider,
        model=values.get("DESKTOP_BOT_MODEL", "llava"),
        endpoint=values.get("DESKTOP_BOT_ENDPOINT", "http://localhost:11434"),
        api_key=values.get("DESKTOP_BOT_API_KEY", ""),
        memory_path=values.get("DESKTOP_BOT_MEMORY_PATH", ".desktop_bot_memory"),
    )


class DesktopBotApp:
    """Tkinter UI that owns configuration, confirmation, and task execution."""

    def __init__(self, root: tk.Tk, config: AppConfig | None = None) -> None:
        self.root = root
        self.config = config or load_config()
        self.root.title("Desktop Bot")
        self.root.geometry("920x620")
        self.root.minsize(760, 520)
        self._busy = False
        self._status = tk.StringVar(value="Ready for a task")
        self._trace = tk.StringVar(value="No actions planned yet")
        self._provider = tk.StringVar(value=self.config.provider)
        self._model = tk.StringVar(value=self.config.model)
        self._endpoint = tk.StringVar(value=self.config.endpoint)
        self._threshold = tk.DoubleVar(value=0.90)
        self._build()
        self._refresh_checklist()

    def _build(self) -> None:
        self.root.configure(bg="#10151c")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#10151c")
        style.configure("Panel.TFrame", background="#18212b")
        style.configure("TLabel", background="#10151c", foreground="#d9e2ec")
        style.configure("Panel.TLabel", background="#18212b", foreground="#d9e2ec")
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#f5c26b")
        style.configure("Muted.TLabel", foreground="#8fa2b5")
        style.configure("Accent.TButton", background="#e59f45", foreground="#10151c", padding=(16, 9))

        outer = ttk.Frame(self.root, padding=28)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="DESKTOP BOT", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Local control panel for perception, replay, and safe execution", style="Muted.TLabel").pack(anchor="w", pady=(2, 20))

        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(1, weight=1)

        task_panel = ttk.Frame(content, style="Panel.TFrame", padding=20)
        task_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 14))
        ttk.Label(task_panel, text="Task", style="Panel.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(task_panel, text="Describe what you want to know about your screen.", style="Panel.TLabel").pack(anchor="w", pady=(4, 10))
        self.task = tk.Text(task_panel, height=7, wrap="word", bg="#0e141b", fg="#e8eef4", insertbackground="#f5c26b", relief="flat", padx=12, pady=12)
        self.task.pack(fill="x")
        controls = ttk.Frame(task_panel, style="Panel.TFrame")
        controls.pack(fill="x", pady=(16, 0))
        ttk.Label(controls, text="Replay threshold", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Scale(controls, from_=0.75, to=0.99, variable=self._threshold, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=12)
        self._threshold_label = tk.StringVar(value=f"{self._threshold.get():.2f}")
        self._threshold.trace_add("write", lambda *_: self._threshold_label.set(f"{self._threshold.get():.2f}"))
        ttk.Label(controls, textvariable=self._threshold_label, style="Panel.TLabel").grid(row=0, column=2)
        controls.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(task_panel, text="Analyze Screen", style="Accent.TButton", command=self.run_task)
        self.run_button.pack(anchor="w", pady=(20, 0))
        ttk.Label(task_panel, textvariable=self._status, style="Panel.TLabel", wraplength=500).pack(anchor="w", pady=(18, 0))
        ttk.Label(task_panel, textvariable=self._trace, style="Panel.TLabel", wraplength=500).pack(anchor="w", pady=(8, 0))

        config_panel = ttk.Frame(content, style="Panel.TFrame", padding=18)
        config_panel.grid(row=0, column=1, sticky="nsew")
        ttk.Label(config_panel, text="Connection", style="Panel.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 12))
        self._field(config_panel, "Provider", self._provider)
        self._field(config_panel, "Model", self._model)
        self._field(config_panel, "Endpoint", self._endpoint)
        ttk.Label(config_panel, text="API keys stay in environment variables.", style="Panel.TLabel", wraplength=280).pack(anchor="w", pady=(10, 0))

        checklist_panel = ttk.Frame(content, style="Panel.TFrame", padding=18)
        checklist_panel.grid(row=1, column=1, sticky="nsew", pady=(14, 0))
        ttk.Label(checklist_panel, text="Readiness checklist", style="Panel.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.checklist = ttk.Frame(checklist_panel, style="Panel.TFrame")
        self.checklist.pack(fill="both", expand=True, pady=(10, 0))

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label, style="Panel.TLabel").pack(anchor="w")
        ttk.Entry(parent, textvariable=variable).pack(fill="x", pady=(3, 10))

    def _refresh_checklist(self) -> None:
        for child in self.checklist.winfo_children():
            child.destroy()
        checks = [
            ("Python package", True, "Importable desktop_bot package"),
            ("Screen capture", self._probe_capture(), "Requires an active Windows display"),
            ("VLM configuration", self._probe_vlm(), f"{self._provider.get()} / {self._model.get()}"),
            ("Local memory", self._probe_memory(), "Encrypted Chroma directory"),
            ("HITL guard", True, "Destructive actions fail closed"),
        ]
        for label, ready, detail in checks:
            color = "#74d39a" if ready else "#e59f45"
            ttk.Label(self.checklist, text=("READY  " if ready else "CHECK  ") + label, style="Panel.TLabel", foreground=color).pack(anchor="w")
            ttk.Label(self.checklist, text=detail, style="Panel.TLabel", foreground="#8fa2b5", wraplength=280).pack(anchor="w", pady=(0, 9))

    def _probe_capture(self) -> bool:
        try:
            import mss
            with mss.mss() as session:
                return len(session.monitors) > 1
        except Exception:
            return False

    def _probe_memory(self) -> bool:
        try:
            import chromadb
            from cryptography.fernet import Fernet
            return bool(chromadb and Fernet)
        except ImportError:
            return False

    def _probe_vlm(self) -> bool:
        if not self._model.get().strip():
            return False
        if self._provider.get().lower() != "ollama":
            return bool(self.config.api_key.strip())
        try:
            import httpx
            endpoint = self._endpoint.get().rstrip("/") + "/api/tags"
            response = httpx.get(endpoint, timeout=1.5)
            response.raise_for_status()
            models = {item.get("name", "").split(":", 1)[0] for item in response.json().get("models", [])}
            return self._model.get().split(":", 1)[0] in models
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False

    def run_task(self) -> None:
        prompt = self.task.get("1.0", "end").strip()
        if not prompt or self._busy:
            return
        self._busy = True
        self.run_button.configure(state="disabled")
        self._status.set("Starting capture and planning...")
        threading.Thread(target=self._run_worker, args=(prompt,), daemon=True).start()

    def _run_worker(self, prompt: str) -> None:
        try:

            # Minimize the bot window so it doesn't appear in screenshots or steal clicks
            self._minimize_window()
            import time
            time.sleep(0.5)  # wait for minimize animation to complete
            self._set_status("Connecting to the vision model...")
            config = AppConfig(self._provider.get(), self._model.get(), self._endpoint.get(), self.config.api_key, self.config.memory_path)
            model = self._make_model(config)
            self._set_status("Capturing the screen...")
            capturer = ScreenCapturer()
            frame = capturer.capture()
            
            self._set_status("Analyzing the screen...")
            analysis = model.analyze(frame.image_bytes, prompt)
            
            self._restore_window()
            self.root.after(0, self._show_analysis, analysis)
            self.root.after(0, self._finish, "completed", "Screen analyzed successfully.", "")
        except Exception as exc:
            self._restore_window()
            self.root.after(0, self._finish, "failed", str(exc), "No action completed")

    def _show_analysis(self, text: str) -> None:
        top = tk.Toplevel(self.root)
        top.title("Screen Analysis")
        top.geometry("600x400")
        txt = tk.Text(top, wrap="word", padx=10, pady=10, font=("Segoe UI", 11))
        txt.insert("1.0", text)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)

    def _minimize_window(self) -> None:
        """Minimize the bot window so it's not visible in screenshots."""
        done = threading.Event()
        def _do():
            self.root.iconify()
            done.set()
        self.root.after(0, _do)
        done.wait(timeout=2.0)

    def _restore_window(self) -> None:
        """Restore the bot window after the task completes."""
        def _do():
            self.root.deiconify()
            self.root.lift()
        self.root.after(0, _do)

    def _make_model(self, config: AppConfig):
        provider = config.provider.lower()
        if provider == "ollama":
            return OllamaVLM(config.model, config.endpoint)
        if provider in {"openai", "openai-compatible"}:
            return OpenAICompatibleVLM(config.endpoint, config.model, config.api_key)
        raise ValueError("Provider must be ollama or openai-compatible")

    def _confirm_replay(self, match: WorkflowMatch) -> bool:
        return self._ask_user("Workflow found", f"{REPLAY_PROMPT}\n\nSimilarity: {match.similarity:.0%}")

    def _confirm_action(self, command) -> bool:
        return self._ask_user("Confirm sensitive action", f"Allow this action?\n\n{command.action.value}: {command.expected_outcome}")

    def _ask_user(self, title: str, message: str) -> bool:
        """Show Tk dialogs on the main thread while the worker waits for an answer."""
        result = {"answer": False}
        completed = threading.Event()

        def ask() -> None:
            result["answer"] = messagebox.askyesno(title, message, parent=self.root)
            completed.set()

        self.root.after(0, ask)
        completed.wait()
        return result["answer"]

    def _set_status(self, message: str) -> None:
        self.root.after(0, self._status.set, message)

    def _finish(self, status: str, detail: str, trace: str = "") -> None:
        self._busy = False
        self.run_button.configure(state="normal")
        self._status.set(f"{status.upper()}: {detail}")
        if trace:
            self._trace.set(trace)
        self._refresh_checklist()


def main() -> None:
    root = tk.Tk()
    DesktopBotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()