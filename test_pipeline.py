import sys
sys.path.insert(0, ".")

from desktop_bot.app import AppConfig
from desktop_bot.capture import ScreenCapturer
from desktop_bot.controller import DesktopController
from desktop_bot.loop import TaskRunner
from desktop_bot.memory import ChromaWorkflowMemory
from desktop_bot.vlm import OllamaVLM

def main():
    print("Initializing...")
    config = AppConfig("ollama", "llava", "http://localhost:11434", "", "memory")
    model = OllamaVLM(config.model, config.endpoint)
    memory = ChromaWorkflowMemory(config.memory_path)
    capturer = ScreenCapturer()
    controller = DesktopController()
    
    runner = TaskRunner(
        capture=capturer,
        model=model,
        controller=controller,
        memory=memory,
        replay_threshold=0.9,
    )
    
    print("Running task...")
    try:
        state = runner.run("Click the Start button", max_steps=2)
        print(f"\nFinal State: {state.status.value}")
        print(f"Failure Context: {state.failure_context}")
        print(f"Steps: {len(state.step_history)}")
    except Exception as e:
        print(f"CRASH: {e}")

if __name__ == "__main__":
    main()
