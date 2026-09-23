import sys
import base64
from io import BytesIO
from PIL import Image, ImageDraw

sys.path.insert(0, ".")

from desktop_bot.app import AppConfig
from desktop_bot.models import ScreenFrame
from desktop_bot.controller import DesktopController
from desktop_bot.loop import TaskRunner
from desktop_bot.memory import ChromaWorkflowMemory
from desktop_bot.vlm import OllamaVLM

class DummyCapturer:
    def capture(self, max_width=1920, jpeg_quality=82):
        img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([900, 500, 1020, 580], fill='blue')
        draw.text((910, 510), 'Start', fill='white')
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=82)
        return ScreenFrame(
            image_bytes=buf.getvalue(),
            width=1920,
            height=1080,
            scale_x=1.0,
            scale_y=1.0,
            monitor=1
        )

def main():
    print("Initializing...")
    model = OllamaVLM("llava", "http://localhost:11434")
    memory = ChromaWorkflowMemory("memory")
    capturer = DummyCapturer()
    controller = DesktopController()
    
    # Patch execute so it doesn't actually click things while we test
    controller.execute = lambda cmd: print(f"[MOCK EXECUTE] {cmd.action.value} at {cmd.target}")

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
