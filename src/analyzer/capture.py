"""
Screenshot capture from Android emulator via ADB.
"""
import subprocess
import time
from pathlib import Path
from typing import Optional
from PIL import Image
import io


class EmulatorCapture:
    """Capture screenshots from Android emulator using ADB."""

    def __init__(self, adb_path: Optional[str] = None):
        """
        Initialize the capture system.

        Args:
            adb_path: Path to adb executable. If None, assumes 'adb' is in PATH.
        """
        self.adb_path = adb_path or "adb"
        self._verify_connection()

    def _verify_connection(self):
        """Verify that an emulator is connected."""
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.strip().split('\n')
            devices = [l for l in lines[1:] if l.strip() and 'device' in l]

            if not devices:
                raise RuntimeError(
                    "No Android device/emulator connected. "
                    "Make sure the emulator is running and 'adb devices' shows it."
                )
            print(f"Connected to: {devices[0].split()[0]}")

        except FileNotFoundError:
            raise RuntimeError(
                f"ADB not found at '{self.adb_path}'. "
                "Make sure Android SDK platform-tools is in your PATH."
            )

    def capture(self) -> Image.Image:
        """
        Capture a screenshot from the emulator.

        Returns:
            PIL Image of the screenshot
        """
        result = subprocess.run(
            [self.adb_path, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=10
        )

        if result.returncode != 0:
            raise RuntimeError(f"Screenshot failed: {result.stderr.decode()}")

        # Load the PNG data directly into PIL
        image = Image.open(io.BytesIO(result.stdout))
        return image

    def capture_and_save(self, path: Path) -> Image.Image:
        """
        Capture a screenshot and save it to disk.

        Args:
            path: Where to save the screenshot

        Returns:
            PIL Image of the screenshot
        """
        image = self.capture()
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return image

    def tap(self, x: int, y: int):
        """
        Tap at a specific screen coordinate.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        subprocess.run(
            [self.adb_path, "shell", "input", "tap", str(x), str(y)],
            timeout=5
        )

    def get_screen_size(self) -> tuple[int, int]:
        """Get the screen resolution of the device."""
        result = subprocess.run(
            [self.adb_path, "shell", "wm", "size"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Output like: "Physical size: 1080x2400"
        size_str = result.stdout.strip().split()[-1]
        width, height = map(int, size_str.split('x'))
        return width, height


def continuous_capture(
    output_dir: Path,
    interval: float = 0.5,
    duration: Optional[float] = None,
    adb_path: Optional[str] = None
):
    """
    Continuously capture screenshots at regular intervals.

    Args:
        output_dir: Directory to save screenshots
        interval: Seconds between captures
        duration: Total duration in seconds (None = until interrupted)
        adb_path: Path to adb executable
    """
    capture = EmulatorCapture(adb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    frame_count = 0

    print(f"Capturing screenshots to {output_dir}")
    print("Press Ctrl+C to stop...")

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"frame_{frame_count:05d}_{timestamp}.png"

            try:
                capture.capture_and_save(filename)
                frame_count += 1
                print(f"Captured frame {frame_count}: {filename.name}")
            except Exception as e:
                print(f"Capture error: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStopped. Captured {frame_count} frames.")

    return frame_count


if __name__ == "__main__":
    # Quick test
    cap = EmulatorCapture()
    img = cap.capture()
    print(f"Captured image: {img.size}")
    img.save("test_screenshot.png")
    print("Saved to test_screenshot.png")
