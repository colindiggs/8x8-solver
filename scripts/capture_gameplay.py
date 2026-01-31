"""
Simple gameplay capture - takes screenshots every N seconds.
Run this while playing, then let Claude analyze the images.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzer.capture import EmulatorCapture
from datetime import datetime
import time

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "gameplay_captures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Connecting to emulator...")
    cap = EmulatorCapture()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {session_dir}")
    print("Will capture every 2 seconds while you play.")
    print("Press Ctrl+C to stop.\n")

    input("Press Enter to start capturing...")

    count = 0
    try:
        while True:
            img = cap.capture()
            path = session_dir / f"frame_{count:04d}.png"
            img.save(path)
            count += 1
            print(f"Captured frame {count}", end='\r')
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\n\nDone! Captured {count} frames to {session_dir}")
        print(f"\nNow ask Claude to analyze: {session_dir}")

if __name__ == "__main__":
    main()
