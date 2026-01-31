"""
Interactive workflow for discovering game rules through empirical observation.

This script helps you:
1. Calibrate the screen regions (grid, blocks, score)
2. Capture and catalog unique block shapes
3. Record scoring observations
4. Export discovered rules to JSON

Run with: python scripts/discover_rules.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image
from analyzer.capture import EmulatorCapture
from analyzer.detector import GameDetector, GameLayout, GridRegion, BlockRegion, ScoreRegion
import numpy as np


# Where to save discovered data
DATA_DIR = Path(__file__).parent.parent / "data" / "discovered"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LAYOUT_FILE = DATA_DIR / "layout.json"
BLOCKS_FILE = DATA_DIR / "discovered_blocks.json"
SCORING_FILE = DATA_DIR / "scoring_observations.json"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_or_create_layout() -> Optional[GameLayout]:
    """Load existing layout or return None."""
    if LAYOUT_FILE.exists():
        return GameLayout.load(LAYOUT_FILE)
    return None


def save_layout(layout: GameLayout):
    """Save layout to file."""
    layout.save(LAYOUT_FILE)
    print(f"Layout saved to {LAYOUT_FILE}")


def calibration_workflow(capture: EmulatorCapture):
    """
    Interactive calibration to define screen regions.

    Takes a screenshot and guides user through defining:
    - The 8x8 game grid bounds
    - The 3 block preview regions
    - The score display region
    """
    print("\n" + "="*60)
    print("CALIBRATION WORKFLOW")
    print("="*60)

    # Capture screenshot
    print("\nCapturing screenshot from emulator...")
    print("Make sure the game is open and showing the main game board!")
    input("Press Enter when ready...")

    img = capture.capture()
    screenshot_path = SCREENSHOTS_DIR / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img.save(screenshot_path)
    print(f"Screenshot saved: {screenshot_path}")
    print(f"Image size: {img.size}")

    print("\n" + "-"*40)
    print("STEP 1: Define the 8x8 game grid")
    print("-"*40)
    print("""
Open the screenshot in an image viewer (e.g., Windows Photos, Paint).
Find the 8x8 game grid and note the pixel coordinates of:
- Top-left corner (x, y)
- Width and height of the entire grid

Tip: In Paint, hover over corners to see coordinates.
    """)

    grid_x = int(input("Grid top-left X: "))
    grid_y = int(input("Grid top-left Y: "))
    grid_width = int(input("Grid width (pixels): "))
    grid_height = int(input("Grid height (pixels): "))

    print("\n" + "-"*40)
    print("STEP 2: Define the 3 block preview regions")
    print("-"*40)
    print("""
Now find the 3 block previews (usually below or beside the grid).
For each block region, provide the bounding box.
    """)

    block_regions = []
    for i in range(3):
        print(f"\nBlock {i+1}:")
        bx = int(input(f"  Block {i+1} top-left X: "))
        by = int(input(f"  Block {i+1} top-left Y: "))
        bw = int(input(f"  Block {i+1} width: "))
        bh = int(input(f"  Block {i+1} height: "))
        block_regions.append(BlockRegion(bx, by, bw, bh))

    print("\n" + "-"*40)
    print("STEP 3: Define the score region (optional)")
    print("-"*40)
    has_score = input("Do you want to define a score region? (y/n): ").lower() == 'y'

    score_region = None
    if has_score:
        sx = int(input("Score region top-left X: "))
        sy = int(input("Score region top-left Y: "))
        sw = int(input("Score region width: "))
        sh = int(input("Score region height: "))
        score_region = ScoreRegion(sx, sy, sw, sh)

    # Create layout
    layout = GameLayout(
        grid=GridRegion(grid_x, grid_y, grid_width, grid_height),
        blocks=block_regions,
        score=score_region
    )

    # Visualize and confirm
    detector = GameDetector(layout)
    annotated = detector.visualize_layout(img)
    annotated_path = SCREENSHOTS_DIR / "calibration_annotated.png"
    annotated.save(annotated_path)

    print(f"\nAnnotated screenshot saved: {annotated_path}")
    print("Please check the annotated image to verify the regions are correct.")

    if input("\nSave this layout? (y/n): ").lower() == 'y':
        save_layout(layout)
        return layout
    else:
        print("Layout not saved. Run calibration again.")
        return None


def block_discovery_workflow(capture: EmulatorCapture, layout: GameLayout):
    """
    Semi-automated workflow to discover all unique block shapes.

    Takes continuous screenshots and detects new block shapes.
    """
    print("\n" + "="*60)
    print("BLOCK DISCOVERY WORKFLOW")
    print("="*60)

    detector = GameDetector(layout)

    # Load existing blocks
    discovered_blocks = []
    if BLOCKS_FILE.exists():
        with open(BLOCKS_FILE) as f:
            data = json.load(f)
            discovered_blocks = [np.array(b) for b in data.get('shapes', [])]

    print(f"\nCurrently discovered: {len(discovered_blocks)} unique shapes")
    print("""
This will continuously capture screenshots and detect block shapes.
Play the game normally - new unique blocks will be saved automatically.

Press Ctrl+C to stop.
    """)

    input("Press Enter to start capturing...")

    def shapes_match(a: np.ndarray, b: np.ndarray) -> bool:
        """Check if two shapes are the same."""
        if a.shape != b.shape:
            return False
        return np.array_equal(a, b)

    def is_new_shape(shape: np.ndarray) -> bool:
        """Check if this shape is new."""
        if shape.sum() == 0:  # Empty shape
            return False
        for existing in discovered_blocks:
            if shapes_match(shape, existing):
                return False
        return True

    def shape_to_ascii(shape: np.ndarray) -> str:
        """Convert shape to ASCII art."""
        lines = []
        for row in shape:
            lines.append(''.join('#' if c else '.' for c in row))
        return '\n'.join(lines)

    capture_count = 0
    new_shapes_this_session = 0

    try:
        while True:
            try:
                img = capture.capture()
                state = detector.detect(img)
                capture_count += 1

                for i, block in enumerate(state.blocks):
                    if is_new_shape(block):
                        discovered_blocks.append(block)
                        new_shapes_this_session += 1

                        print(f"\n{'='*40}")
                        print(f"NEW SHAPE #{len(discovered_blocks)} DISCOVERED!")
                        print(f"{'='*40}")
                        print(shape_to_ascii(block))
                        print(f"\nTotal: {len(discovered_blocks)} shapes")

                        # Save immediately
                        with open(BLOCKS_FILE, 'w') as f:
                            json.dump({
                                'shapes': [b.tolist() for b in discovered_blocks],
                                'count': len(discovered_blocks),
                                'last_updated': datetime.now().isoformat()
                            }, f, indent=2)

                # Status update every 10 captures
                if capture_count % 10 == 0:
                    print(f"  [Captured {capture_count} frames, {len(discovered_blocks)} unique shapes]", end='\r')

                time.sleep(0.5)  # Don't spam too fast

            except Exception as e:
                print(f"\nCapture error: {e}")
                time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n\nStopped. Discovered {new_shapes_this_session} new shapes this session.")
        print(f"Total unique shapes: {len(discovered_blocks)}")
        print(f"Saved to: {BLOCKS_FILE}")


def scoring_observation_workflow():
    """
    Manual workflow to document scoring observations.
    """
    print("\n" + "="*60)
    print("SCORING OBSERVATION WORKFLOW")
    print("="*60)
    print("""
This helps you document scoring rules by recording observations.
Play the game and note what happens when you:
- Place a block (note cells placed, points gained)
- Clear a single line
- Clear multiple lines at once
- Get combos (consecutive line clears)
    """)

    # Load existing observations
    observations = []
    if SCORING_FILE.exists():
        with open(SCORING_FILE) as f:
            observations = json.load(f).get('observations', [])

    print(f"\nExisting observations: {len(observations)}")
    print("\nEnter observations in format: <action> -> <points>")
    print("Examples:")
    print("  placed 3 cells -> 3")
    print("  cleared 1 row -> 10")
    print("  cleared 2 rows same turn -> 30")
    print("  combo x2 -> 5")
    print("\nType 'done' to finish, 'show' to see all observations.\n")

    while True:
        entry = input("Observation: ").strip()

        if entry.lower() == 'done':
            break
        elif entry.lower() == 'show':
            print("\n--- All Observations ---")
            for i, obs in enumerate(observations, 1):
                print(f"{i}. {obs['action']} -> {obs['points']} pts")
                if obs.get('notes'):
                    print(f"   Notes: {obs['notes']}")
            print()
            continue
        elif not entry:
            continue

        try:
            if '->' in entry:
                action, points = entry.split('->')
                action = action.strip()
                points = int(points.strip())
            else:
                action = entry
                points = int(input("  Points gained: "))

            notes = input("  Notes (optional): ").strip() or None

            observations.append({
                'action': action,
                'points': points,
                'notes': notes,
                'timestamp': datetime.now().isoformat()
            })

            # Save after each observation
            with open(SCORING_FILE, 'w') as f:
                json.dump({
                    'observations': observations,
                    'count': len(observations),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)

            print(f"  Saved! ({len(observations)} total observations)")

        except ValueError:
            print("  Invalid format. Try again.")

    print(f"\nSaved {len(observations)} observations to {SCORING_FILE}")


def analyze_discoveries():
    """
    Analyze what we've discovered so far.
    """
    print("\n" + "="*60)
    print("DISCOVERY ANALYSIS")
    print("="*60)

    # Blocks
    if BLOCKS_FILE.exists():
        with open(BLOCKS_FILE) as f:
            data = json.load(f)
            shapes = data.get('shapes', [])

        print(f"\nDISCOVERED BLOCKS: {len(shapes)}")
        print("-"*40)

        for i, shape in enumerate(shapes, 1):
            arr = np.array(shape)
            cells = arr.sum()
            dims = f"{arr.shape[0]}x{arr.shape[1]}"
            print(f"\nBlock #{i} ({cells} cells, {dims}):")
            for row in arr:
                print("  " + ''.join('##' if c else '..' for c in row))
    else:
        print("\nNo blocks discovered yet. Run block discovery first.")

    # Scoring
    if SCORING_FILE.exists():
        with open(SCORING_FILE) as f:
            data = json.load(f)
            obs = data.get('observations', [])

        print(f"\n\nSCORING OBSERVATIONS: {len(obs)}")
        print("-"*40)

        for o in obs:
            print(f"  {o['action']} -> {o['points']} pts")
            if o.get('notes'):
                print(f"    ({o['notes']})")
    else:
        print("\nNo scoring observations yet.")

    # Layout
    if LAYOUT_FILE.exists():
        print(f"\n\nLAYOUT: Calibrated ✓")
        layout = GameLayout.load(LAYOUT_FILE)
        print(f"  Grid: {layout.grid.width}x{layout.grid.height} at ({layout.grid.x}, {layout.grid.y})")
        print(f"  Block regions: {len(layout.blocks)}")
    else:
        print("\n\nLAYOUT: Not calibrated")


def export_to_game_config():
    """
    Export discovered rules to the game configuration format.
    """
    print("\n" + "="*60)
    print("EXPORT TO GAME CONFIG")
    print("="*60)

    if not BLOCKS_FILE.exists():
        print("No blocks discovered yet. Run block discovery first.")
        return

    with open(BLOCKS_FILE) as f:
        shapes = json.load(f).get('shapes', [])

    # Convert to the format used by blocks.py
    blocks_config = {
        'blocks': []
    }

    for i, shape in enumerate(shapes):
        blocks_config['blocks'].append({
            'name': f'discovered_{i+1}',
            'shape': shape,
            'weight': 1.0  # Default weight, adjust based on observations
        })

    output_path = DATA_DIR / "discovered_blocks_config.json"
    with open(output_path, 'w') as f:
        json.dump(blocks_config, f, indent=2)

    print(f"Exported {len(shapes)} blocks to {output_path}")
    print("\nTo use these blocks, update src/core/blocks.py to load from this file.")


def main():
    """Main menu."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           8x8 Color Blocks - Rule Discovery Tool             ║
╠══════════════════════════════════════════════════════════════╣
║  Instead of reverse engineering the APK, we'll discover      ║
║  the rules empirically by playing and observing!             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    capture = None
    layout = load_or_create_layout()

    while True:
        print("\n" + "-"*40)
        print("MAIN MENU")
        print("-"*40)

        status = "✓ Calibrated" if layout else "✗ Not calibrated"
        print(f"  Layout: {status}")

        if BLOCKS_FILE.exists():
            with open(BLOCKS_FILE) as f:
                count = json.load(f).get('count', 0)
            print(f"  Blocks: {count} discovered")

        print()
        print("  1. Calibrate screen regions")
        print("  2. Discover block shapes (auto-capture)")
        print("  3. Record scoring observations")
        print("  4. Analyze discoveries")
        print("  5. Export to game config")
        print("  6. Take test screenshot")
        print("  0. Exit")

        choice = input("\nChoice: ").strip()

        if choice == '0':
            print("Goodbye!")
            break

        elif choice == '1':
            if capture is None:
                try:
                    capture = EmulatorCapture()
                except Exception as e:
                    print(f"Error connecting to emulator: {e}")
                    continue
            layout = calibration_workflow(capture)

        elif choice == '2':
            if not layout:
                print("Please calibrate first (option 1)")
                continue
            if capture is None:
                try:
                    capture = EmulatorCapture()
                except Exception as e:
                    print(f"Error connecting to emulator: {e}")
                    continue
            block_discovery_workflow(capture, layout)

        elif choice == '3':
            scoring_observation_workflow()

        elif choice == '4':
            analyze_discoveries()

        elif choice == '5':
            export_to_game_config()

        elif choice == '6':
            if capture is None:
                try:
                    capture = EmulatorCapture()
                except Exception as e:
                    print(f"Error connecting to emulator: {e}")
                    continue
            img = capture.capture()
            path = SCREENSHOTS_DIR / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            img.save(path)
            print(f"Screenshot saved: {path}")
            print(f"Size: {img.size}")


if __name__ == "__main__":
    main()
