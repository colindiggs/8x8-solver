"""
Launch the Strategy Arena webapp.

Usage:
    python scripts/run_webapp.py

Then open http://127.0.0.1:8000 in your browser.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.arena_app import main

if __name__ == "__main__":
    print("=" * 60)
    print("  8x8 Block Puzzle Strategy Arena")
    print("=" * 60)
    print()
    print("Open http://127.0.0.1:8000 in your browser")
    print()
    print("Features:")
    print("  - 16 different AI strategies")
    print("  - Battle Royale mode (all vs all)")
    print("  - Tournament mode (round-robin)")
    print("  - Live game visualization")
    print("  - Explainable move decisions")
    print()
    main()
