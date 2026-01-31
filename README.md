# 8x8 Block Puzzle Solver

An AI solver for the "Color Blocks" 8x8 puzzle game that uses Deep Reinforcement Learning to discover optimal strategies.

## Project Status

**Phase 1: Reverse Engineering** - In Progress

We need to extract the exact game parameters from the APK before the simulation will be accurate.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Play a random game (for testing)
python -c "from src.core.game import play_random_game; print(f'Score: {play_random_game(verbose=True)}')"
```

## Project Structure

```
8x8-solver/
├── src/
│   ├── core/           # Game engine
│   │   ├── board.py    # 8x8 grid logic
│   │   ├── blocks.py   # Block definitions
│   │   ├── game.py     # Game loop
│   │   └── scoring.py  # Scoring system
│   ├── ai/             # AI agents
│   │   ├── agents/     # Random, heuristic, DQN
│   │   ├── networks/   # Neural networks
│   │   └── training/   # Training pipeline
│   └── simulation/     # Batch execution
├── api/                # FastAPI backend
├── ui/                 # React frontend
├── data/               # Game parameters, models
└── tests/              # Unit and integration tests
```

---

## Phase 1: APK Reverse Engineering Guide

### Prerequisites

1. **Android device** with the game installed
2. **USB Debugging** enabled on the device
3. **ADB** (Android Debug Bridge) installed
4. **JADX** decompiler installed

### Step 1: Extract the APK

```bash
# Connect your Android device via USB

# Find the package name
adb shell pm list packages | findstr jindoblu
# Should show: package:com.JindoBlu.OfflineGames

# Get the APK path
adb shell pm path com.JindoBlu.OfflineGames
# Shows something like: package:/data/app/com.JindoBlu.OfflineGames-xxx/base.apk

# Pull the APK
adb pull /data/app/com.JindoBlu.OfflineGames-xxx/base.apk game.apk
```

### Step 2: Decompile with JADX

```bash
# Using JADX CLI
jadx -d decompiled_output game.apk

# Or use JADX GUI for easier browsing
jadx-gui game.apk
```

### Step 3: Search for Game Logic

In the decompiled code, search for these keywords:

#### Block Shapes
```
Search for: "block", "piece", "shape", "tetris"
Look for: 2D arrays, lists of coordinates, shape definitions
```

#### Scoring
```
Search for: "score", "point", "combo", "multiplier", "bonus"
Look for: Math operations, score calculations, reward formulas
```

#### Board Size
```
Search for: "grid", "board", "8", "size", "width", "height"
Confirm: 8x8 grid size
```

#### Block Generation
```
Search for: "random", "weight", "probability", "spawn", "generate"
Look for: Random selection logic, weight arrays
```

### Step 4: Document Findings

Create/update `data/game_parameters.json` with:

1. **All block shapes** as 2D arrays
2. **Spawn weights** for each shape
3. **Scoring formulas**:
   - Points per cell placed
   - Points per line cleared
   - Combo multiplier formula
4. **Any special rules** discovered

### Step 5: Validate

After updating parameters:

```bash
# Run validation tests
pytest tests/ -v

# Play test games and compare scores with real app
python scripts/validate.py
```

---

## Alternative: Manual Parameter Discovery

If APK decompilation is difficult, use systematic gameplay testing:

### Block Shape Catalog
1. Play 50+ games
2. Screenshot every unique block shape
3. Document each as a 2D array
4. Track how often each appears

### Scoring Analysis
1. Start fresh game
2. Place single block → record points
3. Clear single line → record points
4. Clear 2 lines → record points
5. Deduce formula

### Testing Checklist
- [ ] All block shapes documented (expect 15-30 unique)
- [ ] Block spawn rates measured (500+ samples)
- [ ] Base scoring per cell confirmed
- [ ] Line clear bonus confirmed
- [ ] Combo multiplier formula deduced
- [ ] Edge cases tested (simultaneous clears, etc.)

---

## Development Roadmap

### Phase 1: Reverse Engineering ← Current
- [ ] Extract APK
- [ ] Decompile and analyze
- [ ] Document all parameters
- [ ] Update `game_parameters.json`
- [ ] Validate simulation accuracy

### Phase 2: Game Engine
- [x] Core board logic
- [x] Block definitions
- [x] Game loop
- [x] Scoring system
- [ ] Numba optimization
- [ ] Validation against real game

### Phase 3: AI Development
- [x] Random agent (baseline)
- [x] Heuristic agent
- [ ] DQN agent
- [ ] Training pipeline
- [ ] Evaluation framework

### Phase 4: Web UI
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Real-time visualization
- [ ] AI suggestions overlay

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

MIT License
