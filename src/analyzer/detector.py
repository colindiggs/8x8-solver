"""
Detect game state from screenshots using image processing.

This module analyzes screenshots to extract:
- Board state (8x8 grid of filled/empty cells)
- Available blocks (the 3 pieces to place)
- Score value
"""
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class GridRegion:
    """Defines a region of the screen containing the game grid."""
    x: int  # Top-left X
    y: int  # Top-left Y
    width: int  # Total width
    height: int  # Total height
    rows: int = 8
    cols: int = 8

    @property
    def cell_width(self) -> float:
        return self.width / self.cols

    @property
    def cell_height(self) -> float:
        return self.height / self.rows

    def get_cell_center(self, row: int, col: int) -> Tuple[int, int]:
        """Get the screen coordinates of a cell's center."""
        cx = self.x + (col + 0.5) * self.cell_width
        cy = self.y + (row + 0.5) * self.cell_height
        return int(cx), int(cy)

    def get_cell_bounds(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Get the bounding box of a cell (x1, y1, x2, y2)."""
        x1 = int(self.x + col * self.cell_width)
        y1 = int(self.y + row * self.cell_height)
        x2 = int(self.x + (col + 1) * self.cell_width)
        y2 = int(self.y + (row + 1) * self.cell_height)
        return x1, y1, x2, y2


@dataclass
class BlockRegion:
    """Defines a region containing one of the preview blocks."""
    x: int
    y: int
    width: int
    height: int
    max_cells: int = 3  # Max 3x3 block


@dataclass
class ScoreRegion:
    """Defines a region containing the score display."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class GameLayout:
    """Complete layout of game UI elements on screen."""
    grid: GridRegion
    blocks: List[BlockRegion]
    score: Optional[ScoreRegion] = None

    def to_dict(self) -> Dict:
        return {
            'grid': {
                'x': self.grid.x, 'y': self.grid.y,
                'width': self.grid.width, 'height': self.grid.height
            },
            'blocks': [
                {'x': b.x, 'y': b.y, 'width': b.width, 'height': b.height}
                for b in self.blocks
            ],
            'score': {
                'x': self.score.x, 'y': self.score.y,
                'width': self.score.width, 'height': self.score.height
            } if self.score else None
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'GameLayout':
        return cls(
            grid=GridRegion(**d['grid']),
            blocks=[BlockRegion(**b) for b in d['blocks']],
            score=ScoreRegion(**d['score']) if d.get('score') else None
        )

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'GameLayout':
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class DetectedState:
    """Detected game state from a screenshot."""
    board: np.ndarray  # 8x8 array, 1=filled, 0=empty
    blocks: List[np.ndarray]  # List of block shapes detected
    score: Optional[int] = None
    timestamp: float = 0.0
    raw_colors: Optional[np.ndarray] = None  # For debugging

    def board_hash(self) -> str:
        """Get a hash of the board state for comparison."""
        return self.board.tobytes().hex()

    def to_dict(self) -> Dict:
        return {
            'board': self.board.tolist(),
            'blocks': [b.tolist() for b in self.blocks],
            'score': self.score,
            'timestamp': self.timestamp
        }


class GameDetector:
    """
    Detects game state from screenshots.

    Usage:
        1. First, run calibrate() with a screenshot to define the layout
        2. Then use detect() to extract game state from screenshots
    """

    # Color detection thresholds
    EMPTY_CELL_THRESHOLD = 50  # How dark is "empty"
    FILLED_CELL_THRESHOLD = 100  # How bright is "filled"

    def __init__(self, layout: Optional[GameLayout] = None):
        """
        Initialize the detector.

        Args:
            layout: Pre-defined game layout. If None, must call calibrate().
        """
        self.layout = layout
        self.empty_color = None  # Will be detected during calibration
        self.filled_colors = []  # Different block colors

    def calibrate_interactive(self, image: Image.Image) -> GameLayout:
        """
        Interactively calibrate the detector by having the user click corners.

        This is a simplified version - in practice, you'd want a GUI.
        For now, we'll use automatic detection with manual override.
        """
        print("=== Calibration Mode ===")
        print(f"Image size: {image.size}")
        print("\nLooking for the game grid automatically...")

        # Try automatic detection first
        layout = self._auto_detect_layout(image)

        if layout:
            print(f"Found grid at: ({layout.grid.x}, {layout.grid.y})")
            print(f"Grid size: {layout.grid.width}x{layout.grid.height}")
            self.layout = layout
            return layout

        print("Auto-detection failed. Manual calibration needed.")
        raise NotImplementedError(
            "Manual calibration not implemented yet. "
            "Please provide grid coordinates manually."
        )

    def _auto_detect_layout(self, image: Image.Image) -> Optional[GameLayout]:
        """
        Attempt to automatically detect the game layout.

        Looks for:
        - A square grid region (the 8x8 board)
        - Three block preview areas below
        - Score display above
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # The grid is usually:
        # - Centered horizontally
        # - In the upper portion of the screen
        # - Square or near-square

        # Convert to grayscale for edge detection
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Look for a large square region with grid lines
        # This is a heuristic approach - may need adjustment per game

        # For now, return None to trigger manual calibration
        # We'll add smarter detection later
        return None

    def set_layout_manual(
        self,
        grid_x: int, grid_y: int,
        grid_width: int, grid_height: int,
        block_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        score_region: Optional[Tuple[int, int, int, int]] = None
    ) -> GameLayout:
        """
        Manually set the game layout.

        Args:
            grid_x, grid_y: Top-left corner of the 8x8 grid
            grid_width, grid_height: Size of the grid
            block_regions: List of (x, y, width, height) for block preview areas
            score_region: (x, y, width, height) for score display
        """
        grid = GridRegion(grid_x, grid_y, grid_width, grid_height)

        blocks = []
        if block_regions:
            for x, y, w, h in block_regions:
                blocks.append(BlockRegion(x, y, w, h))

        score = None
        if score_region:
            x, y, w, h = score_region
            score = ScoreRegion(x, y, w, h)

        self.layout = GameLayout(grid, blocks, score)
        return self.layout

    def detect(self, image: Image.Image) -> DetectedState:
        """
        Detect the game state from a screenshot.

        Args:
            image: PIL Image of the game screenshot

        Returns:
            DetectedState with board, blocks, and score
        """
        if not self.layout:
            raise RuntimeError("Detector not calibrated. Call calibrate_interactive() or set_layout_manual() first.")

        img_array = np.array(image)

        # Detect board state
        board = self._detect_board(img_array)

        # Detect available blocks
        blocks = self._detect_blocks(img_array)

        # Detect score (if region defined)
        score = None
        if self.layout.score:
            score = self._detect_score(img_array)

        return DetectedState(
            board=board,
            blocks=blocks,
            score=score
        )

    def _detect_board(self, img: np.ndarray) -> np.ndarray:
        """
        Detect which cells are filled on the 8x8 board.

        Returns:
            8x8 numpy array where 1=filled, 0=empty
        """
        board = np.zeros((8, 8), dtype=np.int8)
        grid = self.layout.grid

        for row in range(8):
            for col in range(8):
                x1, y1, x2, y2 = grid.get_cell_bounds(row, col)

                # Sample the center region of the cell (avoid edges)
                margin_x = int((x2 - x1) * 0.2)
                margin_y = int((y2 - y1) * 0.2)
                cell_region = img[y1+margin_y:y2-margin_y, x1+margin_x:x2-margin_x]

                if cell_region.size == 0:
                    continue

                # Calculate average brightness
                avg_color = np.mean(cell_region)

                # Also check color variance - filled cells tend to be more uniform
                color_std = np.std(cell_region)

                # Determine if cell is filled
                # This threshold may need tuning based on the game's colors
                is_filled = avg_color > self.FILLED_CELL_THRESHOLD

                board[row, col] = 1 if is_filled else 0

        return board

    def _detect_blocks(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Detect the shapes of available blocks.

        Returns:
            List of numpy arrays representing block shapes
        """
        blocks = []

        for block_region in self.layout.blocks:
            # Extract the block preview region
            x1, y1 = block_region.x, block_region.y
            x2, y2 = x1 + block_region.width, y1 + block_region.height

            region = img[y1:y2, x1:x2]

            if region.size == 0:
                blocks.append(np.zeros((1, 1), dtype=np.int8))
                continue

            # Detect the block shape within this region
            block_shape = self._detect_block_shape(region, block_region.max_cells)
            blocks.append(block_shape)

        return blocks

    def _detect_block_shape(self, region: np.ndarray, max_cells: int) -> np.ndarray:
        """
        Detect the shape of a single block from its preview region.
        """
        height, width = region.shape[:2]

        # Divide into a grid of potential cells
        cell_h = height // max_cells
        cell_w = width // max_cells

        shape = np.zeros((max_cells, max_cells), dtype=np.int8)

        for row in range(max_cells):
            for col in range(max_cells):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w

                cell = region[y1:y2, x1:x2]

                if cell.size == 0:
                    continue

                avg_brightness = np.mean(cell)
                shape[row, col] = 1 if avg_brightness > self.FILLED_CELL_THRESHOLD else 0

        # Trim empty rows/cols
        shape = self._trim_shape(shape)

        return shape

    def _trim_shape(self, shape: np.ndarray) -> np.ndarray:
        """Remove empty rows and columns from edges of shape."""
        # Find non-empty rows and cols
        non_empty_rows = np.any(shape, axis=1)
        non_empty_cols = np.any(shape, axis=0)

        if not np.any(non_empty_rows):
            return np.zeros((1, 1), dtype=np.int8)

        # Get bounds
        row_start = np.argmax(non_empty_rows)
        row_end = len(non_empty_rows) - np.argmax(non_empty_rows[::-1])
        col_start = np.argmax(non_empty_cols)
        col_end = len(non_empty_cols) - np.argmax(non_empty_cols[::-1])

        return shape[row_start:row_end, col_start:col_end]

    def _detect_score(self, img: np.ndarray) -> Optional[int]:
        """
        Detect the score value using OCR.

        Note: This requires additional OCR setup (pytesseract or easyocr).
        For now, returns None.
        """
        # TODO: Implement OCR-based score detection
        return None

    def visualize_layout(self, image: Image.Image) -> Image.Image:
        """
        Draw the detected layout on the image for debugging.
        """
        from PIL import ImageDraw

        img = image.copy()
        draw = ImageDraw.Draw(img)

        if not self.layout:
            return img

        # Draw grid
        grid = self.layout.grid
        draw.rectangle(
            [grid.x, grid.y, grid.x + grid.width, grid.y + grid.height],
            outline='red',
            width=3
        )

        # Draw cell grid lines
        for i in range(9):
            # Vertical lines
            x = grid.x + i * grid.cell_width
            draw.line([(x, grid.y), (x, grid.y + grid.height)], fill='red', width=1)
            # Horizontal lines
            y = grid.y + i * grid.cell_height
            draw.line([(grid.x, y), (grid.x + grid.width, y)], fill='red', width=1)

        # Draw block regions
        for i, block in enumerate(self.layout.blocks):
            draw.rectangle(
                [block.x, block.y, block.x + block.width, block.y + block.height],
                outline='blue',
                width=2
            )
            draw.text((block.x, block.y - 15), f"Block {i+1}", fill='blue')

        # Draw score region
        if self.layout.score:
            s = self.layout.score
            draw.rectangle(
                [s.x, s.y, s.x + s.width, s.y + s.height],
                outline='green',
                width=2
            )
            draw.text((s.x, s.y - 15), "Score", fill='green')

        return img


def analyze_color_regions(image: Image.Image) -> Dict[str, Any]:
    """
    Analyze an image to help with calibration.
    Returns information about color distributions and potential grid locations.
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Calculate brightness map
    if len(img_array.shape) == 3:
        brightness = np.mean(img_array, axis=2)
    else:
        brightness = img_array

    return {
        'size': (width, height),
        'brightness_mean': float(np.mean(brightness)),
        'brightness_std': float(np.std(brightness)),
        'brightness_min': float(np.min(brightness)),
        'brightness_max': float(np.max(brightness)),
    }
