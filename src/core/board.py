"""
Board representation for the 8x8 block puzzle game.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ClearResult:
    """Result of clearing lines from the board."""
    rows_cleared: List[int]
    cols_cleared: List[int]
    cells_cleared: int

    @property
    def lines_cleared(self) -> int:
        return len(self.rows_cleared) + len(self.cols_cleared)

    @property
    def is_combo(self) -> bool:
        return self.lines_cleared > 1


class Board:
    """
    8x8 game board for the block puzzle.

    The board uses a numpy array where:
    - 0 = empty cell
    - 1 = filled cell
    """

    SIZE = 8

    def __init__(self, grid: Optional[np.ndarray] = None):
        """Initialize the board, optionally from an existing grid."""
        if grid is not None:
            if grid.shape != (self.SIZE, self.SIZE):
                raise ValueError(f"Grid must be {self.SIZE}x{self.SIZE}")
            self.grid = grid.astype(np.int8)
        else:
            self.grid = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)

    def copy(self) -> Board:
        """Create a deep copy of the board."""
        return Board(self.grid.copy())

    def can_place(self, block: np.ndarray, row: int, col: int) -> bool:
        """
        Check if a block can be placed at the given position.

        Args:
            block: 2D numpy array representing the block shape (1s and 0s)
            row: Top-left row position for placement
            col: Top-left column position for placement

        Returns:
            True if the block can be placed, False otherwise
        """
        block_h, block_w = block.shape

        # Check bounds
        if row < 0 or col < 0:
            return False
        if row + block_h > self.SIZE or col + block_w > self.SIZE:
            return False

        # Check for collisions with existing pieces
        board_section = self.grid[row:row + block_h, col:col + block_w]

        # If any cell where block has a 1 is already filled, can't place
        overlap = np.logical_and(block == 1, board_section == 1)
        return not np.any(overlap)

    def place(self, block: np.ndarray, row: int, col: int) -> bool:
        """
        Place a block on the board.

        Args:
            block: 2D numpy array representing the block shape
            row: Top-left row position
            col: Top-left column position

        Returns:
            True if placement was successful, False otherwise
        """
        if not self.can_place(block, row, col):
            return False

        block_h, block_w = block.shape

        # Place the block
        for i in range(block_h):
            for j in range(block_w):
                if block[i, j] == 1:
                    self.grid[row + i, col + j] = 1

        return True

    def get_clearable_lines(self) -> Tuple[List[int], List[int]]:
        """
        Find all rows and columns that are completely filled.

        Returns:
            Tuple of (row_indices, col_indices) that are complete
        """
        # Check rows (axis=1 means sum across columns for each row)
        complete_rows = np.where(np.all(self.grid == 1, axis=1))[0].tolist()

        # Check columns (axis=0 means sum across rows for each column)
        complete_cols = np.where(np.all(self.grid == 1, axis=0))[0].tolist()

        return complete_rows, complete_cols

    def clear_lines(self) -> ClearResult:
        """
        Clear all complete rows and columns.

        Returns:
            ClearResult with information about what was cleared
        """
        rows, cols = self.get_clearable_lines()

        # Count cells that will be cleared (accounting for overlap)
        cells_cleared = 0

        # Create a mask of cells to clear
        clear_mask = np.zeros_like(self.grid, dtype=bool)

        for row in rows:
            clear_mask[row, :] = True
        for col in cols:
            clear_mask[:, col] = True

        cells_cleared = np.sum(clear_mask & (self.grid == 1))

        # Clear the cells
        self.grid[clear_mask] = 0

        return ClearResult(
            rows_cleared=rows,
            cols_cleared=cols,
            cells_cleared=int(cells_cleared)
        )

    def get_empty_cells(self) -> int:
        """Return the count of empty cells."""
        return int(np.sum(self.grid == 0))

    def get_filled_cells(self) -> int:
        """Return the count of filled cells."""
        return int(np.sum(self.grid == 1))

    def get_valid_placements(self, block: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get all valid (row, col) positions where the block can be placed.

        Args:
            block: 2D numpy array representing the block shape

        Returns:
            List of (row, col) tuples where placement is valid
        """
        block_h, block_w = block.shape
        valid = []

        for row in range(self.SIZE - block_h + 1):
            for col in range(self.SIZE - block_w + 1):
                if self.can_place(block, row, col):
                    valid.append((row, col))

        return valid

    def has_any_valid_placement(self, block: np.ndarray) -> bool:
        """
        Check if there's at least one valid placement for the block.
        More efficient than get_valid_placements when you only need to know if any exist.
        """
        block_h, block_w = block.shape

        for row in range(self.SIZE - block_h + 1):
            for col in range(self.SIZE - block_w + 1):
                if self.can_place(block, row, col):
                    return True

        return False

    def to_array(self) -> np.ndarray:
        """Return a copy of the grid as a numpy array."""
        return self.grid.copy()

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Board:
        """Create a Board from a numpy array."""
        return cls(arr)

    def __str__(self) -> str:
        """Return a string representation of the board."""
        lines = []
        lines.append("+" + "-" * (self.SIZE * 2 + 1) + "+")
        for row in self.grid:
            row_str = "| " + " ".join("#" if cell else "." for cell in row) + " |"
            lines.append(row_str)
        lines.append("+" + "-" * (self.SIZE * 2 + 1) + "+")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board(filled={self.get_filled_cells()}/64)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return False
        return np.array_equal(self.grid, other.grid)

    def __hash__(self) -> int:
        return hash(self.grid.tobytes())
