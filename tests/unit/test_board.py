"""
Unit tests for the Board class.
"""
import pytest
import numpy as np

from src.core.board import Board, ClearResult


class TestBoard:
    """Tests for Board class."""

    def test_init_empty(self):
        """Test creating an empty board."""
        board = Board()
        assert board.grid.shape == (8, 8)
        assert np.all(board.grid == 0)
        assert board.get_empty_cells() == 64
        assert board.get_filled_cells() == 0

    def test_init_from_array(self):
        """Test creating a board from an existing array."""
        grid = np.zeros((8, 8), dtype=np.int8)
        grid[0, 0] = 1
        grid[7, 7] = 1

        board = Board(grid)
        assert board.grid[0, 0] == 1
        assert board.grid[7, 7] == 1
        assert board.get_filled_cells() == 2

    def test_can_place_simple(self):
        """Test basic placement validation."""
        board = Board()
        block = np.array([[1, 1], [1, 1]], dtype=np.int8)  # 2x2 square

        # Can place in corners
        assert board.can_place(block, 0, 0)
        assert board.can_place(block, 0, 6)
        assert board.can_place(block, 6, 0)
        assert board.can_place(block, 6, 6)

        # Cannot place out of bounds
        assert not board.can_place(block, 7, 7)
        assert not board.can_place(block, -1, 0)
        assert not board.can_place(block, 0, -1)

    def test_can_place_with_collision(self):
        """Test placement validation with existing pieces."""
        board = Board()

        # Place a block at (0, 0)
        block = np.array([[1, 1], [1, 1]], dtype=np.int8)
        board.place(block, 0, 0)

        # Cannot place overlapping block
        assert not board.can_place(block, 0, 0)
        assert not board.can_place(block, 0, 1)
        assert not board.can_place(block, 1, 0)
        assert not board.can_place(block, 1, 1)

        # Can place adjacent block
        assert board.can_place(block, 0, 2)
        assert board.can_place(block, 2, 0)

    def test_place_block(self):
        """Test placing a block on the board."""
        board = Board()
        block = np.array([[1, 1], [1, 1]], dtype=np.int8)

        result = board.place(block, 2, 3)

        assert result is True
        assert board.grid[2, 3] == 1
        assert board.grid[2, 4] == 1
        assert board.grid[3, 3] == 1
        assert board.grid[3, 4] == 1
        assert board.get_filled_cells() == 4

    def test_place_l_shape(self):
        """Test placing an L-shaped block."""
        board = Board()
        block = np.array([[1, 0], [1, 1]], dtype=np.int8)

        result = board.place(block, 0, 0)

        assert result is True
        assert board.grid[0, 0] == 1
        assert board.grid[0, 1] == 0
        assert board.grid[1, 0] == 1
        assert board.grid[1, 1] == 1

    def test_get_clearable_lines_row(self):
        """Test detecting a complete row."""
        board = Board()

        # Fill the first row
        board.grid[0, :] = 1

        rows, cols = board.get_clearable_lines()

        assert rows == [0]
        assert cols == []

    def test_get_clearable_lines_column(self):
        """Test detecting a complete column."""
        board = Board()

        # Fill the first column
        board.grid[:, 0] = 1

        rows, cols = board.get_clearable_lines()

        assert rows == []
        assert cols == [0]

    def test_get_clearable_lines_both(self):
        """Test detecting both row and column."""
        board = Board()

        # Fill first row and first column
        board.grid[0, :] = 1
        board.grid[:, 0] = 1

        rows, cols = board.get_clearable_lines()

        assert rows == [0]
        assert cols == [0]

    def test_clear_lines_row(self):
        """Test clearing a complete row."""
        board = Board()
        board.grid[0, :] = 1

        result = board.clear_lines()

        assert result.rows_cleared == [0]
        assert result.cols_cleared == []
        assert result.cells_cleared == 8
        assert np.all(board.grid[0, :] == 0)

    def test_clear_lines_multiple(self):
        """Test clearing multiple lines."""
        board = Board()
        board.grid[0, :] = 1
        board.grid[2, :] = 1
        board.grid[:, 3] = 1

        result = board.clear_lines()

        assert 0 in result.rows_cleared
        assert 2 in result.rows_cleared
        assert 3 in result.cols_cleared
        assert result.lines_cleared == 3

    def test_copy(self):
        """Test copying a board."""
        board = Board()
        board.grid[0, 0] = 1

        copy = board.copy()

        assert np.array_equal(board.grid, copy.grid)

        # Modify copy, original should not change
        copy.grid[1, 1] = 1
        assert board.grid[1, 1] == 0

    def test_get_valid_placements(self):
        """Test getting all valid placements for a block."""
        board = Board()
        block = np.array([[1, 1, 1]], dtype=np.int8)  # 1x3 horizontal line

        placements = board.get_valid_placements(block)

        # Should be able to place in 8 rows x 6 columns = 48 positions
        assert len(placements) == 8 * 6

    def test_get_valid_placements_with_obstacles(self):
        """Test getting valid placements with existing pieces."""
        board = Board()
        board.grid[4, :] = 1  # Fill row 4

        block = np.array([[1], [1], [1]], dtype=np.int8)  # 3x1 vertical line

        placements = board.get_valid_placements(block)

        # Cannot span row 4
        for row, col in placements:
            assert not (row <= 4 < row + 3)

    def test_str_representation(self):
        """Test string representation of board."""
        board = Board()
        board.grid[0, 0] = 1

        s = str(board)
        assert "#" in s  # Filled cell marker
        assert "." in s  # Empty cell marker


class TestClearResult:
    """Tests for ClearResult dataclass."""

    def test_lines_cleared(self):
        """Test lines_cleared property."""
        result = ClearResult(rows_cleared=[0, 1], cols_cleared=[2], cells_cleared=24)
        assert result.lines_cleared == 3

    def test_is_combo(self):
        """Test is_combo property."""
        single = ClearResult(rows_cleared=[0], cols_cleared=[], cells_cleared=8)
        combo = ClearResult(rows_cleared=[0, 1], cols_cleared=[], cells_cleared=16)

        assert not single.is_combo
        assert combo.is_combo
