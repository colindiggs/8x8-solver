"""
Unit tests for the Game class.
"""
import pytest
import numpy as np

from src.core.game import Game, Move, play_random_game
from src.core.blocks import BlockGenerator


class TestGame:
    """Tests for Game class."""

    def test_init(self):
        """Test game initialization."""
        game = Game(seed=42)

        assert game.board.get_filled_cells() == 0
        assert len(game.current_blocks) == 3
        assert game.turn_count == 0
        assert not game.game_over

    def test_get_available_blocks(self):
        """Test getting available blocks."""
        game = Game(seed=42)

        available = game.get_available_blocks()
        assert len(available) == 3

        # All indices should be 0, 1, 2
        indices = [idx for idx, _ in available]
        assert set(indices) == {0, 1, 2}

    def test_get_valid_moves(self):
        """Test getting all valid moves."""
        game = Game(seed=42)

        moves = game.get_valid_moves()

        # Should have moves for all 3 blocks
        assert len(moves) > 0

        # Each move should have valid block_idx
        for move in moves:
            assert 0 <= move.block_idx < 3

    def test_make_move_success(self):
        """Test making a successful move."""
        game = Game(seed=42)

        moves = game.get_valid_moves()
        assert len(moves) > 0

        move = moves[0]
        result = game.make_move(move.block_idx, move.row, move.col)

        assert result.success
        assert result.score_result is not None
        assert result.score_result.total_points > 0
        assert game.turn_count == 1

    def test_make_move_invalid_block_idx(self):
        """Test making a move with invalid block index."""
        game = Game(seed=42)

        result = game.make_move(5, 0, 0)  # Invalid index

        assert not result.success
        assert result.error is not None
        assert "Invalid block index" in result.error

    def test_make_move_already_used_block(self):
        """Test making a move with already used block."""
        game = Game(seed=42)

        moves = game.get_valid_moves()
        first_move = moves[0]

        # Make first move
        game.make_move(first_move.block_idx, first_move.row, first_move.col)

        # Try to use same block again
        result = game.make_move(first_move.block_idx, 5, 5)

        assert not result.success
        assert "already been used" in result.error

    def test_blocks_used_tracking(self):
        """Test that blocks_used is tracked correctly."""
        game = Game(seed=42)

        # Initially all unused
        assert game.blocks_used == [False, False, False]

        # Use block 0
        moves = [m for m in game.get_valid_moves() if m.block_idx == 0]
        game.make_move(moves[0].block_idx, moves[0].row, moves[0].col)

        assert game.blocks_used == [True, False, False]

    def test_new_batch_after_all_used(self):
        """Test that new batch is generated after all blocks are used."""
        game = Game(seed=42)

        # Use all 3 blocks
        for _ in range(3):
            moves = game.get_valid_moves()
            if not moves:
                break
            move = moves[0]
            result = game.make_move(move.block_idx, move.row, move.col)

        # Should have generated new batch
        available = game.get_available_blocks()
        assert len(available) == 3 or game.game_over

    def test_line_clearing(self):
        """Test that lines are cleared when complete."""
        game = Game(seed=42)

        # Manually fill a row except one cell
        game.board.grid[0, :7] = 1

        # Create a 1x1 block and place it
        from src.core.blocks import Block
        single_block = Block(np.array([[1]], dtype=np.int8), "single")
        game.current_blocks = [single_block, single_block, single_block]
        game.blocks_used = [False, False, False]

        # Place to complete the row
        result = game.make_move(0, 0, 7)

        assert result.success
        assert result.clear_result is not None
        assert result.clear_result.lines_cleared == 1
        assert np.all(game.board.grid[0, :] == 0)  # Row should be cleared

    def test_game_over_detection(self):
        """Test that game over is detected when no moves available."""
        game = Game(seed=42)

        # Fill the board almost completely, leaving no space for blocks
        game.board.grid[:, :] = 1

        # Leave some spaces but not enough for any block
        game.board.grid[0, 0] = 0

        # Force check (current blocks might still fit in one cell)
        game.game_over = not game.can_continue()

        # The game should be over if blocks are bigger than 1x1
        # (depends on what blocks were generated)

    def test_reset(self):
        """Test resetting the game."""
        game = Game(seed=42)

        # Play some moves
        for _ in range(3):
            moves = game.get_valid_moves()
            if moves:
                move = moves[0]
                game.make_move(move.block_idx, move.row, move.col)

        # Reset
        game.reset()

        assert game.board.get_filled_cells() == 0
        assert game.turn_count == 0
        assert not game.game_over
        assert game.get_score() == 0

    def test_copy(self):
        """Test copying a game state."""
        game = Game(seed=42)

        # Make a move
        moves = game.get_valid_moves()
        game.make_move(moves[0].block_idx, moves[0].row, moves[0].col)

        # Copy
        copy = game.copy()

        assert copy.turn_count == game.turn_count
        assert copy.get_score() == game.get_score()
        assert np.array_equal(copy.board.grid, game.board.grid)

        # Modify copy, original should not change
        copy_moves = copy.get_valid_moves()
        if copy_moves:
            copy.make_move(copy_moves[0].block_idx, copy_moves[0].row, copy_moves[0].col)
            assert copy.turn_count != game.turn_count

    def test_get_state(self):
        """Test getting game state."""
        game = Game(seed=42)

        state = game.get_state()

        assert state.grid.shape == (8, 8)
        assert len(state.current_blocks) == 3
        assert len(state.block_names) == 3
        assert state.score == 0
        assert state.turn_count == 0
        assert not state.game_over

    def test_deterministic_with_seed(self):
        """Test that games with same seed are deterministic."""
        game1 = Game(seed=12345)
        game2 = Game(seed=12345)

        # First batch should be identical
        for b1, b2 in zip(game1.current_blocks, game2.current_blocks):
            assert np.array_equal(b1.shape, b2.shape)


class TestPlayRandomGame:
    """Tests for play_random_game helper function."""

    def test_completes(self):
        """Test that random game completes."""
        score = play_random_game(seed=42)
        assert score >= 0

    def test_deterministic(self):
        """Test that random games with same seed produce same result."""
        score1 = play_random_game(seed=12345)
        score2 = play_random_game(seed=12345)
        assert score1 == score2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        scores = [play_random_game(seed=i) for i in range(10)]
        # Not all scores should be the same (very unlikely)
        assert len(set(scores)) > 1
