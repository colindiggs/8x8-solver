"""
Main game logic for the 8x8 block puzzle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .board import Board, ClearResult
from .blocks import Block, BlockGenerator
from .scoring import ScoringSystem, ScoringConfig, ScoreResult


@dataclass
class GameState:
    """Serializable representation of the current game state."""
    grid: np.ndarray
    current_blocks: List[np.ndarray]
    block_names: List[str]
    score: int
    combo_streak: int
    turn_count: int
    game_over: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            'grid': self.grid.tolist(),
            'current_blocks': [b.tolist() for b in self.current_blocks],
            'block_names': self.block_names,
            'score': self.score,
            'combo_streak': self.combo_streak,
            'turn_count': self.turn_count,
            'game_over': self.game_over
        }


@dataclass
class MoveResult:
    """Result of executing a move."""
    success: bool
    score_result: Optional[ScoreResult]
    clear_result: Optional[ClearResult]
    blocks_remaining: int
    new_batch_generated: bool
    game_over: bool
    error: Optional[str] = None


@dataclass
class Move:
    """Represents a move in the game."""
    block_idx: int
    row: int
    col: int


class Game:
    """
    Main game controller for the 8x8 block puzzle.

    The game works as follows:
    1. Player starts with 3 blocks to place
    2. Player places blocks one at a time onto the 8x8 board
    3. When a row or column is completely filled, it clears
    4. When all 3 blocks are placed, a new batch of 3 is generated
    5. Game ends when no block can be placed on the board
    """

    BATCH_SIZE = 3

    def __init__(
        self,
        generator: Optional[BlockGenerator] = None,
        scoring_config: Optional[ScoringConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize a new game.

        Args:
            generator: Block generator to use. Creates default if None.
            scoring_config: Scoring configuration. Uses defaults if None.
            seed: Random seed for reproducible games.
        """
        self.board = Board()
        self.generator = generator or BlockGenerator(seed=seed)
        self.scoring = ScoringSystem(scoring_config)

        # Current batch of blocks (indices that haven't been used)
        self.current_blocks: List[Block] = []
        self.blocks_used: List[bool] = []

        # Game state
        self.turn_count = 0
        self.game_over = False

        # Generate initial batch
        self._generate_new_batch()

    def _generate_new_batch(self):
        """Generate a new batch of blocks."""
        self.current_blocks = self.generator.generate_batch(self.BATCH_SIZE)
        self.blocks_used = [False] * self.BATCH_SIZE

    def get_available_blocks(self) -> List[Tuple[int, Block]]:
        """
        Get the blocks that haven't been used yet in this batch.

        Returns:
            List of (index, Block) tuples
        """
        return [
            (i, block)
            for i, (block, used) in enumerate(zip(self.current_blocks, self.blocks_used))
            if not used
        ]

    def get_valid_moves(self) -> List[Move]:
        """
        Get all valid moves that can be made.

        Returns:
            List of Move objects representing all valid placements
        """
        moves = []

        for idx, block in self.get_available_blocks():
            for row, col in self.board.get_valid_placements(block.shape):
                moves.append(Move(block_idx=idx, row=row, col=col))

        return moves

    def can_continue(self) -> bool:
        """
        Check if the game can continue (at least one valid move exists).
        """
        for _, block in self.get_available_blocks():
            if self.board.has_any_valid_placement(block.shape):
                return True
        return False

    def make_move(self, block_idx: int, row: int, col: int) -> MoveResult:
        """
        Execute a move: place a block at the specified position.

        Args:
            block_idx: Index of the block to place (0, 1, or 2)
            row: Row position for placement
            col: Column position for placement

        Returns:
            MoveResult with information about the move outcome
        """
        if self.game_over:
            return MoveResult(
                success=False,
                score_result=None,
                clear_result=None,
                blocks_remaining=sum(1 for used in self.blocks_used if not used),
                new_batch_generated=False,
                game_over=True,
                error="Game is already over"
            )

        # Validate block index
        if block_idx < 0 or block_idx >= self.BATCH_SIZE:
            return MoveResult(
                success=False,
                score_result=None,
                clear_result=None,
                blocks_remaining=sum(1 for used in self.blocks_used if not used),
                new_batch_generated=False,
                game_over=False,
                error=f"Invalid block index: {block_idx}"
            )

        if self.blocks_used[block_idx]:
            return MoveResult(
                success=False,
                score_result=None,
                clear_result=None,
                blocks_remaining=sum(1 for used in self.blocks_used if not used),
                new_batch_generated=False,
                game_over=False,
                error=f"Block {block_idx} has already been used"
            )

        block = self.current_blocks[block_idx]

        # Try to place the block
        if not self.board.place(block.shape, row, col):
            return MoveResult(
                success=False,
                score_result=None,
                clear_result=None,
                blocks_remaining=sum(1 for used in self.blocks_used if not used),
                new_batch_generated=False,
                game_over=False,
                error=f"Cannot place block at ({row}, {col})"
            )

        # Mark block as used
        self.blocks_used[block_idx] = True
        self.turn_count += 1

        # Clear any completed lines
        clear_result = self.board.clear_lines()

        # Calculate score
        score_result = self.scoring.process_move(block.cell_count, clear_result)

        # Check if we need a new batch
        new_batch = False
        if all(self.blocks_used):
            self._generate_new_batch()
            new_batch = True

        # Check for game over
        if not self.can_continue():
            self.game_over = True

        return MoveResult(
            success=True,
            score_result=score_result,
            clear_result=clear_result,
            blocks_remaining=sum(1 for used in self.blocks_used if not used),
            new_batch_generated=new_batch,
            game_over=self.game_over
        )

    def get_state(self) -> GameState:
        """Get the current game state."""
        return GameState(
            grid=self.board.grid.copy(),
            current_blocks=[b.shape.copy() for b in self.current_blocks],
            block_names=[b.name for b in self.current_blocks],
            score=self.scoring.get_total_score(),
            combo_streak=self.scoring.get_combo_streak(),
            turn_count=self.turn_count,
            game_over=self.game_over
        )

    def get_score(self) -> int:
        """Get the current score."""
        return self.scoring.get_total_score()

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game_over

    def reset(self):
        """Reset the game to initial state."""
        self.board = Board()
        self.scoring.reset()
        self.turn_count = 0
        self.game_over = False
        self._generate_new_batch()

    def copy(self) -> Game:
        """
        Create a copy of the game for lookahead/simulation.
        Note: The generator is shared, which may affect reproducibility.
        """
        game = Game.__new__(Game)
        game.board = self.board.copy()
        game.generator = self.generator
        game.scoring = ScoringSystem(self.scoring.config)
        game.scoring.total_score = self.scoring.total_score
        game.scoring.combo_level = self.scoring.combo_level
        game.scoring.moves_since_clear = self.scoring.moves_since_clear
        game.current_blocks = [
            Block(b.shape.copy(), b.name) for b in self.current_blocks
        ]
        game.blocks_used = self.blocks_used.copy()
        game.turn_count = self.turn_count
        game.game_over = self.game_over
        return game

    def __str__(self) -> str:
        """String representation of the game state."""
        lines = [
            f"Turn: {self.turn_count}  Score: {self.get_score()}  Combo: {self.scoring.get_combo_streak()}",
            "",
            str(self.board),
            "",
            "Available blocks:"
        ]

        for idx, block in self.get_available_blocks():
            lines.append(f"  [{idx}] {block.name} ({block.height}x{block.width})")

        if self.game_over:
            lines.append("\n*** GAME OVER ***")

        return "\n".join(lines)


def play_random_game(seed: Optional[int] = None, verbose: bool = False) -> int:
    """
    Play a complete game with random moves.
    Useful for testing and establishing baseline performance.

    Args:
        seed: Random seed for reproducibility
        verbose: If True, print game state after each move

    Returns:
        Final score
    """
    import random

    if seed is not None:
        random.seed(seed)

    game = Game(seed=seed)

    while not game.is_game_over():
        moves = game.get_valid_moves()
        if not moves:
            break

        move = random.choice(moves)
        result = game.make_move(move.block_idx, move.row, move.col)

        if verbose:
            print(f"\nPlaced block {move.block_idx} at ({move.row}, {move.col})")
            print(f"Points: {result.score_result.total_points}")
            print(game)

    if verbose:
        print(f"\nFinal Score: {game.get_score()}")

    return game.get_score()
