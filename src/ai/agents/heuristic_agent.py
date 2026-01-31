"""
Heuristic-based agent for the 8x8 block puzzle.

This agent uses hand-crafted evaluation functions to score moves
and select the best one. It serves as a strong baseline and can
bootstrap learning for the DQN agent.
"""
from typing import Optional, List, Tuple
import numpy as np

from .base import BaseAgent
from core.game import Game, Move
from core.board import Board


class HeuristicAgent(BaseAgent):
    """
    Agent that uses heuristics to evaluate and select moves.

    Evaluation criteria:
    1. Lines cleared by this move (immediate reward)
    2. Board compactness (keep pieces together)
    3. Edge/corner preference (leave center open)
    4. Near-complete lines (positions close to clearing)
    5. Future flexibility (maximize valid placements for remaining blocks)
    """

    def __init__(
        self,
        clear_weight: float = 100.0,
        compactness_weight: float = 1.0,
        edge_weight: float = 0.5,
        near_complete_weight: float = 5.0,
        flexibility_weight: float = 2.0,
    ):
        super().__init__(name="HeuristicAgent")

        # Weights for different evaluation components
        self.clear_weight = clear_weight
        self.compactness_weight = compactness_weight
        self.edge_weight = edge_weight
        self.near_complete_weight = near_complete_weight
        self.flexibility_weight = flexibility_weight

    def select_move(self, game: Game, explore: bool = True) -> Optional[Move]:
        """Select the move with the highest heuristic score."""
        scored_moves = self.get_move_scores(game)

        if not scored_moves:
            return None

        # Return the best move (highest score)
        return scored_moves[0][0]

    def get_move_scores(self, game: Game) -> List[Tuple[Move, float]]:
        """
        Score all valid moves and return sorted by score (descending).
        """
        moves = game.get_valid_moves()

        if not moves:
            return []

        scored = []
        for move in moves:
            score = self._evaluate_move(game, move)
            scored.append((move, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _evaluate_move(self, game: Game, move: Move) -> float:
        """
        Evaluate a single move by simulating it and scoring the result.
        """
        # Make a copy of the game to simulate
        sim_game = game.copy()

        # Get the block being placed
        block = sim_game.current_blocks[move.block_idx]

        # Simulate the placement
        result = sim_game.make_move(move.block_idx, move.row, move.col)

        if not result.success:
            return float('-inf')

        score = 0.0

        # 1. Reward for clearing lines
        if result.clear_result:
            lines = result.clear_result.lines_cleared
            score += self.clear_weight * lines
            # Extra bonus for clearing multiple lines
            if lines > 1:
                score += self.clear_weight * (lines - 1) * 0.5

        # 2. Board compactness (penalize scattered pieces)
        compactness = self._calculate_compactness(sim_game.board)
        score += self.compactness_weight * compactness

        # 3. Edge preference (reward placing near edges, keep center open)
        edge_score = self._calculate_edge_score(move.row, move.col, block.shape)
        score += self.edge_weight * edge_score

        # 4. Near-complete lines (reward positions close to clearing)
        near_complete = self._count_near_complete_lines(sim_game.board)
        score += self.near_complete_weight * near_complete

        # 5. Future flexibility (reward moves that leave options open)
        flexibility = self._calculate_flexibility(sim_game)
        score += self.flexibility_weight * flexibility

        return score

    def _calculate_compactness(self, board: Board) -> float:
        """
        Calculate how compact/connected the filled cells are.
        Higher score = more compact (better).
        """
        filled = board.grid == 1
        if not np.any(filled):
            return 0.0

        # Count adjacent pairs of filled cells
        adjacent_count = 0

        # Horizontal adjacency
        adjacent_count += np.sum(filled[:, :-1] & filled[:, 1:])

        # Vertical adjacency
        adjacent_count += np.sum(filled[:-1, :] & filled[1:, :])

        # Normalize by number of filled cells
        total_filled = np.sum(filled)
        if total_filled == 0:
            return 0.0

        return adjacent_count / total_filled

    def _calculate_edge_score(
        self, row: int, col: int, block_shape: np.ndarray
    ) -> float:
        """
        Score placement based on distance from center.
        Prefer edges/corners to keep center flexible.
        """
        block_h, block_w = block_shape.shape
        center_row = row + block_h / 2
        center_col = col + block_w / 2

        # Distance from board center (3.5, 3.5)
        dist = abs(center_row - 3.5) + abs(center_col - 3.5)

        return dist

    def _count_near_complete_lines(self, board: Board) -> float:
        """
        Count lines that are close to being complete (6+ cells filled).
        """
        score = 0.0

        # Check rows
        for row in range(Board.SIZE):
            filled = np.sum(board.grid[row, :])
            if filled >= 6:
                score += (filled - 5)  # 6->1, 7->2, 8->3

        # Check columns
        for col in range(Board.SIZE):
            filled = np.sum(board.grid[:, col])
            if filled >= 6:
                score += (filled - 5)

        return score

    def _calculate_flexibility(self, game: Game) -> float:
        """
        Calculate how many valid placements remain for unused blocks.
        """
        total_placements = 0

        for idx, block in game.get_available_blocks():
            placements = len(game.board.get_valid_placements(block.shape))
            total_placements += placements

        return total_placements

    def reset(self):
        """No state to reset."""
        pass
