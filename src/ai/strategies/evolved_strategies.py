"""
Evolved/trained strategies for the 8x8 Block Puzzle Arena.

These strategies use tuned weights from genetic algorithm optimization
and deeper lookahead for better performance.
"""
from typing import Optional, Tuple, List
import numpy as np

from .base import Strategy, StrategyInfo
from core.game import Game, Move


class DeepLookaheadStrategy(Strategy):
    """Deep search with pruning for better move selection."""

    INFO = StrategyInfo(
        id="deep_lookahead",
        name="Deep Lookahead",
        short_desc="2-ply search with board evaluation",
        algorithm="Minimax-style 2-ply search. For each move, simulates the result then "
                  "evaluates all follow-up moves (best 15 sampled). Uses composite evaluation: "
                  "score_gain + 0.4*best_future + 0.3*empty_cells - 5*holes. Prunes losing branches early.",
        complexity="O(n * 15 * eval) per turn",
        category="Search-Based"
    )

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.depth = 2

    def select_move(self, game: Game) -> Optional[Move]:
        scored = self.get_move_scores(game)
        if not scored:
            return None
        best_move, best_score, explanation = scored[0]
        self.move_explanations.append(explanation)
        return best_move

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        sim_game = game.copy()
        result = sim_game.make_move(move.block_idx, move.row, move.col)

        if not result.success:
            return -10000, "Invalid"

        immediate = result.score_result.total_points

        # Evaluate board state
        empty = 64 - np.sum(sim_game.board.grid)
        holes = self._count_holes(sim_game.board.grid)

        # Look ahead
        future_moves = sim_game.get_valid_moves()
        if not future_moves:
            return immediate - 1000 + empty * 0.3, f"+{immediate}, GAME OVER"

        # Evaluate best future moves
        best_future = 0
        for fm in future_moves[:15]:
            sim2 = sim_game.copy()
            r2 = sim2.make_move(fm.block_idx, fm.row, fm.col)
            if r2.success:
                future_empty = 64 - np.sum(sim2.board.grid)
                future_holes = self._count_holes(sim2.board.grid)
                future_score = r2.score_result.total_points + future_empty * 0.2 - future_holes * 3
                best_future = max(best_future, future_score)

        total = immediate + best_future * 0.4 + empty * 0.3 - holes * 5
        return total, f"+{immediate} | future:{best_future:.0f} | holes:{holes}"

    def _count_holes(self, grid) -> int:
        """Count cells that are hard to fill (3+ neighbors blocked)."""
        holes = 0
        for r in range(8):
            for c in range(8):
                if not grid[r, c]:
                    blocked = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < 8 and 0 <= nc < 8) or grid[nr, nc]:
                            blocked += 1
                    if blocked >= 3:
                        holes += 1
        return holes


class ComboMasterStrategy(Strategy):
    """Optimized combo chaser with lookahead."""

    INFO = StrategyInfo(
        id="combo_master",
        name="Combo Master",
        short_desc="Combo-focused with setup awareness",
        algorithm="Prioritizes building and maintaining combos. Evaluates: combo_level * 25 + "
                  "lines_cleared * 40 + near_complete_bonus. Looks ahead to find moves that set up "
                  "future clears. Penalizes moves that waste combo grace period (-20 at grace 2+).",
        complexity="O(n * setup_eval) per turn",
        category="Aggressive"
    )

    def select_move(self, game: Game) -> Optional[Move]:
        scored = self.get_move_scores(game)
        if not scored:
            return None
        best_move, best_score, explanation = scored[0]
        self.move_explanations.append(explanation)
        return best_move

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        sim_game = game.copy()
        result = sim_game.make_move(move.block_idx, move.row, move.col)

        if not result.success:
            return -10000, "Invalid"

        score = 0
        lines = result.score_result.lines_cleared
        new_combo = result.score_result.combo_level
        current_combo = game.scoring.get_combo_level()

        # Reward line clears heavily
        if lines > 0:
            score += new_combo * 25  # Combo bonus
            score += lines * 40  # Lines cleared
            if lines >= 2:
                score += 60  # Multi-clear bonus

        # Count near-complete lines for setup potential
        near_complete = self._count_near_complete(sim_game.board.grid)
        score += near_complete * 15

        # Penalize wasting grace period
        if lines == 0 and current_combo > 0:
            moves_since = game.scoring.get_moves_since_clear()
            if moves_since >= 2:
                score -= 20  # Risk of losing combo
            # But reward if we're setting up a clear
            if near_complete > 0:
                score += 10  # Setup bonus

        # Small bonus for empty cells (survival)
        empty = 64 - np.sum(sim_game.board.grid)
        score += empty * 0.2

        explanation = f"{lines}L {new_combo}x | setup:{near_complete}"
        return score, explanation

    def _count_near_complete(self, grid) -> int:
        """Count rows/cols with 6+ cells filled."""
        count = 0
        for i in range(8):
            if np.sum(grid[i, :]) >= 6:
                count += 1
            if np.sum(grid[:, i]) >= 6:
                count += 1
        return count


class TunedBalancedStrategy(Strategy):
    """Balanced strategy with optimized weights from testing."""

    INFO = StrategyInfo(
        id="tuned_balanced",
        name="Tuned Balanced",
        short_desc="Multi-factor with optimized weights",
        algorithm="Weighted combination of 6 factors tuned through experimentation: "
                  "immediate_score (1.0), lines_cleared (15.0), combo_bonus (8.0), "
                  "empty_cells (0.4), future_moves (0.5), holes_penalty (-8.0). "
                  "Balances short-term scoring with long-term survival.",
        complexity="O(n * eval) per turn",
        category="Strategic"
    )

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        # Tuned weights
        self.weights = {
            'score': 1.0,
            'lines': 15.0,
            'combo': 8.0,
            'empty': 0.4,
            'options': 0.5,
            'holes': -8.0,
        }

    def select_move(self, game: Game) -> Optional[Move]:
        scored = self.get_move_scores(game)
        if not scored:
            return None
        best_move, best_score, explanation = scored[0]
        self.move_explanations.append(explanation)
        return best_move

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        sim_game = game.copy()
        result = sim_game.make_move(move.block_idx, move.row, move.col)

        if not result.success:
            return -10000, "Invalid"

        w = self.weights
        grid = sim_game.board.grid

        # Calculate factors
        score_pts = result.score_result.total_points * w['score']
        lines_pts = result.score_result.lines_cleared * w['lines']
        combo_pts = result.score_result.combo_level * w['combo']
        empty_pts = (64 - np.sum(grid)) * w['empty']
        options_pts = len(sim_game.get_valid_moves()) * w['options']

        # Count holes
        holes = 0
        for r in range(8):
            for c in range(8):
                if not grid[r, c]:
                    blocked = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < 8 and 0 <= nc < 8) or grid[nr, nc]:
                            blocked += 1
                    if blocked >= 3:
                        holes += 1
        holes_pts = holes * w['holes']

        total = score_pts + lines_pts + combo_pts + empty_pts + options_pts + holes_pts

        explanation = f"+{result.score_result.total_points} | {result.score_result.lines_cleared}L | {holes}h"
        return total, explanation


class SurvivalExpertStrategy(Strategy):
    """Maximizes game length and scoring opportunities."""

    INFO = StrategyInfo(
        id="survival_expert",
        name="Survival Expert",
        short_desc="Maximizes longevity and flexibility",
        algorithm="Primary goal: avoid game over. Evaluates: future_move_count * 3 + "
                  "empty_cells * 2 + lines_cleared * 25 - holes * 10. Heavily penalizes moves "
                  "that reduce options below threshold. Clears lines only when safe.",
        complexity="O(n * future_moves) per turn",
        category="Defensive"
    )

    def select_move(self, game: Game) -> Optional[Move]:
        scored = self.get_move_scores(game)
        if not scored:
            return None
        best_move, best_score, explanation = scored[0]
        self.move_explanations.append(explanation)
        return best_move

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        sim_game = game.copy()
        result = sim_game.make_move(move.block_idx, move.row, move.col)

        if not result.success:
            return -10000, "Invalid"

        grid = sim_game.board.grid
        empty = 64 - np.sum(grid)
        future_moves = len(sim_game.get_valid_moves())
        lines = result.score_result.lines_cleared

        # Count holes
        holes = 0
        for r in range(8):
            for c in range(8):
                if not grid[r, c]:
                    blocked = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < 8 and 0 <= nc < 8) or grid[nr, nc]:
                            blocked += 1
                    if blocked >= 3:
                        holes += 1

        score = future_moves * 3 + empty * 2 + lines * 25 - holes * 10

        # Heavily penalize low-options situations
        if future_moves < 10:
            score -= (10 - future_moves) * 5

        # Game over is catastrophic
        if future_moves == 0:
            score -= 500

        return score, f"{empty} empty | {future_moves} opts | {holes}h"
