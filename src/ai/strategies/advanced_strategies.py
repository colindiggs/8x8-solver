"""
Advanced strategies for the 8x8 Block Puzzle Arena.

These strategies use more sophisticated algorithms including
lookahead, Monte Carlo sampling, and weighted multi-factor evaluation.
"""
from typing import Optional, Tuple, List, Dict
import numpy as np

from .base import Strategy, StrategyInfo
from core.game import Game, Move


class BalancedStrategy(Strategy):
    """Weighted combination of multiple factors."""

    INFO = StrategyInfo(
        id="balanced",
        name="Balanced",
        short_desc="Multi-factor weighted evaluation",
        algorithm="Combines 5 weighted factors: immediate_score (1.0), lines_cleared (10.0), "
                  "combo_bonus (5.0), empty_cells (0.5), future_moves (0.3). Final score is "
                  "the weighted sum. Attempts to balance short-term gains with long-term position.",
        complexity="O(n * future_moves) per turn",
        category="Strategic"
    )

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.weights = {
            'immediate_score': 1.0,
            'lines_cleared': 10.0,
            'combo_bonus': 5.0,
            'empty_cells': 0.5,
            'future_moves': 0.3,
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
            return -1000, "Invalid"

        w = self.weights
        factors = {}

        factors['score'] = result.score_result.total_points * w['immediate_score']
        factors['lines'] = result.score_result.lines_cleared * w['lines_cleared']
        factors['combo'] = result.score_result.combo_level * w['combo_bonus']
        factors['space'] = (64 - np.sum(sim_game.board.grid)) * w['empty_cells']
        factors['flex'] = len(sim_game.get_valid_moves()) * w['future_moves']

        total = sum(factors.values())
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:2]
        explanation = " + ".join(f"{k}:{v:.0f}" for k, v in top_factors)

        return total, explanation


class LookaheadStrategy(Strategy):
    """Looks ahead one move to evaluate positions."""

    INFO = StrategyInfo(
        id="lookahead",
        name="Lookahead",
        short_desc="1-ply search with position evaluation",
        algorithm="For each candidate move, simulates the move then evaluates all possible "
                  "follow-up moves (up to 10 sampled). Returns immediate_score + 0.5 * best_future_score. "
                  "Penalizes moves that lead to game over (-500). Similar to minimax depth-1.",
        complexity="O(n * 10) simulations per turn",
        category="Search-Based"
    )

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.depth = 1

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
            return -1000, "Invalid"

        immediate = result.score_result.total_points

        future_moves = sim_game.get_valid_moves()
        if not future_moves:
            return immediate - 500, f"+{immediate}, GAME OVER"

        best_future = 0
        for fm in future_moves[:10]:
            sim2 = sim_game.copy()
            r2 = sim2.make_move(fm.block_idx, fm.row, fm.col)
            if r2.success:
                best_future = max(best_future, r2.score_result.total_points)

        total = immediate + best_future * 0.5
        return total, f"+{immediate} | future: +{best_future}"


class MonteCarloStrategy(Strategy):
    """Uses random sampling to evaluate moves."""

    INFO = StrategyInfo(
        id="monte_carlo",
        name="Monte Carlo",
        short_desc="Random playout sampling (MCTS-lite)",
        algorithm="For each candidate move, runs 10 random playouts of 5 moves each. "
                  "Returns the average score gained across all playouts. Standard deviation "
                  "indicates move reliability. Inspired by Monte Carlo Tree Search.",
        complexity="O(n * 10 * 5) simulations per turn",
        category="Search-Based"
    )

    def __init__(self, seed: Optional[int] = None, simulations: int = 10):
        super().__init__(seed)
        self.simulations = simulations

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
            return -1000, "Invalid"

        scores = []
        for _ in range(self.simulations):
            playout_game = sim_game.copy()
            playout_score = self._random_playout(playout_game, max_moves=5)
            scores.append(playout_score)

        avg_score = np.mean(scores)
        std_score = np.std(scores)

        return avg_score, f"Avg: {avg_score:.0f} +/- {std_score:.0f}"

    def _random_playout(self, game: Game, max_moves: int) -> float:
        """Play randomly for a few moves and return total score gained."""
        start_score = game.get_score()

        for _ in range(max_moves):
            moves = game.get_valid_moves()
            if not moves:
                break
            move = moves[self.rng.choice(len(moves))]
            game.make_move(move.block_idx, move.row, move.col)

        return game.get_score() - start_score


class HolesAvoiderStrategy(Strategy):
    """Avoids creating holes (empty cells surrounded by filled cells)."""

    INFO = StrategyInfo(
        id="holes_avoider",
        name="Holes Avoider",
        short_desc="Minimizes trapped empty cells",
        algorithm="Counts 'holes' - empty cells with 3+ orthogonal neighbors filled (or edge). "
                  "Score = -10 * hole_count. These cells are difficult to fill with standard "
                  "block shapes. Line clears add +30 bonus per line.",
        complexity="O(n + 64) per turn",
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
            return -1000, "Invalid"

        grid = sim_game.board.grid
        holes = 0

        for r in range(8):
            for c in range(8):
                if not grid[r, c]:
                    neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 8 and 0 <= nc < 8:
                            if grid[nr, nc]:
                                neighbors += 1
                        else:
                            neighbors += 1
                    if neighbors >= 3:
                        holes += 1

        score = -holes * 10

        if result.score_result.lines_cleared > 0:
            score += result.score_result.lines_cleared * 30

        lines_info = f" +{result.score_result.lines_cleared}L" if result.score_result.lines_cleared else ""
        return score, f"{holes} holes{lines_info}"


class RowColumnBalanceStrategy(Strategy):
    """Keeps row and column fill levels balanced."""

    INFO = StrategyInfo(
        id="balance_rc",
        name="Row-Col Balance",
        short_desc="Minimizes fill-level variance",
        algorithm="Calculates variance of row fill counts and column fill counts. "
                  "Score = -(row_variance + col_variance). Lower variance means more uniform "
                  "distribution, theoretically enabling more simultaneous clears.",
        complexity="O(n + 16) per turn",
        category="Strategic"
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
            return -1000, "Invalid"

        grid = sim_game.board.grid

        row_fills = [np.sum(grid[i, :]) for i in range(8)]
        col_fills = [np.sum(grid[:, i]) for i in range(8)]

        row_var = np.var(row_fills)
        col_var = np.var(col_fills)

        score = -(row_var + col_var)

        if result.score_result.lines_cleared > 0:
            score += result.score_result.lines_cleared * 50

        return score, f"Var: R={row_var:.1f} C={col_var:.1f}"


class AggressiveStrategy(Strategy):
    """Takes risks for high scores."""

    INFO = StrategyInfo(
        id="aggressive",
        name="Aggressive",
        short_desc="Maximizes score with risk tolerance",
        algorithm="Heavily weights scoring factors: score * 2 + combo * 15 + lines * 25. "
                  "Multi-clears (2+ lines) receive additional +50 bonus. Ignores board safety "
                  "metrics entirely. High variance, high potential reward.",
        complexity="O(n) simulations per turn",
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
            return -1000, "Invalid"

        score = result.score_result.total_points * 2
        score += result.score_result.combo_level * 15
        score += result.score_result.lines_cleared * 25

        if result.score_result.lines_cleared >= 2:
            score += 50

        return score, f"+{result.score_result.total_points} | {result.score_result.combo_level}x"


class DefensiveStrategy(Strategy):
    """Plays it safe, prioritizes not losing."""

    INFO = StrategyInfo(
        id="defensive",
        name="Defensive",
        short_desc="Prioritizes survival over scoring",
        algorithm="Score = empty_cells * 3 + future_move_count * 2. Optimizes for game length "
                  "and flexibility rather than immediate points. Line clears receive modest +20 "
                  "bonus. Tends to produce longer games with lower scores.",
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
            return -1000, "Invalid"

        empty = 64 - np.sum(sim_game.board.grid)
        options = len(sim_game.get_valid_moves())

        score = empty * 3 + options * 2

        if result.score_result.lines_cleared > 0:
            score += result.score_result.lines_cleared * 20

        return score, f"{empty} empty | {options} opts"
