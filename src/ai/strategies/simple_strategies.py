"""
Simple, explainable strategies for the 8x8 Block Puzzle Arena.

These strategies use single heuristics that are easy to understand
and explain. They serve as baselines and demonstrate basic approaches.
"""
from typing import Optional, Tuple, List
import numpy as np

from .base import Strategy, StrategyInfo
from core.game import Game, Move


class RandomStrategy(Strategy):
    """Baseline: Makes completely random valid moves."""

    INFO = StrategyInfo(
        id="random",
        name="Random",
        short_desc="Uniformly random move selection",
        algorithm="Enumerates all valid placements and selects one uniformly at random. "
                  "Serves as the baseline for comparing other strategies. Expected score "
                  "reflects pure chance without any optimization.",
        complexity="O(n) where n = valid moves",
        category="Baseline"
    )

    def select_move(self, game: Game) -> Optional[Move]:
        moves = game.get_valid_moves()
        if not moves:
            return None
        move = moves[self.rng.choice(len(moves))]
        self.move_explanations.append(f"Random: ({move.row}, {move.col})")
        return move

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        return self.rng.random(), "Random score"


class GreedyScoreStrategy(Strategy):
    """Maximizes immediate score gain each turn."""

    INFO = StrategyInfo(
        id="greedy",
        name="Greedy",
        short_desc="Maximizes immediate points per move",
        algorithm="Simulates each valid move and calculates the immediate score gain "
                  "(placement points + line clear bonuses). Selects the move with highest "
                  "immediate return. Does not consider future board state or combo potential.",
        complexity="O(n) simulations per turn",
        category="Greedy"
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
            return -1000, "Invalid move"

        score = result.score_result.total_points
        lines = result.score_result.lines_cleared

        if lines > 0:
            return score, f"+{score} pts ({lines} lines)"
        return score, f"+{score} pts"


class EdgeHuggerStrategy(Strategy):
    """Prefers placing blocks along edges and corners."""

    INFO = StrategyInfo(
        id="edge_hugger",
        name="Edge Hugger",
        short_desc="Prioritizes edge and corner placement",
        algorithm="Scores each move based on how many cells touch the board perimeter. "
                  "Corner cells score 3 points, edge cells score 2 points. The hypothesis "
                  "is that edge placement leaves the center open for flexible future moves.",
        complexity="O(n * block_size) per turn",
        category="Positional"
    )

    def select_move(self, game: Game) -> Optional[Move]:
        scored = self.get_move_scores(game)
        if not scored:
            return None
        best_move, best_score, explanation = scored[0]
        self.move_explanations.append(explanation)
        return best_move

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        block = game.current_blocks[move.block_idx]
        edge_cells = 0
        corner_cells = 0

        for dr in range(block.height):
            for dc in range(block.width):
                if block.shape[dr, dc]:
                    r, c = move.row + dr, move.col + dc
                    on_edge = (r == 0 or r == 7 or c == 0 or c == 7)
                    on_corner = (r in [0, 7]) and (c in [0, 7])

                    if on_corner:
                        corner_cells += 1
                    elif on_edge:
                        edge_cells += 1

        score = corner_cells * 3 + edge_cells * 2

        sim_game = game.copy()
        result = sim_game.make_move(move.block_idx, move.row, move.col)
        if result.success and result.score_result.lines_cleared > 0:
            score += result.score_result.lines_cleared * 5

        parts = []
        if corner_cells:
            parts.append(f"{corner_cells} corner")
        if edge_cells:
            parts.append(f"{edge_cells} edge")
        if result.success and result.score_result.lines_cleared:
            parts.append(f"+{result.score_result.lines_cleared} lines")

        return score, " | ".join(parts) if parts else "Interior"


class CenterControlStrategy(Strategy):
    """Focuses on controlling the center of the board."""

    INFO = StrategyInfo(
        id="center",
        name="Center Control",
        short_desc="Minimizes average distance from board center",
        algorithm="Calculates Manhattan distance from each placed cell to board center (3.5, 3.5). "
                  "Prefers moves that minimize average distance. Theory: central control provides "
                  "more adjacent placement options than corner positions.",
        complexity="O(n * block_size) per turn",
        category="Positional"
    )

    def select_move(self, game: Game) -> Optional[Move]:
        scored = self.get_move_scores(game)
        if not scored:
            return None
        best_move, best_score, explanation = scored[0]
        self.move_explanations.append(explanation)
        return best_move

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        block = game.current_blocks[move.block_idx]
        center = 3.5

        total_dist = 0
        cell_count = 0

        for dr in range(block.height):
            for dc in range(block.width):
                if block.shape[dr, dc]:
                    r, c = move.row + dr, move.col + dc
                    dist = abs(r - center) + abs(c - center)
                    total_dist += dist
                    cell_count += 1

        avg_dist = total_dist / cell_count if cell_count else 7
        centrality = 7 - avg_dist

        sim_game = game.copy()
        result = sim_game.make_move(move.block_idx, move.row, move.col)
        if result.success and result.score_result.lines_cleared > 0:
            centrality += result.score_result.lines_cleared * 3

        return centrality, f"Dist: {avg_dist:.1f}"


class LineHunterStrategy(Strategy):
    """Aggressively pursues line clears."""

    INFO = StrategyInfo(
        id="line_hunter",
        name="Line Hunter",
        short_desc="Maximizes line clears per move",
        algorithm="Simulates each move and counts resulting line clears. Moves that clear "
                  "lines score 100 points per line; non-clearing moves score 0. Ignores "
                  "placement points and combo state entirely.",
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

        lines = result.score_result.lines_cleared
        if lines > 0:
            return lines * 100, f"Clears {lines} line(s)"
        return 0, "No clear"


class ComboChaserStrategy(Strategy):
    """Maximizes and maintains combo streaks."""

    INFO = StrategyInfo(
        id="combo_chaser",
        name="Combo Chaser",
        short_desc="Optimizes for combo streak maintenance",
        algorithm="Tracks combo multiplier and grace period. Heavily weights moves that "
                  "continue the combo streak (20 pts per combo level). Penalizes moves that "
                  "would break the streak when grace period is nearly expired. Multi-clears "
                  "receive 50pt bonus.",
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

        current_combo = game.scoring.get_combo_level()
        lines = result.score_result.lines_cleared
        new_combo = result.score_result.combo_level

        score = 0

        if lines > 0:
            score += new_combo * 20
            if lines >= 2:
                score += 50
            explanation = f"{current_combo}x -> {new_combo}x"
        else:
            moves_since = game.scoring.get_moves_since_clear()
            if current_combo > 0 and moves_since >= 2:
                score -= 30
                explanation = f"Risk! Grace {moves_since}/3"
            else:
                explanation = "Combo safe"

        return score, explanation


class SurvivalistStrategy(Strategy):
    """Maximizes open space and flexibility."""

    INFO = StrategyInfo(
        id="survivalist",
        name="Survivalist",
        short_desc="Maximizes board flexibility and survival",
        algorithm="After simulating each move, counts empty cells and available future moves. "
                  "Score = (empty_cells * 2) + future_move_count. Prioritizes keeping options "
                  "open over immediate scoring. Optimizes for game length.",
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

        empty_cells = 64 - np.sum(sim_game.board.grid)
        future_moves = len(sim_game.get_valid_moves())
        score = empty_cells * 2 + future_moves

        return score, f"{empty_cells} empty | {future_moves} options"


class CompactorStrategy(Strategy):
    """Keeps pieces clustered together."""

    INFO = StrategyInfo(
        id="compactor",
        name="Compactor",
        short_desc="Maximizes cell adjacency/clustering",
        algorithm="Counts orthogonal adjacencies between filled cells after each move. "
                  "Higher adjacency means tighter clustering, which theoretically leaves "
                  "cleaner rows/columns for future clears. Each adjacency pair scores 1 point.",
        complexity="O(n + 64) per turn",
        category="Positional"
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
        adjacency = 0

        for r in range(8):
            for c in range(8):
                if grid[r, c]:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 8 and 0 <= nc < 8 and grid[nr, nc]:
                            adjacency += 1

        if result.score_result.lines_cleared > 0:
            adjacency += result.score_result.lines_cleared * 20

        return adjacency, f"Adj: {adjacency}"


class NearCompleteStrategy(Strategy):
    """Prioritizes completing nearly-full rows and columns."""

    INFO = StrategyInfo(
        id="near_complete",
        name="Near Complete",
        short_desc="Targets rows/columns with 6+ filled cells",
        algorithm="After simulating each move, scans all rows and columns for those with "
                  "6+ filled cells (near completion). Scores (fill_count - 5) * 2 for each "
                  "near-complete line. Prioritizes setting up efficient future clears.",
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

        if result.score_result.lines_cleared > 0:
            return 100 + result.score_result.lines_cleared * 50, f"Cleared {result.score_result.lines_cleared}"

        grid = sim_game.board.grid
        near_complete = 0
        details = []

        for i in range(8):
            row_sum = np.sum(grid[i, :])
            col_sum = np.sum(grid[:, i])

            if row_sum >= 6:
                near_complete += (row_sum - 5) * 2
                details.append(f"R{i}:{row_sum}")
            if col_sum >= 6:
                near_complete += (col_sum - 5) * 2
                details.append(f"C{i}:{col_sum}")

        explanation = ", ".join(details[:3]) if details else "Building..."
        return near_complete, explanation
