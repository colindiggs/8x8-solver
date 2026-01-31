"""
Scoring system for the 8x8 block puzzle game.

CONFIRMED through empirical testing on 2026-01-31.
"""
from dataclasses import dataclass
from typing import Optional

from .board import ClearResult


@dataclass
class ScoringConfig:
    """Configuration for scoring calculations."""
    # Points per cell when placing a block
    points_per_cell: int = 1

    # Combo points: 10 * combo_level per line cleared
    combo_points_multiplier: int = 10

    # Bonus for clearing 2+ lines simultaneously (only when combo > 0)
    multi_clear_bonus: int = 30

    # Grace period: combo persists for this many moves without clearing
    combo_grace_period: int = 3


DEFAULT_SCORING = ScoringConfig()


@dataclass
class ScoreResult:
    """Result of a scoring calculation."""
    placement_points: int  # Points from placing the block
    clear_points: int      # Points from clearing lines (combo-based)
    multi_clear_bonus: int # Bonus for clearing 2+ lines at once
    total_points: int      # Total points earned this move

    lines_cleared: int     # Number of lines cleared
    combo_level: int       # Combo level after this move
    moves_since_clear: int # Moves since last clear (for grace period)


class ScoringSystem:
    """
    Handles all scoring calculations for the game.

    Scoring formula (discovered empirically):
    - +1 point per cell placed
    - Each line cleared increments combo, then awards 10 * combo_level points
    - +30 bonus when clearing 2+ lines at once (if combo > 0)
    - Combo persists for up to 3 moves without clearing
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or DEFAULT_SCORING
        self.combo_level = 0
        self.moves_since_clear = 0
        self.total_score = 0

    def process_move(self, cells_placed: int, clear_result: ClearResult) -> ScoreResult:
        """
        Process a complete move and calculate all points.

        Args:
            cells_placed: Number of cells in the block that was placed
            clear_result: Result from clearing lines

        Returns:
            ScoreResult with breakdown of points earned
        """
        # Points from placing cells
        placement_points = cells_placed * self.config.points_per_cell

        # Process line clears
        clear_points = 0
        multi_bonus = 0
        lines_cleared = clear_result.lines_cleared

        if lines_cleared > 0:
            # Scoring formula (empirically discovered):
            # Single clear: 1x = 10, Nx (N>=2) = 10 + 10*N
            # Multi-clear: 10*N per line (no base), +30 bonus if had existing combo
            is_multi_clear = lines_cleared >= 2

            for _ in range(lines_cleared):
                self.combo_level += 1
                if is_multi_clear:
                    # Multi-clear: just 10 * combo_level per line
                    clear_points += self.combo_level * self.config.combo_points_multiplier
                elif self.combo_level == 1:
                    # Single clear at 1x: just base 10
                    clear_points += 10
                else:
                    # Single clear at 2x+: base 10 + combo bonus (10 * N)
                    clear_points += 10 + (self.combo_level * self.config.combo_points_multiplier)

            # Multi-clear bonus (+30 when clearing 2+ lines, if had existing combo)
            if is_multi_clear and self.combo_level > lines_cleared:
                multi_bonus = self.config.multi_clear_bonus

            self.moves_since_clear = 0
        else:
            # No clear - check grace period
            self.moves_since_clear += 1
            if self.moves_since_clear > self.config.combo_grace_period:
                self.combo_level = 0

        # Calculate total
        total = placement_points + clear_points + multi_bonus
        self.total_score += total

        return ScoreResult(
            placement_points=placement_points,
            clear_points=clear_points,
            multi_clear_bonus=multi_bonus,
            total_points=total,
            lines_cleared=lines_cleared,
            combo_level=self.combo_level,
            moves_since_clear=self.moves_since_clear
        )

    def reset(self):
        """Reset the scoring system for a new game."""
        self.combo_level = 0
        self.moves_since_clear = 0
        self.total_score = 0

    def get_total_score(self) -> int:
        """Get the current total score."""
        return self.total_score

    def get_combo_level(self) -> int:
        """Get the current combo level."""
        return self.combo_level

    # Alias for compatibility
    def get_combo_streak(self) -> int:
        """Alias for get_combo_level (for API compatibility)."""
        return self.combo_level

    def get_moves_since_clear(self) -> int:
        """Get moves since last clear."""
        return self.moves_since_clear


def calculate_potential_score(
    cells_placed: int,
    lines_to_clear: int,
    current_combo: int,
    config: Optional[ScoringConfig] = None
) -> int:
    """
    Calculate potential score for a hypothetical move.
    Useful for AI evaluation.

    Args:
        cells_placed: Number of cells in the block
        lines_to_clear: Number of lines that would be cleared
        current_combo: Current combo level before the move

    Returns:
        Estimated points for this move
    """
    cfg = config or DEFAULT_SCORING

    # Placement points
    points = cells_placed * cfg.points_per_cell

    # Clear points: Single = 1x:10, Nx:10+10*N; Multi = 10*N per line
    combo = current_combo
    is_multi_clear = lines_to_clear >= 2

    for _ in range(lines_to_clear):
        combo += 1
        if is_multi_clear:
            points += combo * cfg.combo_points_multiplier
        elif combo == 1:
            points += 10
        else:
            points += 10 + (combo * cfg.combo_points_multiplier)

    # Multi-clear bonus (+30 when clearing 2+ lines with existing combo)
    if is_multi_clear and combo > lines_to_clear:
        points += cfg.multi_clear_bonus

    return points
