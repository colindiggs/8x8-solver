"""
Base strategy class for the 8x8 Block Puzzle Arena.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from core.game import Game, Move


@dataclass
class StrategyInfo:
    """Metadata about a strategy for display and comparison."""
    id: str
    name: str
    short_desc: str   # One-line summary
    algorithm: str    # Technical description of the algorithm
    complexity: str   # "O(n)", "O(n^2)", etc. or "Simple", "Medium", "Advanced"
    category: str     # "Greedy", "Defensive", "Aggressive", "ML-Inspired"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'short_desc': self.short_desc,
            'algorithm': self.algorithm,
            'complexity': self.complexity,
            'category': self.category,
        }


class Strategy(ABC):
    """
    Abstract base class for all game-playing strategies.

    Each strategy implements a different approach to playing the game.
    Strategies should be:
    - Explainable: Clear reasoning for each decision
    - Comparable: Can compete head-to-head
    - Educational: Demonstrate different algorithmic approaches
    """

    # Override this in subclasses
    INFO: StrategyInfo = StrategyInfo(
        id="base",
        name="Base Strategy",
        short_desc="Abstract base class",
        algorithm="Override this in subclasses",
        complexity="N/A",
        category="N/A"
    )

    def __init__(self, seed: Optional[int] = None):
        """Initialize strategy with optional random seed."""
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.move_explanations: List[str] = []

    @abstractmethod
    def select_move(self, game: Game) -> Optional[Move]:
        """
        Select the best move according to this strategy.

        Args:
            game: Current game state

        Returns:
            The selected move, or None if no valid moves
        """
        pass

    def get_move_scores(self, game: Game) -> List[Tuple[Move, float, str]]:
        """
        Score all valid moves and explain each score.

        Returns:
            List of (move, score, explanation) tuples
        """
        moves = game.get_valid_moves()
        if not moves:
            return []

        scored = []
        for move in moves:
            score, explanation = self._evaluate_move(game, move)
            scored.append((move, score, explanation))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _evaluate_move(self, game: Game, move: Move) -> Tuple[float, str]:
        """
        Evaluate a single move. Override in subclasses.

        Returns:
            (score, explanation) tuple
        """
        return 0.0, "Base evaluation"

    def explain_last_move(self) -> str:
        """Get explanation for the last move made."""
        if self.move_explanations:
            return self.move_explanations[-1]
        return "No moves made yet"

    def reset(self):
        """Reset strategy state for a new game."""
        self.move_explanations = []

    @classmethod
    def get_info(cls) -> StrategyInfo:
        """Get strategy metadata."""
        return cls.INFO
