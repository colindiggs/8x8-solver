"""
Random agent - baseline for comparison.
"""
import random
from typing import Optional

from .base import BaseAgent
from core.game import Game, Move


class RandomAgent(BaseAgent):
    """
    Agent that selects moves uniformly at random.

    This serves as a baseline for comparing other agents.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="RandomAgent")
        self.rng = random.Random(seed)

    def select_move(self, game: Game, explore: bool = True) -> Optional[Move]:
        """Select a random valid move."""
        moves = game.get_valid_moves()

        if not moves:
            return None

        return self.rng.choice(moves)

    def reset(self):
        """No state to reset."""
        pass
