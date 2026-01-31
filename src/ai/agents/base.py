"""
Base agent interface for the 8x8 block puzzle solver.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from core.game import Game, Move, GameState


class BaseAgent(ABC):
    """
    Abstract base class for all game-playing agents.

    An agent receives a game state and returns the best move to make.
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def select_move(self, game: Game, explore: bool = True) -> Optional[Move]:
        """
        Select the best move for the current game state.

        Args:
            game: The current game instance
            explore: If True, allow exploration (random moves). If False, be greedy.

        Returns:
            The selected Move, or None if no valid moves exist.
        """
        pass

    def get_move_scores(self, game: Game) -> List[Tuple[Move, float]]:
        """
        Get scores for all valid moves (if the agent supports it).

        Args:
            game: The current game instance

        Returns:
            List of (Move, score) tuples, sorted by score descending.
            Returns empty list if not implemented.
        """
        return []

    def reset(self):
        """Reset any internal state (called at the start of each game)."""
        pass

    def on_game_end(self, final_score: int):
        """Called when a game ends. Useful for learning agents."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
