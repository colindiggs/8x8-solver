"""
Strategy implementations for the 8x8 Block Puzzle Arena.

Each strategy represents a different algorithmic approach to playing the game,
designed to be simple, explainable, and comparable.
"""
from .base import Strategy, StrategyInfo
from .registry import STRATEGIES, get_strategy, list_strategies

__all__ = [
    'Strategy',
    'StrategyInfo',
    'STRATEGIES',
    'get_strategy',
    'list_strategies',
]
