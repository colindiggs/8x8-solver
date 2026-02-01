"""
Strategy registry - central place to access all available strategies.
"""
from typing import Dict, List, Optional, Type

from .base import Strategy, StrategyInfo
from .simple_strategies import (
    RandomStrategy,
    ComboChaserStrategy,
    SurvivalistStrategy,
)
from .advanced_strategies import (
    BalancedStrategy,
    DefensiveStrategy,
)
from .evolved_strategies import (
    DeepLookaheadStrategy,
    ComboMasterStrategy,
    TunedBalancedStrategy,
    SurvivalExpertStrategy,
)


# Registry of all available strategies (curated set of ~5 distinct approaches)
STRATEGIES: Dict[str, Type[Strategy]] = {
    # Baseline
    'random': RandomStrategy,

    # Core strategies (distinct approaches)
    'tuned_balanced': TunedBalancedStrategy,    # Multi-factor balanced
    'survival_expert': SurvivalExpertStrategy,  # Defensive/longevity
    'combo_master': ComboMasterStrategy,        # Aggressive combo focus
    'deep_lookahead': DeepLookaheadStrategy,    # Search-based
}


def get_strategy(strategy_id: str, seed: Optional[int] = None) -> Strategy:
    """
    Get a strategy instance by ID.

    Args:
        strategy_id: The strategy identifier
        seed: Optional random seed

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy_id is not found
    """
    if strategy_id not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {strategy_id}. Available: {available}")

    return STRATEGIES[strategy_id](seed=seed)


def list_strategies() -> List[StrategyInfo]:
    """
    Get info about all available strategies.

    Returns:
        List of StrategyInfo objects
    """
    return [cls.INFO for cls in STRATEGIES.values()]


def get_strategies_by_category() -> Dict[str, List[StrategyInfo]]:
    """
    Get strategies grouped by category.

    Returns:
        Dict mapping category names to lists of StrategyInfo
    """
    by_category: Dict[str, List[StrategyInfo]] = {}

    for cls in STRATEGIES.values():
        info = cls.INFO
        if info.category not in by_category:
            by_category[info.category] = []
        by_category[info.category].append(info)

    return by_category
