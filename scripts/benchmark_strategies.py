"""
Benchmark all strategies to see current performance levels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai.strategies.registry import STRATEGIES, get_strategy
from core.game import Game
import numpy as np

def run_game(strategy, seed=None):
    """Run a single game and return the score."""
    game = Game(seed=seed)
    strategy.reset()

    moves = 0
    while not game.is_game_over():
        move = strategy.select_move(game)
        if move is None:
            break
        game.make_move(move.block_idx, move.row, move.col)
        moves += 1

    return game.get_score(), moves

def benchmark(n_games=20):
    """Benchmark all strategies."""
    print(f"Benchmarking {len(STRATEGIES)} strategies over {n_games} games each...\n")

    results = {}

    for strategy_id in STRATEGIES:
        scores = []
        moves_list = []

        for i in range(n_games):
            strategy = get_strategy(strategy_id, seed=i*1000)
            score, moves = run_game(strategy, seed=i)
            scores.append(score)
            moves_list.append(moves)

        results[strategy_id] = {
            'mean': np.mean(scores),
            'max': np.max(scores),
            'min': np.min(scores),
            'std': np.std(scores),
            'avg_moves': np.mean(moves_list)
        }

    # Sort by mean score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)

    print(f"{'Strategy':<20} {'Mean':>8} {'Max':>8} {'Min':>8} {'Std':>8} {'Moves':>8}")
    print("=" * 72)

    for strategy_id, stats in sorted_results:
        print(f"{strategy_id:<20} {stats['mean']:>8.0f} {stats['max']:>8.0f} {stats['min']:>8.0f} {stats['std']:>8.1f} {stats['avg_moves']:>8.1f}")

    return sorted_results

if __name__ == "__main__":
    benchmark()
