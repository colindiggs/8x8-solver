"""
Train an evolved strategy using genetic algorithm optimization.

This script evolves optimal weights for a multi-factor evaluation function
by running many games and selecting the fittest individuals.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from core.game import Game, Move
import json

@dataclass
class Individual:
    """An individual in the population with its weights."""
    weights: np.ndarray
    fitness: float = 0.0


class EvolvedEvaluator:
    """Evaluator with evolvable weights."""

    def __init__(self, weights: np.ndarray):
        # Weights for: [score, lines, combo, empty, options, holes, near_complete, future_score]
        self.weights = weights

    def evaluate_move(self, game: Game, move: Move) -> float:
        sim_game = game.copy()
        result = sim_game.make_move(move.block_idx, move.row, move.col)

        if not result.success:
            return -10000

        grid = sim_game.board.grid
        w = self.weights

        # Features
        score = result.score_result.total_points
        lines = result.score_result.lines_cleared
        combo = result.score_result.combo_level
        empty = 64 - np.sum(grid)

        future_moves = sim_game.get_valid_moves()
        options = len(future_moves)

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

        # Near-complete lines
        near_complete = 0
        for i in range(8):
            if np.sum(grid[i, :]) >= 6:
                near_complete += 1
            if np.sum(grid[:, i]) >= 6:
                near_complete += 1

        # Best future score (simplified lookahead)
        best_future = 0
        if options > 0:
            for fm in future_moves[:10]:
                sim2 = sim_game.copy()
                r2 = sim2.make_move(fm.block_idx, fm.row, fm.col)
                if r2.success:
                    best_future = max(best_future, r2.score_result.total_points)

        # Combine features with weights
        features = np.array([score, lines, combo, empty, options, holes, near_complete, best_future])
        return np.dot(features, w)

    def select_move(self, game: Game) -> Move:
        moves = game.get_valid_moves()
        if not moves:
            return None

        best_move = None
        best_score = float('-inf')

        for move in moves:
            score = self.evaluate_move(game, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


def run_game(evaluator: EvolvedEvaluator, seed: int) -> Tuple[int, int]:
    """Run a game and return (score, moves)."""
    game = Game(seed=seed)
    moves = 0

    while not game.is_game_over():
        move = evaluator.select_move(game)
        if move is None:
            break
        game.make_move(move.block_idx, move.row, move.col)
        moves += 1

    return game.get_score(), moves


def evaluate_fitness(weights: np.ndarray, n_games: int = 5) -> float:
    """Evaluate fitness of a weight vector over multiple games."""
    evaluator = EvolvedEvaluator(weights)
    scores = []

    for seed in range(n_games):
        score, _ = run_game(evaluator, seed)
        scores.append(score)

    # Fitness is mean score with bonus for consistency
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score - std_score * 0.1  # Slight penalty for high variance


def create_population(size: int, n_weights: int) -> List[Individual]:
    """Create initial population with random weights."""
    population = []
    for _ in range(size):
        # Initialize with reasonable ranges
        weights = np.array([
            np.random.uniform(0.5, 2.0),    # score
            np.random.uniform(10, 30),       # lines
            np.random.uniform(5, 20),        # combo
            np.random.uniform(0.2, 1.0),     # empty
            np.random.uniform(0.3, 1.5),     # options
            np.random.uniform(-15, -3),      # holes (negative)
            np.random.uniform(5, 20),        # near_complete
            np.random.uniform(0.2, 0.8),     # future_score
        ])
        population.append(Individual(weights=weights))
    return population


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """Create child from two parents using crossover."""
    mask = np.random.random(len(parent1.weights)) > 0.5
    child_weights = np.where(mask, parent1.weights, parent2.weights)
    return Individual(weights=child_weights)


def mutate(individual: Individual, mutation_rate: float = 0.2, mutation_strength: float = 0.3) -> Individual:
    """Mutate an individual's weights."""
    new_weights = individual.weights.copy()
    for i in range(len(new_weights)):
        if np.random.random() < mutation_rate:
            new_weights[i] *= (1 + np.random.uniform(-mutation_strength, mutation_strength))
    return Individual(weights=new_weights)


def evolve(
    population_size: int = 30,
    generations: int = 50,
    elite_size: int = 5,
    games_per_eval: int = 5
) -> Individual:
    """Run genetic algorithm to evolve optimal weights."""
    n_weights = 8  # Number of features

    print(f"Evolving with population={population_size}, generations={generations}")
    print(f"Features: score, lines, combo, empty, options, holes, near_complete, future_score")
    print()

    # Initialize population
    population = create_population(population_size, n_weights)

    best_ever = None
    best_fitness = float('-inf')

    for gen in range(generations):
        # Evaluate fitness
        for ind in population:
            ind.fitness = evaluate_fitness(ind.weights, games_per_eval)

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Track best
        if population[0].fitness > best_fitness:
            best_fitness = population[0].fitness
            best_ever = Individual(weights=population[0].weights.copy(), fitness=best_fitness)

        print(f"Gen {gen+1:3d}: Best={population[0].fitness:7.0f} | Avg={np.mean([i.fitness for i in population]):7.0f} | Best Ever={best_fitness:7.0f}")

        # Create next generation
        next_gen = []

        # Keep elite
        for i in range(elite_size):
            next_gen.append(Individual(weights=population[i].weights.copy()))

        # Fill rest with crossover and mutation
        while len(next_gen) < population_size:
            # Tournament selection
            idx1, idx2 = np.random.choice(len(population) // 2, 2, replace=False)
            parent1, parent2 = population[idx1], population[idx2]

            child = crossover(parent1, parent2)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen

    print()
    print("=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best fitness: {best_fitness:.0f}")
    print(f"Best weights:")
    labels = ['score', 'lines', 'combo', 'empty', 'options', 'holes', 'near_complete', 'future_score']
    for label, weight in zip(labels, best_ever.weights):
        print(f"  {label}: {weight:.3f}")

    return best_ever


def validate_best(individual: Individual, n_games: int = 20):
    """Validate the best individual over many games."""
    print()
    print(f"Validating over {n_games} games...")

    evaluator = EvolvedEvaluator(individual.weights)
    scores = []
    moves_list = []

    for seed in range(n_games):
        score, moves = run_game(evaluator, seed)
        scores.append(score)
        moves_list.append(moves)

    print(f"Mean score: {np.mean(scores):.0f}")
    print(f"Max score: {np.max(scores):.0f}")
    print(f"Min score: {np.min(scores):.0f}")
    print(f"Std: {np.std(scores):.0f}")
    print(f"Avg moves: {np.mean(moves_list):.0f}")

    return {
        'weights': individual.weights.tolist(),
        'mean_score': float(np.mean(scores)),
        'max_score': float(np.max(scores)),
        'std_score': float(np.std(scores)),
    }


if __name__ == "__main__":
    # Run evolution
    best = evolve(
        population_size=30,
        generations=40,
        elite_size=5,
        games_per_eval=5
    )

    # Validate
    results = validate_best(best, n_games=20)

    # Save weights
    output_path = Path(__file__).parent.parent / "data" / "evolved_weights.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWeights saved to {output_path}")
