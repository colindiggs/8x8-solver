"""
Demo script to test the game engine and agents.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.game import Game, play_random_game
from src.core.blocks import BlockGenerator, visualize_all_blocks
from src.ai.agents.random_agent import RandomAgent
from src.ai.agents.heuristic_agent import HeuristicAgent


def demo_blocks():
    """Show all available block shapes."""
    print("=" * 60)
    print("AVAILABLE BLOCK SHAPES (Placeholder values)")
    print("=" * 60)

    generator = BlockGenerator()
    blocks = {name: block for name, block in generator.blocks.items()}

    print(visualize_all_blocks(blocks))
    print(f"\nTotal unique blocks: {len(blocks)}")


def demo_random_game():
    """Play a random game and show the result."""
    print("\n" + "=" * 60)
    print("RANDOM GAME DEMO")
    print("=" * 60)

    score = play_random_game(seed=42, verbose=True)
    print(f"\nFinal Score: {score}")


def demo_agents():
    """Compare random vs heuristic agent performance."""
    print("\n" + "=" * 60)
    print("AGENT COMPARISON")
    print("=" * 60)

    num_games = 20

    # Random agent
    random_agent = RandomAgent(seed=42)
    random_scores = []

    for i in range(num_games):
        game = Game(seed=i)
        random_agent.reset()

        while not game.is_game_over():
            move = random_agent.select_move(game)
            if move is None:
                break
            game.make_move(move.block_idx, move.row, move.col)

        random_scores.append(game.get_score())

    print(f"\nRandom Agent ({num_games} games):")
    print(f"  Average Score: {sum(random_scores) / len(random_scores):.1f}")
    print(f"  Max Score: {max(random_scores)}")
    print(f"  Min Score: {min(random_scores)}")

    # Heuristic agent
    heuristic_agent = HeuristicAgent()
    heuristic_scores = []

    for i in range(num_games):
        game = Game(seed=i)
        heuristic_agent.reset()

        while not game.is_game_over():
            move = heuristic_agent.select_move(game)
            if move is None:
                break
            game.make_move(move.block_idx, move.row, move.col)

        heuristic_scores.append(game.get_score())

    print(f"\nHeuristic Agent ({num_games} games):")
    print(f"  Average Score: {sum(heuristic_scores) / len(heuristic_scores):.1f}")
    print(f"  Max Score: {max(heuristic_scores)}")
    print(f"  Min Score: {min(heuristic_scores)}")

    improvement = (sum(heuristic_scores) / sum(random_scores) - 1) * 100
    print(f"\nHeuristic improvement over random: {improvement:+.1f}%")


def demo_single_game_with_heuristic():
    """Play one game with heuristic agent showing moves."""
    print("\n" + "=" * 60)
    print("HEURISTIC AGENT SINGLE GAME")
    print("=" * 60)

    game = Game(seed=123)
    agent = HeuristicAgent()

    turn = 0
    while not game.is_game_over() and turn < 50:  # Limit turns for demo
        move = agent.select_move(game)
        if move is None:
            break

        block = game.current_blocks[move.block_idx]
        result = game.make_move(move.block_idx, move.row, move.col)

        if result.success:
            turn += 1
            print(f"\nTurn {turn}: Placed {block.name} at ({move.row}, {move.col})")
            print(f"  Points: +{result.score_result.total_points} (Total: {game.get_score()})")

            if result.clear_result and result.clear_result.lines_cleared > 0:
                print(f"  Cleared {result.clear_result.lines_cleared} line(s)!")

            if result.new_batch_generated:
                print("  New block batch generated")

    print(f"\n{game}")
    print(f"\nFinal Score: {game.get_score()}")
    print(f"Total Turns: {turn}")


if __name__ == "__main__":
    print("8x8 Block Puzzle Solver - Demo")
    print("=" * 60)

    # Uncomment the demos you want to run:

    demo_blocks()
    # demo_random_game()
    demo_agents()
    # demo_single_game_with_heuristic()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("\nNote: These use PLACEHOLDER block shapes and scoring.")
    print("Run APK reverse engineering to get accurate parameters.")
