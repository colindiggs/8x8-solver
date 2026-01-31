"""
Block definitions and generation for the 8x8 block puzzle.

Block shapes are represented as 2D numpy arrays where 1 indicates a filled cell.
These are placeholder values - exact shapes should be extracted from the APK.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Block:
    """A block piece that can be placed on the board."""

    shape: np.ndarray
    name: str

    @property
    def height(self) -> int:
        return self.shape.shape[0]

    @property
    def width(self) -> int:
        return self.shape.shape[1]

    @property
    def cell_count(self) -> int:
        """Number of filled cells in this block."""
        return int(np.sum(self.shape))

    def __str__(self) -> str:
        lines = []
        for row in self.shape:
            lines.append("".join("#" if cell else " " for cell in row))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Block({self.name}, {self.height}x{self.width}, cells={self.cell_count})"


# ============================================================================
# DISCOVERED BLOCK DEFINITIONS
# Empirically discovered from gameplay screenshots on 2026-01-31
# ============================================================================

DISCOVERED_BLOCKS: Dict[str, List[List[int]]] = {
    # Lines (horizontal) - NO 1x1 exists, smallest is 1x2
    "line_2h": [[1, 1]],
    "line_3h": [[1, 1, 1]],
    "line_4h": [[1, 1, 1, 1]],

    # Lines (vertical)
    "line_2v": [[1], [1]],
    "line_3v": [[1], [1], [1]],
    "line_4v": [[1], [1], [1], [1]],
    "line_5v": [[1], [1], [1], [1], [1]],

    # Squares
    "square_2x2": [[1, 1], [1, 1]],
    "square_3x3": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],

    # Small L-shapes (2x2, 3 cells)
    "L_small_0": [[1, 0], [1, 1]],
    "L_small_90": [[1, 1], [1, 0]],
    "L_small_180": [[1, 1], [0, 1]],
    "L_small_270": [[0, 1], [1, 1]],

    # Large L-shapes (2x3, 4 cells)
    "L_big_0": [[1, 0], [1, 0], [1, 1]],
    "L_big_90": [[1, 1, 1], [1, 0, 0]],
    "L_big_180": [[1, 1], [0, 1], [0, 1]],
    "L_big_270": [[0, 0, 1], [1, 1, 1]],

    # T-shape (only one rotation confirmed, others likely exist)
    "T_shape_0": [[0, 1, 0], [1, 1, 1]],

    # S/Z shapes
    "S_shape": [[0, 1, 1], [1, 1, 0]],
    "Z_shape": [[1, 1, 0], [0, 1, 1]],
}

# Spawn weights - 1x2 is rare, others roughly equal
# TODO: Refine weights through more gameplay observation
DISCOVERED_SPAWN_WEIGHTS: Dict[str, float] = {
    # Lines
    "line_2h": 0.5,   # Rare
    "line_2v": 0.5,   # Rare
    "line_3h": 1.0,
    "line_3v": 1.0,
    "line_4h": 1.0,
    "line_4v": 1.0,
    "line_5v": 0.8,   # Less common

    # Squares
    "square_2x2": 1.0,
    "square_3x3": 0.7,  # Less common (big)

    # L-shapes
    "L_small_0": 1.0,
    "L_small_90": 1.0,
    "L_small_180": 1.0,
    "L_small_270": 1.0,
    "L_big_0": 0.8,
    "L_big_90": 0.8,
    "L_big_180": 0.8,
    "L_big_270": 0.8,

    # T-shape
    "T_shape_0": 0.8,

    # S/Z shapes
    "S_shape": 0.8,
    "Z_shape": 0.8,
}

# Use discovered blocks as the default
PLACEHOLDER_BLOCKS = DISCOVERED_BLOCKS
PLACEHOLDER_SPAWN_WEIGHTS = DISCOVERED_SPAWN_WEIGHTS


class BlockGenerator:
    """
    Generates random blocks according to spawn probabilities.

    The generator can be seeded for reproducible sequences,
    which is useful for testing and validation.
    """

    def __init__(
        self,
        blocks: Optional[Dict[str, np.ndarray]] = None,
        weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the block generator.

        Args:
            blocks: Dictionary mapping block names to shape arrays.
                    If None, uses placeholder blocks.
            weights: Dictionary mapping block names to spawn weights.
                     If None, uses uniform weights.
            seed: Random seed for reproducibility.
        """
        if blocks is None:
            self.blocks = {
                name: Block(np.array(shape, dtype=np.int8), name)
                for name, shape in PLACEHOLDER_BLOCKS.items()
            }
        else:
            self.blocks = {
                name: Block(np.array(shape, dtype=np.int8), name)
                for name, shape in blocks.items()
            }

        self.block_names = list(self.blocks.keys())

        # Set up weights
        if weights is None:
            weights = {name: 1.0 for name in self.block_names}

        # Normalize weights to probabilities
        total_weight = sum(weights.get(name, 1.0) for name in self.block_names)
        self.probabilities = np.array([
            weights.get(name, 1.0) / total_weight
            for name in self.block_names
        ])

        # Set up random generator
        self.rng = np.random.default_rng(seed)

    def generate_one(self) -> Block:
        """Generate a single random block."""
        idx = self.rng.choice(len(self.block_names), p=self.probabilities)
        return self.blocks[self.block_names[idx]]

    def generate_batch(self, count: int = 3) -> List[Block]:
        """Generate a batch of random blocks."""
        return [self.generate_one() for _ in range(count)]

    def get_block_by_name(self, name: str) -> Optional[Block]:
        """Get a specific block by name."""
        return self.blocks.get(name)

    def get_all_blocks(self) -> List[Block]:
        """Get all available block types."""
        return list(self.blocks.values())

    @classmethod
    def from_parameters_file(cls, filepath: str, seed: Optional[int] = None) -> BlockGenerator:
        """
        Load block definitions from a parameters JSON file.

        Expected format:
        {
            "blocks": {
                "name": [[0, 1], [1, 1]],
                ...
            },
            "spawn_weights": {
                "name": 1.0,
                ...
            }
        }
        """
        with open(filepath, 'r') as f:
            params = json.load(f)

        blocks = params.get('blocks', PLACEHOLDER_BLOCKS)
        weights = params.get('spawn_weights', None)

        return cls(blocks=blocks, weights=weights, seed=seed)


def load_blocks_from_json(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load block definitions and weights from a JSON file.

    Returns:
        Tuple of (blocks dict, weights dict)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    blocks = {
        name: np.array(shape, dtype=np.int8)
        for name, shape in data.get('blocks', {}).items()
    }

    weights = data.get('spawn_weights', {})

    return blocks, weights


def visualize_block(block: Block) -> str:
    """Create an ASCII visualization of a block."""
    lines = []
    for row in block.shape:
        line = ""
        for cell in row:
            line += "##" if cell else "  "
        lines.append(line)
    return "\n".join(lines)


def visualize_all_blocks(blocks: Dict[str, Block]) -> str:
    """Create a visualization of all blocks."""
    output = []
    for name, block in blocks.items():
        output.append(f"\n{name} ({block.height}x{block.width}, {block.cell_count} cells):")
        output.append(visualize_block(block))
    return "\n".join(output)
