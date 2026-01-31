"""
FastAPI backend for 8x8 Block Puzzle training visualization.
"""
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.game import Game
from core.blocks import BlockGenerator, DISCOVERED_SPAWN_WEIGHTS
from ai.agents.random_agent import RandomAgent
from ai.agents.heuristic_agent import HeuristicAgent


app = FastAPI(title="8x8 Block Puzzle Trainer")

# Store active training sessions
active_sessions: Dict[str, dict] = {}


@dataclass
class TrainingConfig:
    """Configuration for a training session."""
    agent_type: str = "heuristic"
    num_games: int = 100
    speed: float = 0.1  # seconds between moves (0 = instant)
    seed: Optional[int] = None


@dataclass
class GameMetrics:
    """Metrics for a single game."""
    game_id: int
    final_score: int
    turns: int
    max_combo: int
    lines_cleared: int


def create_agent(agent_type: str, seed: Optional[int] = None):
    """Create an agent by type name."""
    if agent_type == "random":
        return RandomAgent(seed=seed)
    elif agent_type == "heuristic":
        return HeuristicAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def run_single_game(
    game_id: int,
    agent,
    websocket: WebSocket,
    speed: float = 0.1,
    seed: Optional[int] = None
) -> GameMetrics:
    """Run a single game and stream state updates via WebSocket."""
    game = Game(seed=seed)
    max_combo = 0
    total_lines = 0

    # Send initial state
    state = game.get_state()
    await websocket.send_json({
        "type": "game_start",
        "game_id": game_id,
        "state": state.to_dict()
    })

    while not game.game_over:
        move = agent.select_move(game)

        if move is None:
            break

        result = game.make_move(move.block_idx, move.row, move.col)

        if result.success:
            max_combo = max(max_combo, result.score_result.combo_level)
            total_lines += result.score_result.lines_cleared

            # Send move update
            state = game.get_state()
            await websocket.send_json({
                "type": "move",
                "game_id": game_id,
                "move": {
                    "block_idx": move.block_idx,
                    "row": move.row,
                    "col": move.col
                },
                "result": {
                    "score": result.score_result.total_points,
                    "lines_cleared": result.score_result.lines_cleared,
                    "combo": result.score_result.combo_level
                },
                "state": state.to_dict()
            })

            if speed > 0:
                await asyncio.sleep(speed)

    metrics = GameMetrics(
        game_id=game_id,
        final_score=game.get_score(),
        turns=game.turn_count,
        max_combo=max_combo,
        lines_cleared=total_lines
    )

    # Send game end
    await websocket.send_json({
        "type": "game_end",
        "game_id": game_id,
        "metrics": asdict(metrics)
    })

    return metrics


@app.websocket("/ws/train")
async def training_websocket(websocket: WebSocket):
    """WebSocket endpoint for training visualization."""
    await websocket.accept()

    try:
        # Wait for config
        config_data = await websocket.receive_json()
        config = TrainingConfig(**config_data.get("config", {}))

        agent = create_agent(config.agent_type, config.seed)

        await websocket.send_json({
            "type": "training_start",
            "config": asdict(config)
        })

        all_metrics: List[GameMetrics] = []
        running_avg = 0

        for game_id in range(config.num_games):
            metrics = await run_single_game(
                game_id=game_id,
                agent=agent,
                websocket=websocket,
                speed=config.speed,
                seed=(config.seed + game_id) if config.seed else None
            )
            all_metrics.append(metrics)

            # Calculate running stats
            scores = [m.final_score for m in all_metrics]
            running_avg = sum(scores) / len(scores)

            await websocket.send_json({
                "type": "stats_update",
                "games_completed": game_id + 1,
                "running_avg": running_avg,
                "max_score": max(scores),
                "min_score": min(scores),
                "avg_turns": sum(m.turns for m in all_metrics) / len(all_metrics),
                "avg_combo": sum(m.max_combo for m in all_metrics) / len(all_metrics)
            })

        # Send final summary
        scores = [m.final_score for m in all_metrics]
        await websocket.send_json({
            "type": "training_complete",
            "total_games": len(all_metrics),
            "final_avg": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores)
        })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


@app.get("/")
async def root():
    """Serve the main visualization page."""
    ui_path = Path(__file__).parent.parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return HTMLResponse("<h1>8x8 Puzzle Trainer</h1><p>UI not found. Run from project root.</p>")


@app.get("/api/agents")
async def list_agents():
    """List available agent types."""
    return {
        "agents": [
            {"id": "random", "name": "Random Agent", "description": "Makes random valid moves"},
            {"id": "heuristic", "name": "Heuristic Agent", "description": "Uses hand-crafted evaluation function"}
        ]
    }


@app.get("/api/blocks")
async def list_blocks():
    """List all block shapes."""
    generator = BlockGenerator()
    blocks = []
    for block in generator.get_all_blocks():
        blocks.append({
            "name": block.name,
            "width": block.width,
            "height": block.height,
            "cells": block.cell_count,
            "shape": block.shape.tolist()
        })
    return {"blocks": blocks}


def main():
    """Run the server."""
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
