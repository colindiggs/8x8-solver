"""
FastAPI backend for the 8x8 Block Puzzle Strategy Arena.

Supports:
- Multiple concurrent games
- Strategy comparisons
- Battle royale mode
- Live WebSocket streaming
"""
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.game import Game
from ai.strategies import get_strategy, list_strategies, STRATEGIES
from ai.arena import Arena, GameResult


app = FastAPI(title="8x8 Block Puzzle Strategy Arena")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve the arena UI."""
    ui_path = Path(__file__).parent.parent.parent / "ui" / "arena.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    # Fallback to old UI
    ui_path = Path(__file__).parent.parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return HTMLResponse("<h1>8x8 Strategy Arena</h1><p>UI not found.</p>")


@app.get("/api/strategies")
async def get_strategies():
    """List all available strategies."""
    strategies = []
    for strategy_id, cls in STRATEGIES.items():
        info = cls.INFO
        strategies.append(info.to_dict())
    return {"strategies": strategies}


@app.get("/api/strategies/{strategy_id}")
async def get_strategy_info(strategy_id: str):
    """Get detailed info about a specific strategy."""
    if strategy_id not in STRATEGIES:
        return {"error": f"Strategy not found: {strategy_id}"}
    info = STRATEGIES[strategy_id].INFO
    return info.to_dict()


@app.websocket("/ws/battle")
async def battle_websocket(websocket: WebSocket):
    """
    WebSocket for live battle visualization.

    Expects config:
    {
        "mode": "battle_royale" | "head_to_head" | "single",
        "strategies": ["strategy1", "strategy2", ...],
        "speed": 0.1,
        "num_games": 10,
        "seed": optional
    }
    """
    await websocket.accept()

    try:
        config = await websocket.receive_json()
        mode = config.get("mode", "battle_royale")
        strategy_ids = config.get("strategies", ["random", "greedy"])
        speed = config.get("speed", 0.1)
        num_games = config.get("num_games", 1)
        seed = config.get("seed")

        await websocket.send_json({
            "type": "battle_start",
            "mode": mode,
            "strategies": strategy_ids,
            "num_games": num_games
        })

        if mode == "battle_royale":
            await run_battle_royale(websocket, strategy_ids, speed, num_games, seed)
        elif mode == "head_to_head":
            await run_head_to_head(websocket, strategy_ids, speed, num_games, seed)
        elif mode == "single":
            await run_single_strategy(websocket, strategy_ids[0], speed, num_games, seed)
        elif mode == "tournament":
            await run_tournament(websocket, strategy_ids, speed, num_games, seed)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


async def run_battle_royale(
    websocket: WebSocket,
    strategy_ids: List[str],
    speed: float,
    num_games: int,
    seed: Optional[int]
):
    """Run all strategies on the same game, streaming state."""
    import numpy as np

    base_seed = seed if seed is not None else np.random.randint(0, 100000)
    overall_stats = {sid: {"wins": 0, "total_score": 0, "games": 0} for sid in strategy_ids}

    for game_num in range(num_games):
        game_seed = base_seed + game_num

        # Initialize games for all strategies
        games = {}
        strategies = {}
        results = {}
        max_combos = {sid: 0 for sid in strategy_ids}
        total_lines = {sid: 0 for sid in strategy_ids}

        for sid in strategy_ids:
            games[sid] = Game(seed=game_seed)
            strategies[sid] = get_strategy(sid, seed=game_seed)

        # Send game start
        initial_states = {}
        for sid in strategy_ids:
            state = games[sid].get_state()
            initial_states[sid] = state.to_dict()

        await websocket.send_json({
            "type": "game_start",
            "game_num": game_num + 1,
            "seed": game_seed,
            "states": initial_states
        })

        # Play all games in parallel (step by step)
        active = set(strategy_ids)
        turn = 0

        while active:
            turn += 1
            turn_results = {}

            for sid in list(active):
                game = games[sid]
                strategy = strategies[sid]

                if game.game_over:
                    active.discard(sid)
                    continue

                move = strategy.select_move(game)
                if move is None:
                    active.discard(sid)
                    continue

                result = game.make_move(move.block_idx, move.row, move.col)

                if result.success:
                    max_combos[sid] = max(max_combos[sid], result.score_result.combo_level)
                    total_lines[sid] += result.score_result.lines_cleared

                    state = game.get_state()
                    turn_results[sid] = {
                        "state": state.to_dict(),
                        "move": {"block_idx": move.block_idx, "row": move.row, "col": move.col},
                        "score_gained": result.score_result.total_points,
                        "lines_cleared": result.score_result.lines_cleared,
                        "combo": result.score_result.combo_level,
                        "explanation": strategy.explain_last_move(),
                        "game_over": game.game_over
                    }

                    if game.game_over:
                        active.discard(sid)
                else:
                    active.discard(sid)

            if turn_results:
                await websocket.send_json({
                    "type": "turn",
                    "turn": turn,
                    "results": turn_results
                })

                if speed > 0:
                    await asyncio.sleep(speed)

        # Game ended - determine winner
        final_scores = {sid: games[sid].get_score() for sid in strategy_ids}
        winner_id = max(final_scores, key=final_scores.get)
        max_score = final_scores[winner_id]

        # Check for ties
        winners = [sid for sid, score in final_scores.items() if score == max_score]

        for sid in strategy_ids:
            overall_stats[sid]["total_score"] += final_scores[sid]
            overall_stats[sid]["games"] += 1
            if sid in winners:
                overall_stats[sid]["wins"] += 1

        await websocket.send_json({
            "type": "game_end",
            "game_num": game_num + 1,
            "final_scores": final_scores,
            "winner": winners[0] if len(winners) == 1 else None,
            "tied": winners if len(winners) > 1 else None,
            "details": {
                sid: {
                    "score": final_scores[sid],
                    "turns": games[sid].turn_count,
                    "max_combo": max_combos[sid],
                    "lines_cleared": total_lines[sid]
                }
                for sid in strategy_ids
            },
            "overall_stats": overall_stats
        })

    # Send final summary
    rankings = sorted(
        overall_stats.items(),
        key=lambda x: (x[1]["wins"], x[1]["total_score"]),
        reverse=True
    )

    await websocket.send_json({
        "type": "battle_complete",
        "rankings": [
            {
                "rank": i + 1,
                "strategy_id": sid,
                "wins": stats["wins"],
                "avg_score": stats["total_score"] / stats["games"] if stats["games"] else 0,
                "total_games": stats["games"]
            }
            for i, (sid, stats) in enumerate(rankings)
        ]
    })


async def run_head_to_head(
    websocket: WebSocket,
    strategy_ids: List[str],
    speed: float,
    num_games: int,
    seed: Optional[int]
):
    """Run two strategies head-to-head."""
    if len(strategy_ids) < 2:
        await websocket.send_json({"type": "error", "message": "Need at least 2 strategies"})
        return

    # Just use first two
    await run_battle_royale(websocket, strategy_ids[:2], speed, num_games, seed)


async def run_single_strategy(
    websocket: WebSocket,
    strategy_id: str,
    speed: float,
    num_games: int,
    seed: Optional[int]
):
    """Run a single strategy for training/visualization."""
    await run_battle_royale(websocket, [strategy_id], speed, num_games, seed)


async def run_tournament(
    websocket: WebSocket,
    strategy_ids: List[str],
    speed: float,
    games_per_match: int,
    seed: Optional[int]
):
    """Run a round-robin tournament."""
    import numpy as np

    base_seed = seed if seed is not None else np.random.randint(0, 100000)
    n = len(strategy_ids)
    total_matches = n * (n - 1) // 2
    match_num = 0

    stats = {sid: {"wins": 0, "losses": 0, "ties": 0, "total_score": 0, "games": 0}
             for sid in strategy_ids}

    await websocket.send_json({
        "type": "tournament_start",
        "strategies": strategy_ids,
        "total_matches": total_matches,
        "games_per_match": games_per_match
    })

    for i in range(n):
        for j in range(i + 1, n):
            match_num += 1
            s1, s2 = strategy_ids[i], strategy_ids[j]

            await websocket.send_json({
                "type": "match_start",
                "match_num": match_num,
                "total_matches": total_matches,
                "strategy1": s1,
                "strategy2": s2
            })

            s1_wins = 0
            s2_wins = 0
            s1_total = 0
            s2_total = 0

            for game_num in range(games_per_match):
                game_seed = base_seed + match_num * 1000 + game_num

                # Play same game with both strategies
                game1 = Game(seed=game_seed)
                game2 = Game(seed=game_seed)
                strat1 = get_strategy(s1, seed=game_seed)
                strat2 = get_strategy(s2, seed=game_seed)

                # Play game 1
                while not game1.game_over:
                    move = strat1.select_move(game1)
                    if move is None:
                        break
                    game1.make_move(move.block_idx, move.row, move.col)

                # Play game 2
                while not game2.game_over:
                    move = strat2.select_move(game2)
                    if move is None:
                        break
                    game2.make_move(move.block_idx, move.row, move.col)

                score1 = game1.get_score()
                score2 = game2.get_score()
                s1_total += score1
                s2_total += score2

                if score1 > score2:
                    s1_wins += 1
                elif score2 > score1:
                    s2_wins += 1

                if speed > 0:
                    await asyncio.sleep(speed * 0.5)

            # Update stats
            stats[s1]["total_score"] += s1_total
            stats[s2]["total_score"] += s2_total
            stats[s1]["games"] += games_per_match
            stats[s2]["games"] += games_per_match

            if s1_wins > s2_wins:
                stats[s1]["wins"] += 1
                stats[s2]["losses"] += 1
                winner = s1
            elif s2_wins > s1_wins:
                stats[s2]["wins"] += 1
                stats[s1]["losses"] += 1
                winner = s2
            else:
                stats[s1]["ties"] += 1
                stats[s2]["ties"] += 1
                winner = None

            await websocket.send_json({
                "type": "match_end",
                "match_num": match_num,
                "strategy1": s1,
                "strategy2": s2,
                "strategy1_wins": s1_wins,
                "strategy2_wins": s2_wins,
                "strategy1_avg": s1_total / games_per_match,
                "strategy2_avg": s2_total / games_per_match,
                "winner": winner,
                "current_standings": stats
            })

    # Final rankings
    rankings = sorted(
        [(sid, s) for sid, s in stats.items()],
        key=lambda x: (x[1]["wins"], -x[1]["losses"], x[1]["total_score"]),
        reverse=True
    )

    await websocket.send_json({
        "type": "tournament_complete",
        "rankings": [
            {
                "rank": i + 1,
                "strategy_id": sid,
                "wins": s["wins"],
                "losses": s["losses"],
                "ties": s["ties"],
                "avg_score": s["total_score"] / s["games"] if s["games"] else 0
            }
            for i, (sid, s) in enumerate(rankings)
        ]
    })


def main():
    """Run the arena server."""
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
