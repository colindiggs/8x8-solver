"""
Arena system for strategy battles and tournaments.

Supports head-to-head comparisons, round-robin tournaments,
and comprehensive statistics tracking.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
from datetime import datetime

from core.game import Game
from ai.strategies import get_strategy, list_strategies, Strategy, StrategyInfo


@dataclass
class GameResult:
    """Result of a single game."""
    strategy_id: str
    score: int
    turns: int
    max_combo: int
    total_lines: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    """Result of a head-to-head match (multiple games)."""
    strategy1_id: str
    strategy2_id: str
    strategy1_wins: int
    strategy2_wins: int
    ties: int
    strategy1_avg_score: float
    strategy2_avg_score: float
    games_played: int

    @property
    def winner_id(self) -> Optional[str]:
        if self.strategy1_wins > self.strategy2_wins:
            return self.strategy1_id
        elif self.strategy2_wins > self.strategy1_wins:
            return self.strategy2_id
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['winner_id'] = self.winner_id
        return d


@dataclass
class StrategyStats:
    """Accumulated statistics for a strategy."""
    strategy_id: str
    games_played: int = 0
    total_score: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    max_score: int = 0
    min_score: int = float('inf')
    total_turns: int = 0
    max_combo_ever: int = 0
    total_lines: int = 0
    scores: List[int] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        return self.total_score / self.games_played if self.games_played else 0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.games_played if self.games_played else 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.ties
        return self.wins / total if total else 0

    @property
    def score_std(self) -> float:
        return float(np.std(self.scores)) if self.scores else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'win_rate': self.win_rate,
            'avg_score': self.avg_score,
            'max_score': self.max_score,
            'min_score': self.min_score if self.min_score != float('inf') else 0,
            'score_std': self.score_std,
            'avg_turns': self.avg_turns,
            'max_combo_ever': self.max_combo_ever,
            'total_lines': self.total_lines,
        }


class Arena:
    """
    The Arena manages strategy competitions.

    Supports:
    - Single game runs
    - Head-to-head matches
    - Round-robin tournaments
    - Live game streaming
    """

    def __init__(self, seed: Optional[int] = None):
        self.base_seed = seed or int(datetime.now().timestamp())
        self.stats: Dict[str, StrategyStats] = {}
        self.match_history: List[MatchResult] = []

    def run_game(
        self,
        strategy: Strategy,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> GameResult:
        """
        Run a single game with a strategy.

        Args:
            strategy: The strategy to use
            seed: Game seed for reproducibility
            callback: Optional callback(game, move, result) for each move

        Returns:
            GameResult with game statistics
        """
        game_seed = seed if seed is not None else self.base_seed
        game = Game(seed=game_seed)
        strategy.reset()

        max_combo = 0
        total_lines = 0

        while not game.game_over:
            move = strategy.select_move(game)
            if move is None:
                break

            result = game.make_move(move.block_idx, move.row, move.col)

            if result.success:
                max_combo = max(max_combo, result.score_result.combo_level)
                total_lines += result.score_result.lines_cleared

                if callback:
                    callback(game, move, result, strategy.explain_last_move())

        return GameResult(
            strategy_id=strategy.INFO.id,
            score=game.get_score(),
            turns=game.turn_count,
            max_combo=max_combo,
            total_lines=total_lines,
            seed=game_seed
        )

    def run_match(
        self,
        strategy1_id: str,
        strategy2_id: str,
        num_games: int = 10,
        callback: Optional[Callable] = None
    ) -> MatchResult:
        """
        Run a head-to-head match between two strategies.

        Both strategies play the same games (same seeds) for fairness.

        Args:
            strategy1_id: First strategy ID
            strategy2_id: Second strategy ID
            num_games: Number of games to play
            callback: Optional callback for progress

        Returns:
            MatchResult with match statistics
        """
        strategy1 = get_strategy(strategy1_id)
        strategy2 = get_strategy(strategy2_id)

        s1_wins = 0
        s2_wins = 0
        ties = 0
        s1_scores = []
        s2_scores = []

        for i in range(num_games):
            seed = self.base_seed + i

            # Both play same game
            r1 = self.run_game(strategy1, seed=seed)
            r2 = self.run_game(strategy2, seed=seed)

            s1_scores.append(r1.score)
            s2_scores.append(r2.score)

            if r1.score > r2.score:
                s1_wins += 1
            elif r2.score > r1.score:
                s2_wins += 1
            else:
                ties += 1

            # Update stats
            self._update_stats(r1, r2)

            if callback:
                callback(i + 1, num_games, r1, r2)

        result = MatchResult(
            strategy1_id=strategy1_id,
            strategy2_id=strategy2_id,
            strategy1_wins=s1_wins,
            strategy2_wins=s2_wins,
            ties=ties,
            strategy1_avg_score=np.mean(s1_scores),
            strategy2_avg_score=np.mean(s2_scores),
            games_played=num_games
        )

        self.match_history.append(result)
        return result

    def run_tournament(
        self,
        strategy_ids: List[str],
        games_per_match: int = 5,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run a round-robin tournament.

        Every strategy plays against every other strategy.

        Args:
            strategy_ids: List of strategy IDs to compete
            games_per_match: Games per head-to-head match
            callback: Optional progress callback

        Returns:
            Tournament results with rankings
        """
        n = len(strategy_ids)
        results_matrix: Dict[Tuple[str, str], MatchResult] = {}
        total_matches = n * (n - 1) // 2
        match_num = 0

        for i in range(n):
            for j in range(i + 1, n):
                match_num += 1
                s1, s2 = strategy_ids[i], strategy_ids[j]

                if callback:
                    callback('match_start', match_num, total_matches, s1, s2)

                result = self.run_match(s1, s2, num_games=games_per_match)
                results_matrix[(s1, s2)] = result
                results_matrix[(s2, s1)] = result

                if callback:
                    callback('match_end', match_num, total_matches, s1, s2, result)

        # Calculate rankings
        rankings = self._calculate_rankings(strategy_ids)

        return {
            'strategies': strategy_ids,
            'matches': [r.to_dict() for r in self.match_history],
            'rankings': rankings,
            'stats': {sid: self.stats[sid].to_dict() for sid in strategy_ids if sid in self.stats}
        }

    def run_battle_royale(
        self,
        strategy_ids: List[str],
        seed: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, GameResult]:
        """
        All strategies play the same game simultaneously.

        Args:
            strategy_ids: Strategies to compete
            seed: Shared game seed
            callback: Optional per-move callback

        Returns:
            Dict mapping strategy_id to GameResult
        """
        game_seed = seed if seed is not None else self.base_seed
        results = {}

        for sid in strategy_ids:
            strategy = get_strategy(sid)
            results[sid] = self.run_game(strategy, seed=game_seed, callback=callback)

        return results

    def _update_stats(self, *results: GameResult):
        """Update strategy statistics from game results."""
        for r in results:
            if r.strategy_id not in self.stats:
                self.stats[r.strategy_id] = StrategyStats(strategy_id=r.strategy_id)

            stats = self.stats[r.strategy_id]
            stats.games_played += 1
            stats.total_score += r.score
            stats.scores.append(r.score)
            stats.max_score = max(stats.max_score, r.score)
            stats.min_score = min(stats.min_score, r.score)
            stats.total_turns += r.turns
            stats.max_combo_ever = max(stats.max_combo_ever, r.max_combo)
            stats.total_lines += r.total_lines

    def _calculate_rankings(self, strategy_ids: List[str]) -> List[Dict[str, Any]]:
        """Calculate tournament rankings."""
        rankings = []

        for sid in strategy_ids:
            if sid in self.stats:
                stats = self.stats[sid]
                info = get_strategy(sid).INFO

                rankings.append({
                    'rank': 0,  # Will be set after sorting
                    'strategy_id': sid,
                    'name': info.name,
                    'emoji': info.emoji,
                    'wins': stats.wins,
                    'losses': stats.losses,
                    'ties': stats.ties,
                    'win_rate': stats.win_rate,
                    'avg_score': stats.avg_score,
                    'max_score': stats.max_score,
                })

        # Sort by win rate, then avg score
        rankings.sort(key=lambda x: (x['win_rate'], x['avg_score']), reverse=True)

        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current leaderboard based on accumulated stats."""
        return self._calculate_rankings(list(self.stats.keys()))
