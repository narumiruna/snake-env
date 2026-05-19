from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

EpisodeResult = dict[str, Any]


@dataclass(frozen=True)
class PerformanceMetrics:
    """Win-first efficiency metrics for Snake model selection."""

    episodes: int
    wins: int
    win_rate: float
    avg_steps: float
    avg_steps_to_win: float
    avg_score: float
    avg_reward: float
    efficiency_score: float

    @classmethod
    def from_results(cls, results: list[EpisodeResult]) -> PerformanceMetrics:
        if not results:
            raise ValueError("results must not be empty")

        wins = sum(1 for result in results if result["won"])
        won_results = [result for result in results if result["won"]]
        avg_steps_to_win = mean(result["steps"] for result in won_results) if won_results else float("inf")
        win_rate = wins / len(results)
        avg_score = mean(result["score"] for result in results)
        avg_steps = mean(result["steps"] for result in results)
        avg_reward = mean(result["reward"] for result in results)

        # Win rate dominates model ranking; among equally reliable models, fewer
        # steps to win is better. This makes efficiency an explicit objective.
        efficiency_score = (win_rate * 1_000_000.0) + avg_score - avg_steps_to_win

        return cls(
            episodes=len(results),
            wins=wins,
            win_rate=win_rate,
            avg_steps=avg_steps,
            avg_steps_to_win=avg_steps_to_win,
            avg_score=avg_score,
            avg_reward=avg_reward,
            efficiency_score=efficiency_score,
        )

    def as_wandb_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {
            f"{prefix}episodes": self.episodes,
            f"{prefix}wins": self.wins,
            f"{prefix}win_rate": self.win_rate,
            f"{prefix}avg_steps": self.avg_steps,
            f"{prefix}avg_steps_to_win": self.avg_steps_to_win,
            f"{prefix}avg_score": self.avg_score,
            f"{prefix}avg_reward": self.avg_reward,
            f"{prefix}efficiency_score": self.efficiency_score,
        }
