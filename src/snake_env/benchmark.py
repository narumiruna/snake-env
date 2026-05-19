from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from dataclasses import dataclass

from loguru import logger

from snake_env.metrics import PerformanceMetrics
from snake_env.models import EfficientHamiltonianPolicy
from snake_env.models import HamiltonianPolicy
from snake_env.runner import run_agent_episodes
from snake_env.wandb_utils import init_wandb


@dataclass(frozen=True)
class BenchmarkConfig:
    size: int = 20
    episodes: int = 5
    seed: int = 10_007
    wandb_project: str = "snake-env"


def run_benchmark(config: BenchmarkConfig) -> dict[str, PerformanceMetrics]:
    run = init_wandb(project=config.wandb_project, job_type="benchmark", config=asdict(config))
    policies = {
        "hamiltonian": HamiltonianPolicy(config.size),
        "efficient_hamiltonian": EfficientHamiltonianPolicy(config.size),
    }
    metrics: dict[str, PerformanceMetrics] = {}

    try:
        for name, policy in policies.items():
            results = run_agent_episodes(policy, size=config.size, episodes=config.episodes, seed=config.seed)
            metric = PerformanceMetrics.from_results(results)
            metrics[name] = metric
            run.log({f"benchmark/{name}/{key}": value for key, value in metric.as_wandb_dict().items()})
            logger.info("{} metrics: {}", name, metric)

        baseline = metrics["hamiltonian"]
        efficient = metrics["efficient_hamiltonian"]
        improvement = baseline.avg_steps_to_win - efficient.avg_steps_to_win
        improvement_pct = improvement / baseline.avg_steps_to_win
        run.log(
            {
                "benchmark/step_improvement": improvement,
                "benchmark/step_improvement_pct": improvement_pct,
                "benchmark/selected_model": "efficient_hamiltonian",
            }
        )
        logger.info("step improvement: {:.0f} ({:.2%})", improvement, improvement_pct)
    finally:
        run.finish()

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Snake policies with win-first efficiency metrics.")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_007)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "snake-env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_benchmark(
        BenchmarkConfig(size=args.size, episodes=args.episodes, seed=args.seed, wandb_project=args.wandb_project)
    )
    if metrics["efficient_hamiltonian"].win_rate < 1.0:
        raise SystemExit("efficient_hamiltonian did not win every episode")
    if metrics["efficient_hamiltonian"].avg_steps_to_win >= metrics["hamiltonian"].avg_steps_to_win:
        raise SystemExit("efficient_hamiltonian did not improve average steps to win")


if __name__ == "__main__":
    main()
