from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from snake_env.agent import TabularQAgent
from snake_env.runner import EpisodeResult
from snake_env.runner import RenderConfig
from snake_env.runner import run_agent_episodes


@dataclass(frozen=True)
class InferenceConfig:
    model_path: str = "models/snake_q_20x20.pkl"
    size: int = 20
    episodes: int = 1
    seed: int = 10_007
    render: bool = True
    render_every: int = 200
    render_delay: float = 0.02
    use_wandb: bool = True
    wandb_project: str = "snake-env"


def run_inference(config: InferenceConfig) -> list[dict[str, Any]]:
    """Load a trained Snake agent and run deterministic inference episodes."""
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}; run `uv run python -m snake_env.train` first")

    agent = TabularQAgent.load(model_path)

    run = None
    if config.use_wandb:
        import wandb

        os.environ.setdefault("WANDB_MODE", "offline")
        run = wandb.init(project=config.wandb_project, job_type="inference", config=asdict(config))

    def log_result(episode: int, result: EpisodeResult) -> None:
        logger.info(
            "inference_episode={} score={} length={} steps={} won={} reason={} reward={:.2f}",
            episode,
            result["score"],
            result["length"],
            result["steps"],
            result["won"],
            result.get("reason"),
            result["reward"],
        )
        if run is not None:
            run.log(
                {
                    "episode": episode,
                    "score": result["score"],
                    "length": result["length"],
                    "steps": result["steps"],
                    "won": int(result["won"]),
                    "reward": result["reward"],
                }
            )

    try:
        results = run_agent_episodes(
            agent,
            size=config.size,
            episodes=config.episodes,
            seed=config.seed,
            render=RenderConfig(
                enabled=config.render,
                every=config.render_every,
                delay=config.render_delay,
                show_initial_frame=True,
            ),
            on_episode_end=log_result,
        )
    finally:
        if run is not None:
            run.finish()

    wins = sum(1 for result in results if result["won"])
    logger.info("inference wins: {}/{}", wins, len(results))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained 20x20 Snake agent.")
    parser.add_argument("--model-path", default="models/snake_q_20x20.pkl")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10_007)
    parser.add_argument("--no-render", action="store_true", help="Disable terminal rendering.")
    parser.add_argument("--render-every", type=int, default=200, help="Render every N inference steps.")
    parser.add_argument("--render-delay", type=float, default=0.02, help="Seconds to sleep after each rendered frame.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "snake-env"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        model_path=args.model_path,
        size=args.size,
        episodes=args.episodes,
        seed=args.seed,
        render=not args.no_render,
        render_every=args.render_every,
        render_delay=args.render_delay,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )
    results = run_inference(config)
    if any(not result["won"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
