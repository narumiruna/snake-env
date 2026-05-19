from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from snake_env.agent import TabularQAgent
from snake_env.env import SnakeEnv
from snake_env.runner import EpisodeResult
from snake_env.runner import RenderConfig
from snake_env.runner import run_agent_episodes


@dataclass(frozen=True)
class TrainConfig:
    size: int = 20
    episodes: int = 3
    seed: int = 7
    model_path: str = "models/snake_q_20x20.pkl"
    use_wandb: bool = True
    wandb_project: str = "snake-env"


def train(config: TrainConfig) -> TabularQAgent:
    env = SnakeEnv(size=config.size)
    agent = TabularQAgent(size=config.size)

    run = None
    if config.use_wandb:
        import wandb

        os.environ.setdefault("WANDB_MODE", "offline")
        run = wandb.init(project=config.wandb_project, config=asdict(config))

    for episode in range(config.episodes):
        env.reset(seed=config.seed + episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            state = agent.state(env)
            action = agent.expert.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            del obs
            shaped_reward = reward + (0.05 if not terminated and not truncated else 0.0)
            next_state = agent.state(env)
            agent.learn_step(state, action, shaped_reward, next_state, terminated or truncated)
            agent.reinforce_expert_action(state, action)
            total_reward += reward

        logger.info(
            "episode={} score={} length={} steps={} reward={:.2f} won={}",
            episode + 1,
            info["score"],
            info["length"],
            info["steps"],
            total_reward,
            info["won"],
        )
        if run is not None:
            run.log(
                {
                    "score": info["score"],
                    "length": info["length"],
                    "steps": info["steps"],
                    "won": int(info["won"]),
                    "reward": total_reward,
                }
            )

    agent.save(config.model_path)
    logger.info("saved model to {}", config.model_path)
    if run is not None:
        run.save(config.model_path)
        run.finish()
    return agent


def evaluate(
    model_path: str | Path,
    *,
    size: int = 20,
    episodes: int = 5,
    seed: int = 1000,
    render: bool = False,
    render_every: int = 200,
    render_delay: float = 0.02,
) -> list[EpisodeResult]:
    agent = TabularQAgent.load(model_path)

    def log_result(episode: int, result: EpisodeResult) -> None:
        logger.info(
            "eval_episode={} score={} steps={} won={} reason={}",
            episode,
            result["score"],
            result["steps"],
            result["won"],
            result.get("reason"),
        )

    return run_agent_episodes(
        agent,
        size=size,
        episodes=episodes,
        seed=seed,
        render=RenderConfig(enabled=render, every=render_every, delay=render_delay),
        on_episode_end=log_result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a 20x20 Snake RL agent.")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", default="models/snake_q_20x20.pkl")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "snake-env"))
    parser.add_argument("--render", action="store_true", help="Show evaluation games in the terminal.")
    parser.add_argument("--render-every", type=int, default=200, help="Render every N evaluation steps.")
    parser.add_argument("--render-delay", type=float, default=0.02, help="Seconds to sleep after each rendered frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        size=args.size,
        episodes=args.episodes,
        seed=args.seed,
        model_path=args.model_path,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )
    train(config)
    results = evaluate(
        args.model_path,
        size=args.size,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        render=args.render,
        render_every=args.render_every,
        render_delay=args.render_delay,
    )
    wins = sum(1 for result in results if result["won"])
    logger.info("evaluation wins: {}/{}", wins, len(results))
    if wins != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
