from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from snake_env.agent import TabularQAgent
from snake_env.env import SnakeEnv

EpisodeResult = dict[str, Any]
EpisodeCallback = Callable[[int, EpisodeResult], None]


@dataclass(frozen=True)
class RenderConfig:
    enabled: bool = False
    every: int = 200
    delay: float = 0.02
    show_initial_frame: bool = False

    def __post_init__(self) -> None:
        if self.every <= 0:
            raise ValueError("render_every must be positive")
        if self.delay < 0:
            raise ValueError("render_delay must be non-negative")


def run_agent_episodes(
    agent: TabularQAgent,
    *,
    size: int,
    episodes: int,
    seed: int,
    render: RenderConfig | None = None,
    on_episode_end: EpisodeCallback | None = None,
) -> list[EpisodeResult]:
    """Run deterministic policy episodes for evaluation or inference."""
    render = render or RenderConfig()
    env = SnakeEnv(size=size, render_mode="ansi" if render.enabled else None)
    results: list[EpisodeResult] = []

    for episode in range(1, episodes + 1):
        env.reset(seed=seed + episode - 1)
        terminated = truncated = False
        info: EpisodeResult = {"score": 0, "length": 1, "steps": 0, "won": False}
        total_reward = 0.0

        if render.enabled and render.show_initial_frame:
            render_frame(env, episode=episode, info=info, delay=0.0)

        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(agent.act(env))
            total_reward += reward
            if render.enabled and _should_render(info, terminated=terminated, truncated=truncated, every=render.every):
                render_frame(env, episode=episode, info=info, delay=render.delay)

        result = {**info, "episode": episode, "reward": total_reward}
        results.append(result)
        if on_episode_end is not None:
            on_episode_end(episode, result)

    return results


def render_frame(env: SnakeEnv, *, episode: int, info: EpisodeResult, delay: float) -> None:
    print("\x1b[2J\x1b[H", end="")
    print(env.render())
    print(f"episode={episode} score={info['score']} length={info['length']} steps={info['steps']}")
    if delay > 0:
        time.sleep(delay)


def _should_render(info: EpisodeResult, *, terminated: bool, truncated: bool, every: int) -> bool:
    return info["steps"] % every == 0 or terminated or truncated
