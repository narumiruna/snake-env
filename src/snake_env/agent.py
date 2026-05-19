from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from random import Random

from snake_env.env import Action
from snake_env.env import Pos
from snake_env.env import SnakeEnv


def hamiltonian_cycle(size: int) -> list[Pos]:
    """Return a Hamiltonian cycle for an even-sized square board."""
    if size % 2 != 0:
        raise ValueError("size must be even")

    cells: list[Pos] = [Pos(x, 0) for x in range(size)]
    for x in range(size - 1, 0, -1):
        offset = size - 1 - x
        if offset % 2 == 0:
            cells.extend(Pos(x, y) for y in range(1, size))
        else:
            cells.extend(Pos(x, y) for y in range(size - 1, 0, -1))
    cells.extend(Pos(0, y) for y in range(size - 1, 0, -1))
    return cells


class HamiltonianPolicy:
    """Safe expert policy: following this cycle eventually fills the board."""

    def __init__(self, size: int) -> None:
        cycle = hamiltonian_cycle(size)
        self.size = size
        self.index = {pos: i for i, pos in enumerate(cycle)}
        self.next_pos = {pos: cycle[(i + 1) % len(cycle)] for i, pos in enumerate(cycle)}

    def act(self, env: SnakeEnv) -> int:
        target = self.next_pos[env.head]
        dx = target.x - env.head.x
        dy = target.y - env.head.y
        if dx == 1:
            return int(Action.RIGHT)
        if dx == -1:
            return int(Action.LEFT)
        if dy == 1:
            return int(Action.DOWN)
        if dy == -1:
            return int(Action.UP)
        raise RuntimeError(f"non-adjacent Hamiltonian move from {env.head} to {target}")


@dataclass
class TabularQAgent:
    """Tabular Q-learning agent with a Hamiltonian safety prior.

    The Q table is learned from episodes, while the prior guarantees safe action
    selection for states that were not seen often enough during training.
    """

    size: int
    alpha: float = 0.35
    gamma: float = 0.98
    q: defaultdict[tuple, list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        self.expert = HamiltonianPolicy(self.size)

    def state(self, env: SnakeEnv) -> tuple:
        return env.state_key()

    def act(self, env: SnakeEnv, *, greedy: bool = True, epsilon: float = 0.0, rng: Random | None = None) -> int:
        state = self.state(env)
        if rng is not None and not greedy and rng.random() < epsilon:
            return int(env.action_space.sample())
        values = self.q.get(state)
        if not values or max(values) <= 0.0:
            return self.expert.act(env)
        return max(range(4), key=lambda action: (values[action], action == self.expert.act(env)))

    def learn_step(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool) -> None:
        values = self.q[state]
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        values[action] += self.alpha * (target - values[action])

    def reinforce_expert_action(self, state: tuple, action: int, reward: float = 0.05) -> None:
        self.q[state][action] = max(self.q[state][action], reward)

    def save(self, path: str | Path) -> None:
        payload = {"size": self.size, "alpha": self.alpha, "gamma": self.gamma, "q": dict(self.q)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> TabularQAgent:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        agent = cls(size=payload["size"], alpha=payload["alpha"], gamma=payload["gamma"])
        agent.q.update(payload["q"])
        return agent
