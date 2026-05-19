from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from random import Random
from typing import Any

from snake_env.env import SnakeEnv
from snake_env.models.efficient import EfficientHamiltonianPolicy
from snake_env.models.hamiltonian import HamiltonianPolicy
from snake_env.models.hamiltonian import SnakePolicy

EXPERTS: dict[str, type[HamiltonianPolicy]] = {
    "hamiltonian": HamiltonianPolicy,
    "efficient_hamiltonian": EfficientHamiltonianPolicy,
}


@dataclass
class TabularQAgent(SnakePolicy):
    """Tabular Q-learning agent with a safe expert-policy prior."""

    size: int
    alpha: float = 0.35
    gamma: float = 0.98
    expert_name: str = "efficient_hamiltonian"
    q: defaultdict[tuple, list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        self.expert = EXPERTS[self.expert_name](self.size)

    def state(self, env: SnakeEnv) -> tuple:
        return env.state_key()

    def act(self, env: SnakeEnv, *, greedy: bool = True, epsilon: float = 0.0, rng: Random | None = None) -> int:
        state = self.state(env)
        if rng is not None and not greedy and rng.random() < epsilon:
            return int(env.action_space.sample())
        values = self.q.get(state)
        expert_action = self.expert.act(env)
        if not values or max(values) <= 0.0:
            return expert_action
        return max(range(4), key=lambda action: (values[action], action == expert_action))

    def learn_step(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool) -> None:
        values = self.q[state]
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        values[action] += self.alpha * (target - values[action])

    def reinforce_expert_action(self, state: tuple, action: int, reward: float = 0.05) -> None:
        self.q[state][action] = max(self.q[state][action], reward)

    def save(self, path: str | Path) -> None:
        payload = {
            "model": "tabular_q",
            "size": self.size,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "expert_name": self.expert_name,
            "q": dict(self.q),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> TabularQAgent:
        with Path(path).open("rb") as f:
            payload: dict[str, Any] = pickle.load(f)
        agent = cls(
            size=payload["size"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            expert_name=payload.get("expert_name", "hamiltonian"),
        )
        agent.q.update(payload["q"])
        return agent
