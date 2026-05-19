from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from snake_env.agent import HamiltonianPolicy
from snake_env.env import SnakeEnv


def test_env_passes_gymnasium_checker() -> None:
    check_env(SnakeEnv(size=4), skip_render_check=True)


def test_hamiltonian_policy_wins_small_board() -> None:
    env = SnakeEnv(size=4)
    policy = HamiltonianPolicy(size=4)
    env.reset(seed=123)

    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(policy.act(env))

    assert info["won"] is True
    assert info["length"] == 16


def test_hamiltonian_policy_wins_20x20() -> None:
    env = SnakeEnv(size=20)
    policy = HamiltonianPolicy(size=20)
    env.reset(seed=123)

    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(policy.act(env))

    assert info["won"] is True
    assert info["length"] == 400
