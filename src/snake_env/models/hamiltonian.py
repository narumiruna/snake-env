from __future__ import annotations

from snake_env.env import Action
from snake_env.env import Pos
from snake_env.env import SnakeEnv


class SnakePolicy:
    """Minimal policy protocol for deterministic Snake agents."""

    size: int

    def act(self, env: SnakeEnv) -> int:
        raise NotImplementedError


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


class HamiltonianPolicy(SnakePolicy):
    """Safe baseline policy: following this cycle eventually fills the board."""

    def __init__(self, size: int) -> None:
        cycle = hamiltonian_cycle(size)
        self.size = size
        self.index = {pos: i for i, pos in enumerate(cycle)}
        self.next_pos = {pos: cycle[(i + 1) % len(cycle)] for i, pos in enumerate(cycle)}

    def act(self, env: SnakeEnv) -> int:
        return self._action_toward(env.head, self.next_pos[env.head])

    def cycle_distance(self, start: Pos, end: Pos) -> int:
        return (self.index[end] - self.index[start]) % (self.size * self.size)

    @staticmethod
    def _action_toward(start: Pos, target: Pos) -> int:
        dx = target.x - start.x
        dy = target.y - start.y
        if dx == 1:
            return int(Action.RIGHT)
        if dx == -1:
            return int(Action.LEFT)
        if dy == 1:
            return int(Action.DOWN)
        if dy == -1:
            return int(Action.UP)
        raise RuntimeError(f"non-adjacent Hamiltonian move from {start} to {target}")
