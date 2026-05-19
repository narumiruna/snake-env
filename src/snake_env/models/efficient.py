from __future__ import annotations

from snake_env.env import OPPOSITE
from snake_env.env import Action
from snake_env.env import SnakeEnv
from snake_env.models.hamiltonian import HamiltonianPolicy


class EfficientHamiltonianPolicy(HamiltonianPolicy):
    """Hamiltonian-safe shortcut policy optimized for fewer steps.

    The model keeps the Hamiltonian cycle as a safety invariant, but it can jump
    forward along the cycle whenever that move keeps the head behind the tail.
    Among safe moves it greedily minimizes remaining Hamiltonian distance to the
    apple. This preserves perfect-game reliability while reducing steps versus
    the plain cycle-following baseline.
    """

    def act(self, env: SnakeEnv) -> int:
        if env.apple is None:
            return super().act(env)

        candidates = self._safe_forward_actions(env)
        if not candidates:
            return super().act(env)

        return min(candidates)[2]

    def _safe_forward_actions(self, env: SnakeEnv) -> list[tuple[int, int, int]]:
        assert env.apple is not None
        board_cells = self.size * self.size
        head = env.head
        tail = env.snake[-1]
        tail_distance = self.cycle_distance(head, tail)
        candidates: list[tuple[int, int, int]] = []

        for action in Action:
            if len(env.snake) > 1 and action == OPPOSITE[env.direction]:
                continue

            next_pos = head.move(action)
            if env._out_of_bounds(next_pos):
                continue

            will_grow = next_pos == env.apple
            occupied = env.snake if will_grow else env.snake[:-1]
            if next_pos in occupied:
                continue

            forward_distance = self.cycle_distance(head, next_pos)
            if forward_distance <= 0:
                continue

            max_forward_distance = tail_distance if will_grow else tail_distance + 1
            if len(env.snake) < board_cells and forward_distance >= max_forward_distance:
                continue

            apple_distance = self.cycle_distance(next_pos, env.apple)
            candidates.append((apple_distance, -forward_distance, int(action)))

        return candidates
