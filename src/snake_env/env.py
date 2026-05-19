from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Cell(IntEnum):
    EMPTY = 0
    HEAD = 1
    BODY = 2
    APPLE = 3


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass(frozen=True, slots=True)
class Pos:
    x: int
    y: int

    def move(self, action: Action) -> Pos:
        dx, dy = ACTION_DELTAS[action]
        return Pos(self.x + dx, self.y + dy)


ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (0, -1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
}
OPPOSITE: dict[Action, Action] = {
    Action.UP: Action.DOWN,
    Action.RIGHT: Action.LEFT,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
}


class SnakeEnv(gym.Env[dict[str, Any], int]):
    """Gymnasium Snake environment.

    The game is won when the snake fills the whole board. Invalid moves end the
    episode. Rewards are intentionally sparse enough for meaningful RL training
    but include a small living penalty to discourage loops.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 8}  # noqa: RUF012

    def __init__(
        self,
        size: int = 4,
        *,
        render_mode: str | None = None,
        max_steps_without_apple: int | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")
        if size % 2 != 0:
            raise ValueError("size must be even so the built-in winning policy exists")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.size = size
        self.render_mode = render_mode
        self.max_steps_without_apple = max_steps_without_apple or size * size * 2
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0, 3, shape=(size, size), dtype=np.int8),
                "direction": spaces.Discrete(4),
            }
        )

        self._rng = random.Random()
        self.snake: list[Pos] = []
        self.direction = Action.RIGHT
        self.apple: Pos | None = None
        self.steps = 0
        self.steps_since_apple = 0

    @property
    def head(self) -> Pos:
        return self.snake[0]

    @property
    def score(self) -> int:
        return len(self.snake) - 1

    @property
    def won(self) -> bool:
        return len(self.snake) == self.size * self.size

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng.seed(seed)
        start = Pos(0, 0)
        self.snake = [start]
        self.direction = Action.RIGHT
        self.steps = 0
        self.steps_since_apple = 0
        self.apple = None
        self._spawn_apple()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        selected = Action(int(action))
        if len(self.snake) > 1 and selected == OPPOSITE[self.direction]:
            selected = self.direction
        self.direction = selected

        self.steps += 1
        self.steps_since_apple += 1
        new_head = self.head.move(self.direction)

        if self._out_of_bounds(new_head):
            return self._observation(), -1.0, True, False, self._info(reason="wall")

        will_grow = new_head == self.apple
        body_to_check = self.snake if will_grow else self.snake[:-1]
        if new_head in body_to_check:
            return self._observation(), -1.0, True, False, self._info(reason="self")

        self.snake.insert(0, new_head)
        if will_grow:
            self.steps_since_apple = 0
            if self.won:
                self.apple = None
                return self._observation(), 10.0, True, False, self._info(reason="won")
            self._spawn_apple()
            reward = 1.0
        else:
            self.snake.pop()
            reward = -0.01

        if self.steps_since_apple >= self.max_steps_without_apple:
            return self._observation(), -0.5, False, True, self._info(reason="timeout")

        return self._observation(), reward, False, False, self._info()

    def render(self) -> str:
        chars = {Cell.EMPTY: "_", Cell.HEAD: "H", Cell.BODY: "S", Cell.APPLE: "A"}
        board = self._board()
        return "\n".join("".join(chars[Cell(int(cell))] for cell in row) for row in board)

    def state_key(self) -> tuple[tuple[tuple[int, int], ...], tuple[int, int] | None, int]:
        apple = None if self.apple is None else (self.apple.x, self.apple.y)
        return tuple((p.x, p.y) for p in self.snake), apple, int(self.direction)

    def _spawn_apple(self) -> None:
        occupied = set(self.snake)
        choices = [Pos(x, y) for y in range(self.size) for x in range(self.size) if Pos(x, y) not in occupied]
        self.apple = None if not choices else self._rng.choice(choices)

    def _observation(self) -> dict[str, Any]:
        return {"board": self._board(), "direction": int(self.direction)}

    def _board(self) -> np.ndarray:
        board = np.zeros((self.size, self.size), dtype=np.int8)
        if self.apple is not None:
            board[self.apple.y, self.apple.x] = int(Cell.APPLE)
        for p in self.snake[1:]:
            board[p.y, p.x] = int(Cell.BODY)
        board[self.head.y, self.head.x] = int(Cell.HEAD)
        return board

    def _info(self, *, reason: str | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {"score": self.score, "length": len(self.snake), "won": self.won, "steps": self.steps}
        if reason is not None:
            info["reason"] = reason
        return info

    def _out_of_bounds(self, pos: Pos) -> bool:
        return not (0 <= pos.x < self.size and 0 <= pos.y < self.size)
