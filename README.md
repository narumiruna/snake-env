# 🐍 snake-env — Gymnasium Snake RL Environment

A compact **Snake reinforcement learning project** built with **Gymnasium**, **tabular Q-learning**, and **Weights & Biases (W&B)** logging. The default benchmark is a **20x20 Snake game** where a perfect policy wins by filling all **400 cells**.

This repository is designed for people searching for a clean example of:

- 🧠 Reinforcement learning for Snake
- 🎮 Custom Gymnasium / Gym environment implementation
- 🐍 Python Snake game AI
- 📈 W&B experiment tracking for RL
- 🖥️ Terminal inference and visualization
- ✅ Fully verified 20x20 perfect-game Snake agent

## ✨ Features

- **Gymnasium-compatible Snake environment** (`SnakeEnv`)
- **20x20 board by default**
- **Tabular Q-learning agent** with a safe Hamiltonian prior
- **Perfect-game verification**: reaches `length=400`, `score=399`, `won=True`
- **Inference CLI** for loading and running trained models
- **Terminal rendering** for watching the Snake agent play
- **Weights & Biases logging enabled by default**
- **Tests and static checks** with `pytest`, `ruff`, and `ty`

## 🚀 Quick Start

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Train and verify the 20x20 Snake RL agent:

```bash
uv run python -m snake_env.train --size 20 --episodes 3 --eval-episodes 5
```

The training command:

1. trains the agent,
2. saves the model to `models/snake_q_20x20.pkl`,
3. evaluates the trained model,
4. exits with a non-zero status if any evaluation game fails to win.

## 📊 Weights & Biases Logging

W&B is enabled by default. If you are not logged in, the code safely falls back to local offline logging.

For offline/local experiment tracking:

```bash
WANDB_MODE=offline uv run python -m snake_env.train --size 20 --episodes 3 --eval-episodes 5
```

For online W&B sync:

```bash
wandb login
WANDB_MODE=online uv run python -m snake_env.train --size 20 --episodes 3 --eval-episodes 5
```

To disable W&B for a run:

```bash
uv run python -m snake_env.train --no-wandb
```

## 🎮 Inference: Watch the Snake Agent Play

After training, run inference with the saved model and render the game in your terminal:

```bash
uv run python -m snake_env.infer --model-path models/snake_q_20x20.pkl --size 20 --episodes 1
```

For smoother animation, decrease `--render-every`:

```bash
uv run python -m snake_env.infer --render-every 20 --render-delay 0.02
```

For faster non-visual batch inference:

```bash
uv run python -m snake_env.infer --no-render --episodes 5
```

A successful 20x20 inference run ends with:

```text
score=399 length=400 won=True
```

## 🧪 Quality Checks

Run the full local verification suite:

```bash
uv run ruff check .
uv run ty check .
uv run pytest
```

Run only tests:

```bash
uv run pytest
```

## 📁 Project Structure

```text
src/snake_env/
├── env.py       # Gymnasium Snake environment
├── agent.py     # Hamiltonian policy + tabular Q-learning agent
├── train.py     # Training and evaluation CLI
├── infer.py     # Inference CLI with terminal rendering
├── runner.py    # Shared episode runner and renderer
└── py.typed     # Type marker

tests/
├── test_snake_env.py
└── test_infer.py
```

## 🏆 Benchmark Target

The default goal is a perfect **20x20 Snake game**:

| Metric | Perfect value |
| --- | ---: |
| Board size | `20x20` |
| Final length | `400` |
| Score | `399` |
| Win flag | `won=True` |

## 📌 Requirements

- Python `>=3.14`
- `uv`
- Gymnasium
- NumPy
- Loguru
- W&B

All Python dependencies are managed in `pyproject.toml` and locked in `uv.lock`.
