# 🐍 snake-env — Efficient Gymnasium Snake RL Environment

A compact **Snake reinforcement learning project** built with **Gymnasium**, **tabular Q-learning**, **Hamiltonian-safe efficiency optimization**, and **Weights & Biases (W&B) cloud logging**. The default benchmark is a **20x20 Snake game** where a perfect policy wins by filling all **400 cells** while minimizing steps-to-win.

This repository is designed for people searching for a clean example of:

- 🧠 Reinforcement learning for Snake
- 🎮 Custom Gymnasium / Gym environment implementation
- 🐍 Python Snake game AI
- ⚡ Efficient Snake policy optimization
- 📈 W&B cloud experiment tracking for RL
- 🖥️ Terminal inference and visualization
- ✅ Fully verified 20x20 perfect-game Snake agent

## ✨ Features

- **Gymnasium-compatible Snake environment** (`SnakeEnv`)
- **20x20 board by default**
- **Efficiency-first performance metric**: win rate first, then fewer `avg_steps_to_win`
- **Efficient Hamiltonian model** with safe forward shortcuts
- **Tabular Q-learning agent** with the efficient Hamiltonian model as its safety prior
- **Perfect-game verification**: reaches `length=400`, `score=399`, `won=True`
- **Inference CLI** for loading and running trained models
- **Terminal rendering** for watching the Snake agent play
- **W&B cloud logging for every train, benchmark, and inference run**
- **Tests and static checks** with `pytest`, `ruff`, and `ty`

## 🚀 Quick Start

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Log in to Weights & Biases before running experiments:

```bash
wandb login
```

Train and verify the efficiency-optimized 20x20 Snake RL agent:

```bash
uv run python -m snake_env.train --size 20 --episodes 3 --eval-episodes 5
```

The training command:

1. trains the agent with the `efficient_hamiltonian` expert prior,
2. logs the experiment to W&B cloud,
3. saves the model to `models/snake_q_20x20.pkl`,
4. evaluates the trained model,
5. exits with a non-zero status if any evaluation game fails to win.

## 📊 Performance Metric: Most Efficient Winning Policy

The project now ranks models with a **win-first efficiency metric**:

```text
efficiency_score = win_rate * 1_000_000 + avg_score - avg_steps_to_win
```

This means:

1. **Win rate dominates** — a model must reliably win.
2. **Steps-to-win is the optimization target** among winning models.
3. **Score remains visible** for debugging and comparison.

Useful W&B metrics include:

- `train/efficiency`
- `inference/efficiency_score`
- `benchmark/hamiltonian/avg_steps_to_win`
- `benchmark/efficient_hamiltonian/avg_steps_to_win`
- `benchmark/step_improvement_pct`

## ⚡ Benchmark Efficient vs Baseline Model

Compare the baseline Hamiltonian policy against the efficient shortcut model:

```bash
uv run python -m snake_env.benchmark --size 20 --episodes 5
```

The benchmark uploads results to W&B cloud and fails unless:

- `efficient_hamiltonian` wins every episode, and
- `efficient_hamiltonian` uses fewer average steps than the baseline `hamiltonian` policy.

## 📊 Weights & Biases Cloud Logging

W&B cloud logging is always enabled for experiment CLIs. Use:

```bash
wandb login
```

Then run training, benchmarking, or inference normally:

```bash
uv run python -m snake_env.train
uv run python -m snake_env.benchmark
uv run python -m snake_env.infer --no-render
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
├── env.py              # Gymnasium Snake environment
├── metrics.py          # Win-first efficiency metrics
├── train.py            # Training and evaluation CLI
├── infer.py            # Inference CLI with terminal rendering
├── benchmark.py        # Baseline-vs-efficient model benchmark
├── runner.py           # Shared episode runner and renderer
├── wandb_utils.py      # Always-on W&B cloud initialization
├── agent.py            # Backward-compatible model exports
├── models/             # Algorithms and model policies
│   ├── hamiltonian.py  # Safe baseline Hamiltonian policy
│   ├── efficient.py    # Efficient Hamiltonian shortcut policy
│   └── q_learning.py   # Tabular Q-learning agent
└── py.typed            # Type marker

tests/
├── test_snake_env.py
└── test_infer.py
```

## 🏆 Benchmark Target

The default goal is a perfect and efficient **20x20 Snake game**:

| Metric | Target |
| --- | ---: |
| Board size | `20x20` |
| Final length | `400` |
| Score | `399` |
| Win flag | `won=True` |
| Primary objective | `win_rate=1.0` |
| Secondary objective | minimize `avg_steps_to_win` |

## 📌 Requirements

- Python `>=3.14`
- `uv`
- Gymnasium
- NumPy
- Loguru
- W&B

All Python dependencies are managed in `pyproject.toml` and locked in `uv.lock`.
