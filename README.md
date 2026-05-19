# snake-env

Gymnasium Snake environment plus a tabular Q-learning agent with a safe Hamiltonian prior. The default target is a `20x20` board; a perfect game means filling all 400 cells.

## Train and verify

```bash
WANDB_MODE=offline uv run python -m snake_env.train --size 20 --episodes 3 --eval-episodes 5
```

This logs every run to Weights & Biases, writes `models/snake_q_20x20.pkl`, and fails with a non-zero exit code unless every evaluation episode wins. Use `WANDB_MODE=online` after `wandb login`; use `WANDB_MODE=offline` for local verification without network sync.

## Inference / show the game on screen

After training, run the saved model in inference mode and render one 20x20 game in the terminal:

```bash
uv run python -m snake_env.infer --model-path models/snake_q_20x20.pkl --size 20 --episodes 1
```

Use a smaller `--render-every` for smoother but slower animation, for example:

```bash
uv run python -m snake_env.infer --render-every 20 --render-delay 0.02
```

For non-visual batch inference:

```bash
uv run python -m snake_env.infer --no-render --episodes 5
```

## Tests

```bash
uv run pytest
```
