from __future__ import annotations

from unittest.mock import Mock

from snake_env.infer import InferenceConfig
from snake_env.infer import run_inference
from snake_env.models import TabularQAgent


def test_run_inference_loads_model_and_wins(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "agent.pkl"
    TabularQAgent(size=4).save(model_path)
    run = Mock()
    monkeypatch.setattr("snake_env.infer.init_wandb", Mock(return_value=run))

    results = run_inference(
        InferenceConfig(
            model_path=str(model_path),
            size=4,
            episodes=2,
            seed=123,
            render=False,
        )
    )

    assert [result["won"] for result in results] == [True, True]
    assert [result["length"] for result in results] == [16, 16]
    assert run.log.called
    run.finish.assert_called_once()
