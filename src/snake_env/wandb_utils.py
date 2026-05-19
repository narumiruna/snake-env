from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import cast


class WandbRun(Protocol):
    def log(self, data: Mapping[str, object]) -> None: ...

    def save(self, glob_str: str) -> bool | list[str]: ...

    def finish(self) -> None: ...


def init_wandb(*, project: str, job_type: str, config: Mapping[str, object]) -> WandbRun:
    """Initialize a cloud W&B run for every experiment."""
    import wandb

    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_SILENT", "true")
    return cast(WandbRun, wandb.init(project=project, job_type=job_type, config=dict[str, Any](config)))
