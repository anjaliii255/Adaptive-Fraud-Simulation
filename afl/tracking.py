"""Experiment tracking — MLflow when it is reachable, in-memory always.

`history` is the source of truth for the convergence curve, so the loop never depends on a
tracking server being up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    """What the loop needs from a tracker: log metrics, log params, keep history."""

    history: list[dict[str, Any]]

    def log(self, **kwargs: Any) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_artifact(self, path: str | Path) -> None: ...


class InMemoryTracker:
    """Zero-dependency tracker. What tests and the smoke loop use."""

    def __init__(self, run_name: str = "local") -> None:
        self.run_name = run_name
        self.history: list[dict[str, Any]] = []
        self.params: dict[str, Any] = {}
        self.artifacts: list[str] = []

    def log(self, **kwargs: Any) -> None:
        self.history.append(dict(kwargs))

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_artifact(self, path: str | Path) -> None:
        self.artifacts.append(str(path))

    def dump(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"run_name": self.run_name, "params": self.params, "history": self.history},
                indent=2,
                default=str,
            )
        )
        return path


class MLflowTracker(InMemoryTracker):
    """Mirrors every call to MLflow while keeping the local history intact."""

    def __init__(self, run_name: str = "local", experiment: str | None = None) -> None:
        super().__init__(run_name)
        import mlflow

        self._mlflow = mlflow
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(
            experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "adaptive-fraud-lab")
        )
        self._run = mlflow.start_run(run_name=run_name)

    def log(self, **kwargs: Any) -> None:
        super().log(**kwargs)
        step = kwargs.get("round")
        numeric = {k: v for k, v in kwargs.items() if isinstance(v, int | float) and k != "round"}
        self._mlflow.log_metrics(numeric, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        super().log_params(params)
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_artifact(self, path: str | Path) -> None:
        super().log_artifact(path)
        self._mlflow.log_artifact(str(path))

    def finish(self) -> None:
        self._mlflow.end_run()


def get_tracker(
    kind: str = "auto", run_name: str = "local", experiment: str | None = None
) -> Tracker:
    """kind: "memory" | "mlflow" | "auto" (mlflow if importable + configured, else memory)."""
    if kind == "memory":
        return InMemoryTracker(run_name)
    try:
        return MLflowTracker(run_name, experiment)
    except Exception:
        if kind == "mlflow":
            raise
        return InMemoryTracker(run_name)
