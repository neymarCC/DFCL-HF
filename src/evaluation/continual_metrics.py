from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class PerformanceMatrix:
    def __init__(self, metric_name: str, random_baseline: dict[int, float] | None = None) -> None:
        self.metric_name = metric_name
        self.random_baseline = random_baseline or {}
        self._values: dict[tuple[int, int], float] = {}

    def add(self, model_stage: int, test_stage: int, value: float) -> None:
        self._values[(int(model_stage), int(test_stage))] = float(value)

    def get(self, model_stage: int, test_stage: int) -> float:
        return self._values.get((int(model_stage), int(test_stage)), float("nan"))

    @property
    def model_stages(self) -> list[int]:
        return sorted({model for model, _ in self._values})

    @property
    def test_stages(self) -> list[int]:
        return sorted({test for _, test in self._values})

    def _mean(self, values: list[float]) -> float:
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    def average_performance(self, model_stage: int) -> float:
        return self._mean(
            [self.get(model_stage, test_stage) for test_stage in self.test_stages if test_stage <= model_stage]
        )

    def backward_transfer(self, model_stage: int) -> float:
        if model_stage <= 0:
            return float("nan")
        return self._mean(
            [
                self.get(model_stage, test_stage) - self.get(test_stage, test_stage)
                for test_stage in self.test_stages
                if test_stage < model_stage and np.isfinite(self.get(test_stage, test_stage))
            ]
        )

    def forward_transfer(self, model_stage: int) -> float:
        if model_stage <= 0 or not self.random_baseline:
            return float("nan")
        return self._mean(
            [
                self.get(test_stage - 1, test_stage) - self.random_baseline[test_stage]
                for test_stage in self.test_stages
                if 1 <= test_stage <= model_stage
                and test_stage in self.random_baseline
                and np.isfinite(self.get(test_stage - 1, test_stage))
            ]
        )

    def forward_transfer_init(self, model_stage: int) -> float:
        """FWT relative to the stage-0 model: mean_j [R(j-1,j) - R(0,j)]."""
        if model_stage <= 0:
            return float("nan")
        return self._mean(
            [
                self.get(test_stage - 1, test_stage) - self.get(0, test_stage)
                for test_stage in self.test_stages
                if 1 <= test_stage <= model_stage
                and np.isfinite(self.get(test_stage - 1, test_stage))
                and np.isfinite(self.get(0, test_stage))
            ]
        )

    def forgetting(self, model_stage: int) -> float:
        if model_stage <= 0:
            return float("nan")
        per_stage: list[float] = []
        for test_stage in self.test_stages:
            if test_stage >= model_stage:
                continue
            previous = [
                self.get(previous_stage, test_stage)
                for previous_stage in self.model_stages
                if test_stage <= previous_stage < model_stage
                and np.isfinite(self.get(previous_stage, test_stage))
            ]
            current = self.get(model_stage, test_stage)
            if previous and np.isfinite(current):
                per_stage.append(max(previous) - current)
        return self._mean(per_stage)

    def summary(self, model_stage: int) -> dict[str, Any]:
        return {
            "model_stage": int(model_stage),
            "metric": self.metric_name,
            "average_performance": self.average_performance(model_stage),
            "backward_transfer": self.backward_transfer(model_stage),
            "forward_transfer": self.forward_transfer(model_stage),
            "forward_transfer_init": self.forward_transfer_init(model_stage),
            "forgetting": self.forgetting(model_stage),
        }

    def to_wide(self) -> pd.DataFrame:
        rows = []
        for model_stage in self.model_stages:
            row = {"model_stage": model_stage}
            for test_stage in self.test_stages:
                row[f"T{test_stage}"] = self.get(model_stage, test_stage)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.to_wide().to_csv(directory / f"{self.metric_name}_matrix.csv", index=False)
        pd.DataFrame([self.summary(stage) for stage in self.model_stages]).to_csv(
            directory / f"{self.metric_name}_continual_metrics.csv",
            index=False,
        )
