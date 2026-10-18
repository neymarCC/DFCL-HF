from __future__ import annotations

from pathlib import Path

import pandas as pd


class StageManager:
    def __init__(self, manifest_root: str | Path, n_stages: int) -> None:
        self.manifest_root = Path(manifest_root)
        self.n_stages = n_stages

    def load(self, stage: int, split: str) -> pd.DataFrame:
        path = self.manifest_root / f"stage_{stage}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def subsample(self, frame: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
        parts = []
        for label in (0, 1):
            subset = frame.loc[frame["label"] == label]
            take = min(n_per_class, len(subset))
            parts.append(subset.sample(n=take, random_state=seed))
        return pd.concat(parts, ignore_index=True)
