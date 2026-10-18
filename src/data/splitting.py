"""Patient-level split helpers. Historical ECG from previous stages is never replayed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("sample_id", "patient_id", "label", "file_path")


def patient_level_split(
    frame: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    patients = (
        frame[["patient_id", "label"]]
        .drop_duplicates("patient_id")
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    n = len(patients)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_ids = set(patients.iloc[:n_train]["patient_id"].astype(str))
    val_ids = set(patients.iloc[n_train : n_train + n_val]["patient_id"].astype(str))
    test_ids = set(patients.iloc[n_train + n_val :]["patient_id"].astype(str))
    pid = frame["patient_id"].astype(str)
    splits = {
        "train": frame.loc[pid.isin(train_ids)].reset_index(drop=True),
        "val": frame.loc[pid.isin(val_ids)].reset_index(drop=True),
        "test": frame.loc[pid.isin(test_ids)].reset_index(drop=True),
    }
    assigned = train_ids | val_ids | test_ids
    if len(assigned) != n or train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("patient-level split leaked or dropped patients")
    return splits
