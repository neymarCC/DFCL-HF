"""Copy existing EchoNext E0–E6 70/10/20 manifests into stage_N names for DFCL-HF."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "echonext_raw_splits"
DST = ROOT / "data" / "echonext_manifests"
STAGES = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]


def summarize(frame: pd.DataFrame, stage: str, split: str) -> dict:
    return {
        "stage": stage,
        "split": split,
        "n_total": int(len(frame)),
        "n_case": int((frame["label"] == 1).sum()),
        "n_control": int((frame["label"] == 0).sum()),
        "n_patients": int(frame["patient_id"].nunique()),
        "years": ",".join(str(int(y)) for y in sorted(frame["year"].dropna().unique())),
    }


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, name in enumerate(STAGES):
        for split in ("train", "val", "test"):
            src = SRC / f"{name}_{split}.csv"
            dst = DST / f"stage_{idx}_{split}.csv"
            frame = pd.read_csv(src)
            frame.to_csv(dst, index=False)
            rows.append(summarize(frame, f"{name}=stage_{idx}", split))
    future = pd.read_csv(SRC / "E_future_2022_test.csv")
    future.to_csv(DST / "future_2022_test.csv", index=False)
    rows.append(summarize(future, "2022_future", "heldout_test"))
    summary = pd.DataFrame(rows)
    summary.to_csv(DST / "split_counts.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[done] {DST}")


if __name__ == "__main__":
    main()
