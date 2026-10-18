#!/usr/bin/env python
"""EchoNext E0-E6 DFCL-HF training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual.dfcl_hf_trainer import DFCLHFTrainer
from src.utils import load_config, select_device, set_seed


def apply_overrides(config: dict, pairs: list[str]) -> dict:
    import yaml

    config = dict(config)
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid --set {item}; expected KEY=VALUE")
        key, raw = item.split("=", 1)
        config[key] = yaml.safe_load(raw)
    return config


def apply_smoke(config: dict) -> dict:
    config = dict(config)
    config["epochs_stage0"] = 1
    config["epochs_updated_teacher"] = 1
    config["epochs_student"] = 1
    config["early_stopping_patience"] = 1
    config["batch_size"] = min(int(config["batch_size"]), 16)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "dfcl_hf_echonext.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    if args.output_root:
        config["output_root"] = args.output_root
    if args.smoke:
        config = apply_smoke(config)
    if args.device:
        config["device"] = args.device
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    DFCLHFTrainer(config, device, smoke=args.smoke).fit()


if __name__ == "__main__":
    main()
