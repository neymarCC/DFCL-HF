from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual.dfcl_hf_trainer import DFCLHFTrainer
from src.utils import load_config, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "dfcl_hf.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config KEY=VALUE (repeatable). VALUE is YAML-parsed.",
    )
    return parser.parse_args()


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
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    if args.output_root:
        config["output_root"] = args.output_root
    if args.smoke:
        config = apply_smoke(config)
    if args.device:
        config["device"] = args.device
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    print(f"[device] {device}")
    print(f"[smoke] {args.smoke}")
    trainer = DFCLHFTrainer(config, device, smoke=args.smoke)
    trainer.fit()


if __name__ == "__main__":
    main()
