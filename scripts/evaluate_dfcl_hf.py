from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual.stage_manager import StageManager
from src.data.dataset import make_loader
from src.evaluation.prediction import evaluate_loader, save_predictions
from src.utils import build_model, load_config, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "dfcl_hf.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-stage", type=int, required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    manager = StageManager(config["manifest_root"], int(config["n_stages"]))
    model = build_model(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output_root"]) / "eval"
    for test_stage in range(int(config["n_stages"])):
        manifest = manager.load(test_stage, "test")
        loader = make_loader(
            manifest,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            seed=int(config["seed"]),
            num_workers=int(config["num_workers"]),
            target_length=int(config["target_length"]),
            normalize=bool(config["normalize"]),
            device=device,
        )
        metrics, rows = evaluate_loader(model, loader, device, args.model_stage, test_stage)
        save_predictions(rows, output_dir / f"model_stage_{args.model_stage:02d}_test_stage_{test_stage:02d}.csv")
        print(f"T{test_stage}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
