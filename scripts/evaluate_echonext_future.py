"""Single post-hoc evaluation of EchoNext DFCL S6 on held-out 2022."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import make_loader
from src.evaluation.prediction import evaluate_loader, save_predictions
from src.utils import build_model, load_config, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "dfcl_hf_echonext.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = select_device(args.device)
    model = build_model(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    manifest_path = args.manifest or config.get("future_test_manifest")
    if not manifest_path:
        raise ValueError("pass --manifest or set future_test_manifest in the YAML")
    manifest = pd.read_csv(manifest_path)
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
    metrics, rows = evaluate_loader(model, loader, device, model_stage=6, test_stage=2022)
    metrics["sensitivity"] = metrics["recall"]
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["evaluation_type"] = "EchoNext 2022 independent future temporal validation"
    metrics["used_in_train_val_or_selection"] = False
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output_root"]) / "future_2022"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(rows, output_dir / "s6_future_2022_predictions.csv")
    cm = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["true_control_0", "true_case_1"],
        columns=["pred_control_0", "pred_case_1"],
    )
    cm.to_csv(output_dir / "s6_future_2022_confusion_matrix.csv")
    with (output_dir / "s6_future_2022_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
