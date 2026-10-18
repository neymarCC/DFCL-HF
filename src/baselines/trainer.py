from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from ..continual.memory_teacher import build_memory_teacher
from ..continual.stage_manager import StageManager
from ..continual.supervised import (
    assert_finite,
    autocast_context,
    make_optimizer,
    make_scaler,
    make_scheduler,
    train_supervised,
)
from ..data.dataset import make_loader
from ..evaluation.continual_metrics import PerformanceMatrix
from ..evaluation.prediction import evaluate_loader, save_predictions
from ..losses.focal_loss import WeightedFocalLoss
from ..utils import build_model, class_weights
from .dualcp import DualCPModel
from .methods import (
    clone_trainable,
    dkd_loss,
    empirical_fisher,
    ewc_penalty,
    isolate_by_fisher,
    lwf_loss,
    lwht_loss,
)

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "recall",
    "specificity",
    "auroc",
    "auprc",
)


def wrap_model(model: nn.Module, method: str) -> nn.Module:
    if method == "dualcp":
        return DualCPModel(model)
    return model


class BaselineCLTrainer:
    def __init__(self, config: dict[str, Any], device: torch.device, smoke: bool = False) -> None:
        self.config = config
        self.device = device
        self.smoke = smoke
        self.method = str(config["method"]).lower()
        self.manager = StageManager(config["manifest_root"], int(config["n_stages"]))
        self.output_root = Path(config["output_root"])
        if smoke:
            self.output_root = self.output_root / "smoke"
        self.n_stages = 2 if smoke else int(config["n_stages"])
        self.smoke_n = 48 if smoke else None

    def _prepare_manifest(self, stage: int, split: str) -> pd.DataFrame:
        frame = self.manager.load(stage, split)
        if self.smoke:
            frame = self.manager.subsample(frame, self.smoke_n, int(self.config["seed"]) + stage)
        return frame

    def _loader(self, manifest: pd.DataFrame, shuffle: bool, seed_offset: int):
        return make_loader(
            manifest,
            batch_size=int(self.config["batch_size"]),
            shuffle=shuffle,
            seed=int(self.config["seed"]) + seed_offset,
            num_workers=int(self.config["num_workers"]),
            target_length=int(self.config["target_length"]),
            normalize=bool(self.config["normalize"]),
            device=self.device,
        )

    def _criterion(self, train_manifest: pd.DataFrame) -> WeightedFocalLoss:
        return WeightedFocalLoss(
            gamma=float(self.config["focal_gamma"]),
            class_weight=class_weights(train_manifest, self.device, self.config["focal_alpha"]),
            eps=float(self.config["focal_eps"]),
        )

    def _evaluate_all(self, model, model_stage: int, matrices, test_loaders) -> None:
        prediction_dir = self.output_root / "predictions"
        for test_stage, loader in test_loaders.items():
            metrics, rows = evaluate_loader(
                model, loader, self.device, model_stage=model_stage, test_stage=test_stage
            )
            save_predictions(
                rows,
                prediction_dir / f"model_stage_{model_stage:02d}_test_stage_{test_stage:02d}.csv",
            )
            for name in METRIC_NAMES:
                matrices[name].add(model_stage, test_stage, float(metrics[name]))
        for matrix in matrices.values():
            matrix.save(self.output_root / "matrices")

    def _evaluate_future(self, model, model_stage: int) -> None:
        future_path = self.config.get("future_test_manifest")
        if not future_path:
            return
        path = Path(future_path)
        if not path.exists():
            print(f"[warn] future test missing: {path}")
            return
        manifest = pd.read_csv(path)
        if self.smoke:
            manifest = manifest.head(self.smoke_n)
        loader = self._loader(manifest, False, 90_000)
        metrics, rows = evaluate_loader(model, loader, self.device, model_stage=model_stage, test_stage=-1)
        out = self.output_root / "future_2022"
        save_predictions(rows, out / "predictions.csv")
        with (out / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump({k: float(v) for k, v in metrics.items()}, file, indent=2)
        print(
            f"[future 2022] AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
            f"Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
            f"Sens={metrics['recall']:.4f} Spec={metrics['specificity']:.4f}"
        )

    def _train_regularized(
        self,
        model: nn.Module,
        teacher: nn.Module | None,
        train_loader,
        val_loader,
        criterion,
        ewc_states: list[dict[str, Any]],
        checkpoint_path: Path,
        log_path: Path,
        stage_name: str,
        learning_rate: float,
    ) -> None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = make_optimizer_params(
            trainable, learning_rate, float(self.config["weight_decay"]), str(self.config["optimizer"])
        )
        scheduler = make_scheduler(
            optimizer,
            str(self.config["scheduler"]),
            float(self.config["scheduler_factor"]),
            int(self.config["scheduler_patience"]),
        )
        scaler = make_scaler(self.device, bool(self.config["use_amp"]))
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_auc = -math.inf
        bad_epochs = 0
        rows: list[dict[str, Any]] = []
        epochs = int(self.config["epochs_student"])
        use_amp = bool(self.config["use_amp"])
        clip = float(self.config["gradient_clip"])
        if teacher is not None:
            teacher.eval()

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(self.device, use_amp):
                    features, logits, _ = model(x)
                    loss = criterion(logits, y)
                    if self.method == "ewc":
                        loss = loss + ewc_penalty(model, ewc_states, float(self.config["ewc_lambda"]))
                    elif self.method in {"lwf", "dkd", "lwht"} and teacher is not None:
                        with torch.no_grad():
                            t_feat, t_logits, _ = teacher(x)
                        if self.method == "lwf":
                            loss = loss + lwf_loss(
                                logits.float(),
                                t_logits.float(),
                                float(self.config["kd_temperature"]),
                                float(self.config["lwf_alpha"]),
                            )
                        elif self.method == "dkd":
                            loss = loss + dkd_loss(
                                logits.float(),
                                t_logits.float(),
                                y,
                                float(self.config["dkd_alpha"]),
                                float(self.config["dkd_beta"]),
                                float(self.config["kd_temperature"]),
                            )
                        else:
                            loss = loss + lwht_loss(
                                logits.float(),
                                t_logits.float(),
                                features.float(),
                                t_feat.float(),
                                float(self.config["kd_temperature"]),
                                float(self.config["lwht_alpha"]),
                                float(self.config["lwht_gamma"]),
                            )
                assert_finite("baseline_loss", loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, clip)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))

            val_metrics, _ = evaluate_loader(model, val_loader, self.device, -1, -1)
            val_auc = float(val_metrics["auroc"])
            score = val_auc if math.isfinite(val_auc) else -math.inf
            row = {
                "epoch": epoch,
                "stage_name": stage_name,
                "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
                "val_auroc": val_auc,
                "val_auprc": float(val_metrics["auprc"]),
                "val_f1": float(val_metrics["f1"]),
            }
            rows.append(row)
            print(
                f"[{stage_name}] epoch={epoch:03d} "
                f"train={row['train_loss']:.4f} val_auc={val_auc:.4f}"
            )
            if scheduler is not None:
                scheduler.step(score)
            if score > best_auc + 1e-6:
                best_auc = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model": best_state, "val_auroc": best_auc, "method": self.method}, checkpoint_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.config["early_stopping_patience"]):
                    break

        pd.DataFrame(rows).to_csv(log_path, index=False)
        model.load_state_dict(best_state)

    def fit(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_root / "logs").mkdir(parents=True, exist_ok=True)
        with (self.output_root / "run_config.json").open("w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=2)
        with (self.output_root / "METHOD_ASSUMPTIONS.md").open("w", encoding="utf-8") as file:
            file.write(ASSUMPTIONS)

        test_manifests = {stage: self._prepare_manifest(stage, "test") for stage in range(self.n_stages)}
        test_loaders = {
            stage: self._loader(manifest, False, 10_000 + stage)
            for stage, manifest in test_manifests.items()
        }
        model = wrap_model(build_model(self.config), self.method).to(self.device)
        untrained = wrap_model(build_model(self.config), self.method).to(self.device).eval()
        random_baseline = {
            stage: float(evaluate_loader(untrained, loader, self.device, -1, stage)[0]["auroc"])
            for stage, loader in test_loaders.items()
        }
        matrices = {
            name: PerformanceMatrix(name, random_baseline if name == "auroc" else None)
            for name in METRIC_NAMES
        }
        ewc_states: list[dict[str, Any]] = []
        isolated: list[str] = []

        for stage in range(self.n_stages):
            print(f"===== {self.method} Stage {stage} =====")
            train_m = self._prepare_manifest(stage, "train")
            val_m = self._prepare_manifest(stage, "val")
            criterion = self._criterion(train_m)
            ckpt = self.output_root / "checkpoints" / f"stage_{stage}_best.pt"
            log_path = self.output_root / "logs" / f"stage_{stage}_training_log.csv"
            lr = float(self.config["learning_rate_stage0"] if stage == 0 else self.config["learning_rate_student"])

            if stage == 0:
                train_supervised(
                    model=model,
                    train_loader=self._loader(train_m, True, 1),
                    val_loader=self._loader(val_m, False, 2),
                    criterion=criterion,
                    device=self.device,
                    epochs=int(self.config["epochs_stage0"]),
                    learning_rate=lr,
                    weight_decay=float(self.config["weight_decay"]),
                    patience=int(self.config["early_stopping_patience"]),
                    gradient_clip=float(self.config["gradient_clip"]),
                    use_amp=bool(self.config["use_amp"]),
                    optimizer_name=str(self.config["optimizer"]),
                    scheduler_name=str(self.config["scheduler"]),
                    scheduler_factor=float(self.config["scheduler_factor"]),
                    scheduler_patience=int(self.config["scheduler_patience"]),
                    checkpoint_path=ckpt,
                    log_path=log_path,
                    stage_name=f"{self.method}_stage0",
                )
            else:
                teacher = None
                if self.method in {"lwf", "dkd", "lwht"}:
                    teacher = build_memory_teacher(model)
                    model = clone_trainable(model).to(self.device)
                elif self.method == "dualcp" and isinstance(model, DualCPModel):
                    model.freeze_extractor()
                    model.add_expert()
                    model.to(self.device)
                elif self.method in {"naive", "ewc", "api"}:
                    for parameter in model.parameters():
                        parameter.requires_grad_(True)
                    if self.method == "api":
                        reapply_isolation(model, isolated)
                self._train_regularized(
                    model=model,
                    teacher=teacher,
                    train_loader=self._loader(train_m, True, 100 + stage),
                    val_loader=self._loader(val_m, False, 200 + stage),
                    criterion=criterion,
                    ewc_states=ewc_states,
                    checkpoint_path=ckpt,
                    log_path=log_path,
                    stage_name=f"{self.method}_stage{stage}",
                    learning_rate=lr,
                )

            best_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            with (self.output_root / "logs" / f"stage_{stage}_best.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "stage": stage,
                        "method": self.method,
                        "best_val_auroc": float(best_meta.get("val_auroc", float("nan"))),
                        "checkpoint": str(ckpt),
                    },
                    file,
                    indent=2,
                )
            self._evaluate_all(model, stage, matrices, test_loaders)

            if self.method == "ewc":
                fisher_loader = self._loader(train_m, False, 500 + stage)
                fisher = empirical_fisher(model, fisher_loader, self.device, bool(self.config["use_amp"]))
                ewc_states.append(
                    {
                        "fisher": fisher,
                        "params": {
                            name: parameter.detach().cpu().clone()
                            for name, parameter in model.named_parameters()
                            if name in fisher
                        },
                    }
                )
            if self.method == "api":
                fisher_loader = self._loader(train_m, False, 600 + stage)
                fisher = empirical_fisher(model, fisher_loader, self.device, bool(self.config["use_amp"]))
                newly = isolate_by_fisher(model, fisher, float(self.config["api_freeze_fraction"]))
                isolated.extend(newly)
                with (self.output_root / "logs" / f"stage_{stage}_api_frozen.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump({"frozen": isolated}, file, indent=2)

        final_stage = self.n_stages - 1
        torch.save(
            {"model": model.state_dict(), "stage": final_stage, "method": self.method},
            self.output_root / "checkpoints" / f"final_stage{final_stage}.pt",
        )
        self._evaluate_future(model, final_stage)
        print(f"[done] method={self.method} output={self.output_root}")


def make_optimizer_params(params, lr: float, weight_decay: float, name: str):
    if name.lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def isolated_params(model: nn.Module, names: list[str]) -> set[torch.nn.Parameter]:
    lookup = dict(model.named_parameters())
    return {lookup[name] for name in names if name in lookup}


def reapply_isolation(model: nn.Module, names: list[str]) -> None:
    lookup = dict(model.named_parameters())
    for name in names:
        if name in lookup:
            lookup[name].requires_grad_(False)


ASSUMPTIONS = """# Baseline method assumptions

Official paper clones used when the algorithm is well-specified for this CNN-BiLSTM backbone:
- Naive Fine-tuning: sequential L_sup only, initialize from previous best checkpoint.
- EWC: diagonal empirical Fisher on the just-finished stage train set (not historical ECG replay).
- LwF: frozen previous model, temperature-scaled KL on current data.
- DKD: Zhao et al. CVPR 2022 TCKD+NCKD on current data.

Idea-based adaptations (no official in-repo implementation; not claimed as paper-faithful clones):
- API (Adaptive Parameter Isolation): after each stage, freeze the top `api_freeze_fraction`
  of still-trainable parameters by mean diagonal Fisher. Last classifier layer stays trainable.
  Inspired by parameter-isolation CL (PackNet/HAT-style), not a named API paper port.
- LwHT (Learning with Hierarchical Teachers): frozen previous student as teacher;
  logit KD plus feature MSE (FitNet hint). Hierarchical = output + representation.
  Not DFCL-HF (no CAGD entropy gating, no dual Memory/Updated teachers, no PSR).
- DualCP (Dual Complementary Prompts): DualPrompt-style G-prompt + E-prompts with
  cosine key matching, added to pooled CNN-BiLSTM features. Extractor frozen after Stage 0.
  Not ViT prefix-tuning DualPrompt; inference is task-id-free via key matching.

Shared protocol: same manifests, backbone, preprocessing, batch size, Adam lr, epochs,
early stopping, val-AUROC selection. No test-set tuning. No raw historical ECG replay.
"""
