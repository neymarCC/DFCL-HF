from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..data.dataset import make_loader
from ..evaluation.continual_metrics import PerformanceMatrix
from ..evaluation.prediction import evaluate_loader, save_predictions
from ..losses.focal_loss import WeightedFocalLoss
from ..utils import build_model, class_weights
from .initial_model import load_initial_model, save_initial_model, tensors_equal
from .memory_teacher import build_memory_teacher
from .stage_manager import StageManager
from .student_trainer import train_student
from .supervised import train_supervised
from .updated_teacher import train_updated_teacher


METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "recall",
    "specificity",
    "auroc",
    "auprc",
)


class DFCLHFTrainer:
    def __init__(self, config: dict[str, Any], device: torch.device, smoke: bool = False) -> None:
        if abs(float(config["lambda_m"]) + float(config["lambda_u"]) - 1.0) > 1e-6:
            raise ValueError("lambda_m + lambda_u must equal 1")
        self.config = config
        self.device = device
        self.smoke = smoke
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

    def _evaluate_all(
        self,
        model,
        model_stage: int,
        matrices: dict[str, PerformanceMatrix],
        eval_loaders: dict[int, Any],
        prediction_dirname: str = "predictions",
    ) -> None:
        prediction_dir = self.output_root / prediction_dirname
        for eval_stage, loader in eval_loaders.items():
            metrics, rows = evaluate_loader(
                model, loader, self.device, model_stage=model_stage, test_stage=eval_stage
            )
            save_predictions(
                rows,
                prediction_dir / f"model_stage_{model_stage:02d}_test_stage_{eval_stage:02d}.csv",
            )
            for name in METRIC_NAMES:
                matrices[name].add(model_stage, eval_stage, float(metrics[name]))
        for matrix in matrices.values():
            matrix.save(self.output_root / "matrices")

    def fit(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_root / "logs").mkdir(parents=True, exist_ok=True)
        with (self.output_root / "run_config.json").open("w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=2)

        eval_split = str(self.config.get("eval_split", "test"))
        pred_dirname = "predictions" if eval_split == "test" else "predictions_val"
        eval_manifests = {stage: self._prepare_manifest(stage, eval_split) for stage in range(self.n_stages)}
        eval_loaders = {
            stage: self._loader(manifest, False, 10_000 + stage)
            for stage, manifest in eval_manifests.items()
        }

        student = build_model(self.config).to(self.device)
        runtime: dict[str, Any] = {"stage_seconds": {}, "t0": time.time()}
        random_baseline = None
        if eval_split == "test" and not bool(self.config.get("skip_untrained_baseline", False)):
            untrained = build_model(self.config).to(self.device).eval()
            random_baseline = {
                stage: float(evaluate_loader(untrained, loader, self.device, -1, stage)[0]["auroc"])
                for stage, loader in eval_loaders.items()
            }
        matrices = {
            name: PerformanceMatrix(name, random_baseline if name == "auroc" else None)
            for name in METRIC_NAMES
        }

        print(f"===== Stage 0: supervised student only (eval_split={eval_split}) =====")
        t_stage = time.time()
        train0 = self._prepare_manifest(0, "train")
        val0 = self._prepare_manifest(0, "val")
        criterion0 = self._criterion(train0)
        student_ckpt = self.output_root / "checkpoints" / "stage_0_student_best.pt"
        init_s0 = self.config.get("init_stage0_checkpoint")
        if init_s0:
            payload = torch.load(init_s0, map_location=self.device, weights_only=False)
            student.load_state_dict(payload["model"])
            torch.save(
                {"model": student.state_dict(), "stage": 0, "copied_from": str(init_s0)},
                student_ckpt,
            )
            print(f"[stage0] loaded official checkpoint {init_s0}")
        else:
            train_supervised(
                model=student,
                train_loader=self._loader(train0, True, 1),
                val_loader=self._loader(val0, False, 2),
                criterion=criterion0,
                device=self.device,
                epochs=int(self.config["epochs_stage0"]),
                learning_rate=float(self.config["learning_rate_stage0"]),
                weight_decay=float(self.config["weight_decay"]),
                patience=int(self.config["early_stopping_patience"]),
                gradient_clip=float(self.config["gradient_clip"]),
                use_amp=bool(self.config["use_amp"]),
                optimizer_name=str(self.config["optimizer"]),
                scheduler_name=str(self.config["scheduler"]),
                scheduler_factor=float(self.config["scheduler_factor"]),
                scheduler_patience=int(self.config["scheduler_patience"]),
                checkpoint_path=student_ckpt,
                log_path=self.output_root / "logs" / "stage_0_training_log.csv",
                stage_name="stage0_student",
            )
        torch.save(
            {"model": student.state_dict(), "stage": 0},
            self.output_root / "checkpoints" / "stage_0_memory_teacher.pt",
        )
        initial_ckpt = self.output_root / "checkpoints" / "stage_0_initial_model.pt"
        initial_parameters = save_initial_model(
            student,
            initial_ckpt,
            self.output_root / "logs" / "stage_0_parameter_inventory.json",
        )
        with (self.output_root / "logs" / "stage_0_config.json").open("w", encoding="utf-8") as file:
            json.dump({"stage": 0, **self.config}, file, indent=2)
        self._evaluate_all(student, 0, matrices, eval_loaders, pred_dirname)
        runtime["stage_seconds"]["0"] = time.time() - t_stage

        memory = build_memory_teacher(student)
        for stage in range(1, self.n_stages):
            print(f"===== Stage {stage}: Memory / Updated Teacher / Student =====")
            t_stage = time.time()
            train_m = self._prepare_manifest(stage, "train")
            val_m = self._prepare_manifest(stage, "val")
            criterion = self._criterion(train_m)
            torch.save(
                {"model": memory.state_dict(), "stage": stage},
                self.output_root / "checkpoints" / f"stage_{stage}_memory_teacher.pt",
            )

            if float(self.config.get("lambda_u", 0.5)) <= 1e-12:
                updated = memory
                print(f"[ablation] skip Updated Teacher (lambda_u=0) at stage {stage}")
            else:
                updated, _ = train_updated_teacher(
                    memory_teacher=memory,
                    train_loader=self._loader(train_m, True, 100 + stage),
                    val_loader=self._loader(val_m, False, 200 + stage),
                    criterion=criterion,
                    config=self.config,
                    device=self.device,
                    checkpoint_path=self.output_root
                    / "checkpoints"
                    / f"stage_{stage}_updated_teacher_best.pt",
                    log_path=self.output_root / "logs" / f"stage_{stage}_updated_teacher_log.csv",
                )

            reloaded = load_initial_model(initial_ckpt)
            if not tensors_equal(initial_parameters, reloaded):
                raise RuntimeError("PSR Initial Model snapshot changed on disk; aborting.")
            student, _ = train_student(
                memory_teacher=memory,
                updated_teacher=updated,
                initial_parameters=initial_parameters,
                train_loader=self._loader(train_m, True, 300 + stage),
                val_loader=self._loader(val_m, False, 400 + stage),
                criterion=criterion,
                config=self.config,
                device=self.device,
                checkpoint_path=self.output_root / "checkpoints" / f"stage_{stage}_student_best.pt",
                log_path=self.output_root / "logs" / f"stage_{stage}_training_log.csv",
                val_log_path=self.output_root / "logs" / f"stage_{stage}_validation_metrics.csv",
                stage=stage,
            )
            with (self.output_root / "logs" / f"stage_{stage}_config.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump({"stage": stage, **self.config}, file, indent=2)
            self._evaluate_all(student, stage, matrices, eval_loaders, pred_dirname)
            runtime["stage_seconds"][str(stage)] = time.time() - t_stage
            memory = build_memory_teacher(student)

        final_stage = self.n_stages - 1
        torch.save(
            {"model": student.state_dict(), "stage": final_stage},
            self.output_root / "checkpoints" / f"final_student_stage{final_stage}.pt",
        )
        runtime["total_seconds"] = time.time() - float(runtime["t0"])
        with (self.output_root / "logs" / "runtime.json").open("w", encoding="utf-8") as file:
            json.dump(runtime, file, indent=2)
        print(f"[done] output={self.output_root} total_s={runtime['total_seconds']:.1f}")
