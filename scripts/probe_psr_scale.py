from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.continual.initial_model import snapshot_parameters
from src.continual.memory_teacher import build_memory_teacher, freeze_module
from src.losses.focal_loss import WeightedFocalLoss
from src.losses.psr import cosine_similarity_eq23, official_mean_psr_vs_memory, psr_loss
from src.losses.total_loss import student_total_loss
from src.models.backbone import CNNBiLSTMAttention


def _psr_float(student, init) -> float:
    return float(psr_loss(student, init).detach())


def main() -> None:
    s0 = CNNBiLSTMAttention()
    init = snapshot_parameters(s0)
    identical = _psr_float(s0, init)
    student = copy.deepcopy(s0)
    with torch.no_grad():
        student.classifier[0].weight.add_(0.02)
    terms = []
    for name, parameter in student.named_parameters():
        value = 1.0 - float(cosine_similarity_eq23(parameter, init[name], 1e-8))
        terms.append({"name": name, "one_minus_cos": value, "ndim": parameter.ndim, "numel": int(parameter.numel())})
    manuscript = sum(item["one_minus_cos"] for item in terms)
    memory = build_memory_teacher(s0)
    official = float(official_mean_psr_vs_memory(student, memory, include_bias=False))
    x = torch.randn(8, 12, 2000)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    criterion = WeightedFocalLoss(2.0, torch.tensor([1.0, 1.0]))
    frozen = freeze_module(copy.deepcopy(s0))
    with torch.no_grad():
        _, memory_logits, _ = frozen(x)
    eta_rows = []
    for eta in (1.5, 0.1, 0.05, 0.02):
        current = copy.deepcopy(student)
        _, student_logits, _ = current(x)
        losses = student_total_loss(
            current, init, student_logits, memory_logits, memory_logits, y, criterion,
            2.0, 0.3, 0.5, 0.5, 1.0, eta, 1e-8, True, "all", 1e-8,
        )
        losses["loss_total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(current.parameters(), 1e9))
        eta_rows.append(
            {
                "eta": eta,
                "loss_sup": float(losses["loss_sup"].detach()),
                "loss_cagd": float(losses["loss_cagd"].detach()),
                "loss_psr": float(losses["loss_psr"].detach()),
                "eta_times_psr": eta * float(losses["loss_psr"].detach()),
                "loss_total": float(losses["loss_total"].detach()),
                "grad_norm": grad_norm,
            }
        )
    trained = copy.deepcopy(s0)
    opt = torch.optim.Adam(trained.parameters(), lr=2e-4)
    for _ in range(5):
        opt.zero_grad()
        _, logits, _ = trained(x)
        criterion(logits, y).backward()
        opt.step()
    trained_init = snapshot_parameters(trained)
    trained_identical = _psr_float(trained, trained_init)
    perturb = copy.deepcopy(trained)
    with torch.no_grad():
        perturb.classifier[0].weight.add_(0.02)
    trained_psr = _psr_float(perturb, trained_init)
    trained_official = float(official_mean_psr_vs_memory(perturb, freeze_module(copy.deepcopy(trained)), False).detach())
    payload = {
        "untrained_identical_student_psr": identical,
        "trained_s0_identical_psr": trained_identical,
        "trained_s0_perturbed_manuscript_sum": trained_psr,
        "trained_s0_perturbed_official_mean": trained_official,
        "n_tensors": len(terms),
        "n_1d_tensors": sum(item["ndim"] <= 1 for item in terms),
        "manuscript_sum": manuscript,
        "manuscript_mean_equivalent": manuscript / len(terms),
        "official_mean_vs_memory_no_bias": official,
        "sum_over_mean_ratio": manuscript / official if official else None,
        "contribution_1d": sum(item["one_minus_cos"] for item in terms if item["ndim"] <= 1),
        "contribution_ndgt1": sum(item["one_minus_cos"] for item in terms if item["ndim"] > 1),
        "largest_terms": sorted(terms, key=lambda item: -item["one_minus_cos"])[:8],
        "eta_probe": eta_rows,
    }
    out = ROOT / "outputs" / "psr_scale_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
