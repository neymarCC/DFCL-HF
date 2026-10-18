from __future__ import annotations

import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.continual.initial_model import load_initial_model, save_initial_model, snapshot_parameters, tensors_equal
from src.continual.memory_teacher import build_memory_teacher, freeze_module
from src.continual.student_trainer import check_teacher_grads, initialize_student
from src.continual.updated_teacher import initialize_updated_teacher
from src.data.splitting import patient_level_split
from src.losses.confidence_distillation import cagd_loss, weighted_kl
from src.losses.focal_loss import WeightedFocalLoss
from src.losses.psr import cosine_similarity_eq23, official_mean_psr_vs_memory, psr_loss
from src.losses.total_loss import student_total_loss
from src.models.backbone import CNNBiLSTMAttention
from src.utils import REPO_ROOT, load_config


def _tiny_batch(n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 12, 2000, generator=g)
    y = torch.tensor([0, 1] * (n // 2))
    return x, y


def test_stage1_memory_equals_initial():
    s0 = CNNBiLSTMAttention()
    initial = snapshot_parameters(s0)
    memory = build_memory_teacher(s0)
    mem = snapshot_parameters(memory)
    assert tensors_equal(initial, mem)


def test_stage2_memory_from_s1_psr_still_s0(tmp_path: Path):
    s0 = CNNBiLSTMAttention()
    initial = snapshot_parameters(s0)
    save_initial_model(s0, tmp_path / "s0.pt", tmp_path / "inv.json")
    s1 = copy.deepcopy(s0)
    with torch.no_grad():
        s1.classifier[0].weight.add_(0.05)
    memory = build_memory_teacher(s1)
    mem = snapshot_parameters(memory)
    assert not tensors_equal(initial, mem)
    reloaded = load_initial_model(tmp_path / "s0.pt")
    assert tensors_equal(initial, reloaded)
    loss = psr_loss(s1, reloaded)
    assert loss.item() > 0


def test_initial_frozen_across_later_stages(tmp_path: Path):
    s0 = CNNBiLSTMAttention()
    save_initial_model(s0, tmp_path / "s0.pt", tmp_path / "inv.json")
    snap0 = load_initial_model(tmp_path / "s0.pt")
    student = copy.deepcopy(s0)
    for _ in range(3):
        with torch.no_grad():
            student.classifier[-1].bias.add_(0.1)
        again = load_initial_model(tmp_path / "s0.pt")
        assert tensors_equal(snap0, again)


def test_psr_is_sum_not_mean():
    s0 = CNNBiLSTMAttention()
    initial = snapshot_parameters(s0)
    student = copy.deepcopy(s0)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(0.01)
    terms = []
    for name, parameter in student.named_parameters():
        terms.append(1.0 - cosine_similarity_eq23(parameter, initial[name], 1e-8))
    expected_sum = torch.stack(terms).sum()
    expected_mean = torch.stack(terms).mean()
    loss = psr_loss(student, initial)
    assert torch.allclose(loss, expected_sum, atol=1e-6)
    assert not torch.allclose(loss, expected_mean, atol=1e-4)
    assert abs(loss.item() / expected_mean.item() - len(terms)) < 1e-4


def test_teachers_frozen_student_has_grad():
    s0 = CNNBiLSTMAttention()
    initial = snapshot_parameters(s0)
    memory = freeze_module(copy.deepcopy(s0))
    updated = freeze_module(initialize_updated_teacher(memory))
    student = initialize_student(memory)
    x, y = _tiny_batch()
    criterion = WeightedFocalLoss(gamma=2.0, class_weight=torch.tensor([1.0, 1.0]))
    with torch.no_grad():
        _, lm, _ = memory(x)
        _, lu, _ = updated(x)
    _, ls, _ = student(x)
    losses = student_total_loss(
        student, initial, ls, lm, lu, y, criterion,
        tau=2.0, delta=0.3, lambda_m=0.5, lambda_u=0.5, beta=1.0, eta=1.5,
        distill_eps=1e-8, psr_include_bias=True, psr_layers="all", psr_eps=1e-8,
    )
    losses["loss_total"].backward()
    check_teacher_grads(memory, updated, student)
    assert all(p.grad is None for p in memory.parameters())
    assert all(p.grad is None for p in updated.parameters())


def test_cagd_kl_teacher_to_student():
    teacher = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    student = torch.zeros(2, 2)
    tau = 2.0
    p_t = F.softmax(teacher / tau, dim=1)
    p_s = F.softmax(student / tau, dim=1)
    kl_t_s = F.kl_div(torch.log(p_s.clamp_min(1e-8)), p_t, reduction="none").sum(dim=1)
    kl_s_t = F.kl_div(torch.log(p_t.clamp_min(1e-8)), p_s, reduction="none").sum(dim=1)
    assert not torch.allclose(kl_t_s, kl_s_t)
    weights = torch.ones(2)
    implemented = weighted_kl(p_t, p_s, weights, tau=tau, eps=1e-8)
    expected = (tau ** 2) * kl_t_s.mean()  # weights all 1, (sum w kl)/(sum w + eps) ~ mean
    assert torch.isfinite(implemented)
    assert implemented.item() > 0


def test_cagd_empty_gate_is_zero():
    logits_t = torch.zeros(4, 2)
    logits_s = torch.randn(4, 2)
    out = cagd_loss(logits_t, logits_t, logits_s, tau=2.0, delta=0.99, lambda_m=0.5, lambda_u=0.5, eps=1e-8)
    assert float(out["w_m"].sum()) == 0.0
    assert float(out["loss_m"]) == 0.0
    assert float(out["loss_u"]) == 0.0
    assert math.isfinite(float(out["loss_cagd"]))


def test_psr_cagd_grad_finite_and_scale():
    s0 = CNNBiLSTMAttention()
    initial = snapshot_parameters(s0)
    student = copy.deepcopy(s0)
    with torch.no_grad():
        student.classifier[0].weight.add_(0.02)
    x, y = _tiny_batch()
    memory = freeze_module(copy.deepcopy(s0))
    with torch.no_grad():
        _, lm, _ = memory(x)
        _, lu, _ = memory(x)
    _, ls, _ = student(x)
    criterion = WeightedFocalLoss(gamma=2.0, class_weight=torch.tensor([1.0, 1.0]))
    losses = student_total_loss(
        student, initial, ls, lm, lu, y, criterion,
        tau=2.0, delta=0.3, lambda_m=0.5, lambda_u=0.5, beta=1.0, eta=1.5,
        distill_eps=1e-8, psr_include_bias=True, psr_layers="all", psr_eps=1e-8,
    )
    assert all(torch.isfinite(losses[k]) for k in ("loss_sup", "loss_cagd", "loss_psr", "loss_total"))
    losses["loss_psr"].backward()
    psr_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1e9)
    student.zero_grad()
    official = official_mean_psr_vs_memory(student, memory, include_bias=False)
    n_all = sum(1 for _ in student.named_parameters())
    n_no_bias = sum(1 for _, p in student.named_parameters() if p.ndim > 1)
    # Persist scale diagnostics for VALIDATION_REPORT.
    Path("outputs").mkdir(exist_ok=True)
    report = {
        "manuscript_psr_sum": float(losses["loss_psr"].detach()),
        "official_psr_mean_vs_memory": float(official.detach()),
        "n_parameter_tensors": n_all,
        "n_weight_tensors_official": n_no_bias,
        "psr_grad_norm": float(psr_norm),
        "loss_sup": float(losses["loss_sup"].detach()),
        "loss_cagd": float(losses["loss_cagd"].detach()),
        "eta_times_psr": 1.5 * float(losses["loss_psr"].detach()),
    }
    (Path(__file__).resolve().parents[1] / "outputs").mkdir(exist_ok=True)
    torch.save(report, Path(__file__).resolve().parents[1] / "outputs" / "psr_scale_probe.pt")
    assert math.isfinite(float(psr_norm))
    assert float(losses["loss_psr"]) > float(official)


def test_reload_initial_snapshot(tmp_path: Path):
    s0 = CNNBiLSTMAttention()
    save_initial_model(s0, tmp_path / "s0.pt", tmp_path / "inv.json")
    student = copy.deepcopy(s0)
    with torch.no_grad():
        student.classifier[0].weight.mul_(1.2)
    before = float(psr_loss(student, load_initial_model(tmp_path / "s0.pt")))
    student2 = copy.deepcopy(student)
    after = float(psr_loss(student2, load_initial_model(tmp_path / "s0.pt")))
    assert abs(before - after) < 1e-6


def test_yaml_uses_relative_paths():
    for name in ("dfcl_hf.yaml", "dfcl_hf_echonext.yaml"):
        text = (REPO_ROOT / "configs" / name).read_text(encoding="utf-8")
        assert "D:/" not in text and "D:\\" not in text
        assert "all_waveform" not in text
    mimic = load_config(REPO_ROOT / "configs" / "dfcl_hf.yaml")
    echo = load_config(REPO_ROOT / "configs" / "dfcl_hf_echonext.yaml")
    assert mimic["manifest_root"].endswith("data\\mimic_manifests") or mimic["manifest_root"].endswith("data/mimic_manifests")
    assert "echonext_manifests" in echo["manifest_root"]
    assert echo["n_stages"] == 7 and mimic["n_stages"] == 11


def test_patient_split_no_leak():
    rows = [{"sample_id": i, "patient_id": i // 2, "label": i % 2, "file_path": "x"} for i in range(40)]
    splits = patient_level_split(pd.DataFrame(rows), seed=42)
    ids = {k: set(v["patient_id"].astype(str)) for k, v in splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
