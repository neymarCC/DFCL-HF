# DFCL-HF

Dual Frozen Continual Learning for Heart Failure (DFCL-HF) trains a 12-lead ECG classifier across sequential stages **without replaying historical waveforms**.

This repository is the DFCL-HF training and evaluation code corresponding to the manuscript:

- **CAGD**: temperature-scaled teacher probabilities, entropy confidence, gated KL (teacher \(\to\) student).
- **PSR**: cosine penalty of the current Student against a **frozen Stage-0 Initial Model**, summed over all named parameter tensors (Eqs. 23–24).

ECG recordings, patient manifests, and trained weights are **not** redistributed. Obtain MIMIC-IV-ECG and EchoNext under their data-use agreements and build patient-level CSVs locally.

## Method modules

| Component | Role | Code |
|---|---|---|
| CNN–BiLSTM–attention | Student / teacher backbone | `src/models/backbone.py` |
| Memory Teacher \(M_t\) | Frozen copy of \(S_{t-1}^*\) | `src/continual/memory_teacher.py` |
| Updated Teacher \(U_t\) | Current-stage supervised FT, then frozen | `src/continual/updated_teacher.py` |
| Initial Model \(S_0^*\) | Permanent PSR reference (no forward pass) | `src/continual/initial_model.py` |
| CAGD | Eqs. 15–22 | `src/losses/confidence_distillation.py` |
| PSR | Eqs. 23–24 | `src/losses/psr.py` |
| Continual loop | Stage 0 then no-replay stages | `src/continual/dfcl_hf_trainer.py` |

Inference uses the **Student** only.

## Layout

```text
configs/                 experiment hyperparameters
src/
scripts/                 train, evaluate, paired tests
tests/                   synthetic checks (no ECG)
examples/manifest_template.csv
data/README.md           where to put manifests (directory is gitignored)
ALGORITHM_DIFF.md
METHOD_ALIGNMENT.md
VALIDATION_REPORT.md
RERUN_PLAN.md
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA or CPU setup.

## Data

1. Credentialed PhysioNet access: [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/).
2. EchoNext under its data-use agreement.
3. Build **patient-level** stage manifests (approximately 70/10/20 train/val/test within each stage). A patient in stage \(t\) train must not appear in that stage’s val or test.
4. Place CSVs as:

```text
data/mimic_manifests/stage_{0..10}_{train,val,test}.csv
data/echonext_manifests/stage_{0..6}_{train,val,test}.csv
data/echonext_manifests/future_2022_test.csv   # optional
```

Required columns: `sample_id`, `patient_id`, `label`, `file_path`. Optional: `year`, `sample_index`.

Signals are resized to length 2000 and z-normalized per lead. Continual training loads **only the current stage** train/val files.

Helper: `src.data.splitting.patient_level_split` (seed 42).

## Commands

All commands assume the repository root as the working directory.

MIMIC-IV-ECG (S0–S10):

```bash
python scripts/train_dfcl_hf.py --config configs/dfcl_hf.yaml --device cuda
```

EchoNext (E0–E6):

```bash
python scripts/train_echonext_dfcl.py --config configs/dfcl_hf_echonext.yaml --device cuda
```

Evaluate a Student checkpoint:

```bash
python scripts/evaluate_dfcl_hf.py \
  --config configs/dfcl_hf_echonext.yaml \
  --checkpoint outputs/echonext/checkpoints/stage_6_student_best.pt \
  --model-stage 6 \
  --output-dir outputs/echonext/eval
```

2022 hold-out:

```bash
python scripts/evaluate_echonext_future.py \
  --config configs/dfcl_hf_echonext.yaml \
  --checkpoint outputs/echonext/checkpoints/stage_6_student_best.pt \
  --manifest data/echonext_manifests/future_2022_test.csv \
  --output-dir outputs/echonext/future_2022
```

MIMIC Student, zero extra fine-tuning on EchoNext:

```bash
python scripts/evaluate_echonext_zeroshot.py \
  --config configs/dfcl_hf.yaml \
  --checkpoint outputs/mimic/checkpoints/final_student_stage10.pt \
  --manifest data/echonext_manifests/external_test.csv \
  --output-dir outputs/mimic/echonext_zeroshot
```

Paired DeLong + bootstrap (\(B=2000\), seed 42):

```bash
python scripts/compare_2022_s6_vs_s10.py \
  --pred-s6 outputs/echonext/future_2022/s6_future_2022_predictions.csv \
  --pred-s10 outputs/mimic/echonext_2022/mimic_s10_echonext_2022_predictions.csv \
  --output-dir outputs/paired_2022
```

Override YAML fields:

```bash
python scripts/train_echonext_dfcl.py --set eta=0.1 --output-root outputs/ablations/eta_0.1
```

## Hyperparameters

| Item | Value |
|---|---|
| Optimizer | Adam, lr \(2\times10^{-4}\) |
| Batch / epochs | 32 / 50 |
| Early stopping | patience 5 on validation AUROC |
| Seed | 42 |
| \(\tau,\delta\) | 2.0, 0.3 |
| \(\lambda_m,\lambda_u\) | 0.5, 0.5 |
| \(\beta,\eta\) | 1.0, 1.5 |
| Focal \(\gamma,\alpha\) | 2.0, auto |
| PSR | Stage-0 Initial Model, sum of \(1-\cos\), all named parameters |

## Tests

```bash
python -m pytest tests -q
```

These tests check the model, losses, and freeze/gradient behaviour on synthetic tensors. They do not require MIMIC or EchoNext files.

## Availability

| Item | In this repository |
|---|---|
| Source, configs, tests | Yes |
| MIMIC / EchoNext ECG | No |
| Patient manifests | No (template only) |
| Model weights | No (train locally) |

See `METHOD_ALIGNMENT.md` for the mapping from manuscript equations to code.

Please cite the DFCL-HF paper when using the code. PhysioNet and EchoNext terms still apply to derived models.
