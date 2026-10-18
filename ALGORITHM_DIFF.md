# Algorithm diff: official `shiyan` vs manuscript-aligned

Compared with the original official trainer (mean PSR vs Memory Teacher). This tree is the manuscript-aligned implementation.

## PSR (Eqs. 23–24) — changed

| Item | Official `shiyan/src/losses/psr.py` | This version `src/losses/psr.py` |
|---|---|---|
| Reference | Current-stage Memory Teacher \(M_t=\mathrm{deepcopy}(S_{t-1}^*)\) | Permanently frozen Stage-0 snapshot \(S_0^*\) |
| Updated later? | Yes, every stage | No; `stage_0_initial_model.pt` is re-checked each stage |
| Reduction | `torch.stack(terms).mean()` | `torch.stack(terms).sum()` |
| Parameter set | Skip 1-D tensors (`psr_include_bias: false`) | All `named_parameters`, including bias |
| Cosine | `F.cosine_similarity(..., eps)` | Eq. (23): \(\langle a,b\rangle/(\|a\|_2\|b\|_2+\varepsilon)\) |
| Forward on reference | Not required | Not required (parameter dict only) |

Official mean-vs-memory implementation is kept as `official_mean_psr_vs_memory` for tests only.

## Training loop — changed

| Item | Official | This version |
|---|---|---|
| After Stage 0 | Save Student / Memory Teacher | Also save `checkpoints/stage_0_initial_model.pt` and `logs/stage_0_parameter_inventory.json` |
| Student loss | `psr_loss(student, memory_teacher)` | `psr_loss(student, initial_parameters)` |
| Memory Teacher | Unchanged: freeze \(S_{t-1}^*\) | Unchanged |
| Updated Teacher | Unchanged: FT on current stage, freeze | Unchanged |
| No replay | Unchanged | Unchanged |

## CAGD (Eqs. 15–22) — almost unchanged

Official `confidence_distillation.py` already matched temperature softmax, entropy confidence, \(w=q\cdot\mathbf{1}[q\ge\delta]\), per-teacher weighted KL, \(\tau^2\), and \(\lambda_m,\lambda_u\).

The only code change: if a teacher’s weight mass is 0 in the batch, `weighted_kl` returns exact `0` instead of `0 / (0 + eps)`. Numerically both are 0; this avoids relying on \(\varepsilon\) for the empty-gate case.

PyTorch `kl_div(log(p_s), p_t)` remains \(\mathrm{KL}(p_t\|p_s)\).

Denominator actually implemented: \(\tau^2 \sum_i w_i\,\mathrm{KL}_i / (\sum_i w_i + \varepsilon)\). Not \(\sum_i(w_i+\varepsilon)\).

## Config

Optimizer defaults copied from official YAML (\(\beta=1,\eta=1.5,\delta=0.3,\tau=2,\lambda_m=\lambda_u=0.5\)). PSR flags:

- `psr_reference: stage0_initial_model`
- `psr_reduction: sum`
- `psr_include_bias: true`

`output_root` points at this directory so official `shiyan/runs` are not overwritten.

## Scale change (synthetic probe)

See `VALIDATION_REPORT.md`. After identical untrained weights, manuscript PSR is **not** 0 because several zero-initialized bias tensors make Eq. (23) undefined (cosine of two zero vectors \(\to 0\) with \(+\varepsilon\), so \(1-\cos=1\)). After a few supervised steps those biases leave zero and identical-student PSR \(\approx 0\). Sum vs official mean is roughly an order of magnitude larger; \(\eta=1.5\) was **not** retuned.
