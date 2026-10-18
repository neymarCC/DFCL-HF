# Method alignment (manuscript 3.2.2 and 3.3)

## Stage protocol

1. Stage 0: supervised Student on stage-0 train; best checkpoint by validation AUROC is \(S_0^*\).
2. Snapshot \(S_0^*\) named parameters \(\to\) Initial Model. Frozen for all later PSR. No Initial-Model forward pass.
3. For \(t\ge 1\):
   - \(M_t=\mathrm{deepcopy}(S_{t-1}^*)\), frozen (forward only for CAGD).
   - \(U_t\) initialized from the same \(S_{t-1}^*\), supervised FT on stage \(t\) only, then frozen.
   - \(S_t\) independent deepcopy of \(S_{t-1}^*\); optimize \(\mathcal{L}_\mathrm{sup}+\beta\mathcal{L}_\mathrm{CAGD}+\eta\mathcal{L}_\mathrm{PSR}\).
   - No previous-stage ECG in the loader.

Code: `src/continual/dfcl_hf_trainer.py`, `student_trainer.py`, `updated_teacher.py`, `memory_teacher.py`, `initial_model.py`.

## CAGD Eqs. (15)–(22)

| Eq. | Statement | Code |
|---|---|---|
| 15 | \(p=\mathrm{softmax}(z/\tau)\) | `temperature_softmax` |
| 16–17 | Shannon entropy, \(q=1-H/\log C\) | `teacher_confidence` |
| 18 | \(w=q\cdot\mathbf{1}[q\ge\delta]\) | same |
| 19 | Separate \(w^{(m)},w^{(u)}\) | `cagd_loss` |
| 20–21 | \(\tau^2\) weighted \(\mathrm{KL}(\mathrm{teacher}\|\mathrm{student})\) | `weighted_kl` |
| 22 | \(\lambda_m\mathcal{L}_m+\lambda_u\mathcal{L}_u\) | `loss_cagd` |

Empty-gate: if \(\sum w=0\), the teacher term is 0 (no NaN).

**Denominator \(\varepsilon\) placement.** Implemented form:

\[
\frac{\tau^2\sum_i w_i\,\mathrm{KL}_i}{\sum_i w_i+\varepsilon}
\]

If the manuscript writes \(\sum_i(w_i+\varepsilon)\) in the denominator, that is a different (larger) divisor; the official runs used \(\sum w+\varepsilon\). Not changed except the explicit zero when \(\sum w=0\).

## PSR Eqs. (23)–(25)

| Eq. | Statement | Code |
|---|---|---|
| 23 | Cosine of flattened tensors with \(\varepsilon\) | `cosine_similarity_eq23` |
| 24 | \(\sum_k(1-\cos(\theta_k))\) vs \(S_0^*\) | `psr_loss` returns `.sum()` |
| 25 | \(\mathcal{L}_\mathrm{sup}+\beta\mathcal{L}_\mathrm{CAGD}+\eta\mathcal{L}_\mathrm{PSR}\) | `student_total_loss` |

Parameter set: every `named_parameters()` tensor, including bias (`psr_include_bias: true`). Buffers (BN running stats) are not parameters and are not in PSR.

## Incomplete mathematics (reported, not patched)

Eq. (23) is undefined when \(\|a\|_2=\|b\|_2=0\). With \(+\varepsilon\), cosine evaluates to 0 so \(1-\cos=1\) even if Student equals Initial Model. Several PyTorch default biases start at 0. After Stage-0 training those tensors are typically nonzero. Cosine of a nonzero Student tensor against a **still-zero** Initial tensor is also identically 0, so that term is a constant with no gradient. This was **not** used as a reason to drop bias from the sum.
