# Method alignment

## Stage protocol

1. Stage 0: supervised Student on stage-0 train; the best validation-AUROC checkpoint is \(S_0^*\).
2. Snapshot \(S_0^*\) named parameters as the Initial Model. Frozen for all later PSR. No Initial-Model forward pass.
3. For \(t\ge 1\):
   - \(M_t=\mathrm{deepcopy}(S_{t-1}^*)\), frozen (forward only for CAGD).
   - \(U_t\) initialized from the same \(S_{t-1}^*\), supervised fine-tuning on stage \(t\) only, then frozen.
   - \(S_t\) independent deepcopy of \(S_{t-1}^*\); optimize \(\mathcal{L}_\mathrm{sup}+\beta\mathcal{L}_\mathrm{CAGD}+\eta\mathcal{L}_\mathrm{PSR}\).
   - No previous-stage ECG in the loader.

Code: `src/continual/dfcl_hf_trainer.py`, `student_trainer.py`, `updated_teacher.py`, `memory_teacher.py`, `initial_model.py`.

## CAGD Eqs. (15)–(22)

| Eq. | Statement | Code |
|---|---|---|
| 15 | \(p=\mathrm{softmax}(z/\tau)\) | `temperature_softmax` |
| 16–17 | Shannon entropy, \(q=1-H/\log C\) | `teacher_confidence` |
| 18 | \(w=q\cdot\mathbf{1}[q\ge\delta]\) | `teacher_confidence` |
| 19 | Separate \(w^{(m)},w^{(u)}\) | `cagd_loss` |
| 20–21 | \(\tau^2\) weighted \(\mathrm{KL}(\mathrm{teacher}\|\mathrm{student})\) | `weighted_kl` |
| 22 | \(\lambda_m\mathcal{L}_m+\lambda_u\mathcal{L}_u\) | `loss_cagd` |

If \(\sum w=0\) for a teacher in the current batch, that term is 0.

## PSR Eqs. (23)–(25)

| Eq. | Statement | Code |
|---|---|---|
| 23 | Cosine of flattened tensors with \(\varepsilon\) | `cosine_similarity_eq23` |
| 24 | \(\sum_k(1-\cos(\theta_k))\) vs \(S_0^*\) | `psr_loss` (sum over named parameters) |
| 25 | \(\mathcal{L}_\mathrm{sup}+\beta\mathcal{L}_\mathrm{CAGD}+\eta\mathcal{L}_\mathrm{PSR}\) | `student_total_loss` |

All `named_parameters()` tensors are included, including bias. Batch-norm running buffers are not parameters and are not in PSR.
