# Algorithm

DFCL-HF trains a 12-lead Student without replaying historical ECG. After Stage 0, each increment uses a frozen Memory Teacher, a frozen Updated Teacher, CAGD, and PSR against the Stage-0 Initial Model.

## Training loop

1. Stage 0: supervised training on stage-0 data. The best validation-AUROC Student is \(S_0^*\).
2. Save \(S_0^*\) as `checkpoints/stage_0_initial_model.pt` with a parameter inventory. This snapshot is the PSR reference for every later stage and is never updated.
3. For stage \(t \ge 1\):
   - Memory Teacher \(M_t=\mathrm{deepcopy}(S_{t-1}^*)\), frozen.
   - Updated Teacher \(U_t\) starts from the same \(S_{t-1}^*\), is fine-tuned on stage-\(t\) data only, then frozen.
   - Student \(S_t\) is an independent copy of \(S_{t-1}^*\).
   - Optimize \(\mathcal{L}_\mathrm{sup}+\beta\mathcal{L}_\mathrm{CAGD}+\eta\mathcal{L}_\mathrm{PSR}\) on the current stage only.
4. The best validation Student is passed to the next stage.

Code: `src/continual/dfcl_hf_trainer.py`, `student_trainer.py`, `updated_teacher.py`, `memory_teacher.py`, `initial_model.py`.

## CAGD

Temperature softmax, entropy confidence, weights \(w=q\cdot\mathbf{1}[q\ge\delta]\), per-teacher \(\tau^2\)-weighted \(\mathrm{KL}(\mathrm{teacher}\|\mathrm{student})\), then \(\lambda_m\mathcal{L}_m+\lambda_u\mathcal{L}_u\).

If a teacher has no samples above \(\delta\) in the batch, that teacher term is 0.

Code: `src/losses/confidence_distillation.py`.

## PSR

Reference: frozen Stage-0 parameters \(S_0^*\). Each named parameter tensor is flattened; cosine uses \(\langle a,b\rangle/(\|a\|_2\|b\|_2+\varepsilon)\). The loss is the **sum** of \(1-\cos\) over all named parameters, including bias. The Initial Model is not run in the forward pass.

Code: `src/losses/psr.py`. Checkpoint: `stage_0_initial_model.pt`.

## Config

- \(\beta=1\), \(\eta=1.5\), \(\delta=0.3\), \(\tau=2\), \(\lambda_m=\lambda_u=0.5\)
- `psr_reference: stage0_initial_model`
- `psr_reduction: sum`
- `psr_include_bias: true`
