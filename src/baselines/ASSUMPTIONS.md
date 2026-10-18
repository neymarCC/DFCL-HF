# Baseline method assumptions

Official-style implementations on this CNN-BiLSTM backbone:
- Naive Fine-tuning: sequential supervised loss only, initialize from the previous best checkpoint.
- EWC: diagonal empirical Fisher on the just-finished stage training set (uses that stage once; not historical ECG replay).
- LwF: frozen previous model; temperature-scaled KL on current data.
- DKD: Zhao et al., CVPR 2022, TCKD + NCKD on current data.

Idea-based adaptations (no official in-repo paper code; not claimed as faithful paper clones):
- API: after each stage, freeze the top `api_freeze_fraction` of still-trainable parameters by mean diagonal Fisher. The last classifier layer stays trainable.
- LwHT: frozen previous student; logit KD plus feature MSE. Hierarchical = output-level + representation-level. Not DFCL-HF.
- DualCP: DualPrompt-style shared G-prompt plus expert E-prompts with cosine key matching, added to pooled CNN-BiLSTM features. Extractor frozen after Stage 0. Not ViT prefix-tuning.

Shared protocol with DFCL-HF: same manifests, backbone, preprocessing, batch size, Adam/LR, max epochs, early stopping, validation AUROC selection. No test-set tuning. No raw historical ECG replay.
