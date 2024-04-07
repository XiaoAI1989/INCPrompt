# INCPrompt: Task-Aware Incremental Prompting for Rehearsal-Free Class-Incremental Learning

This repository contains the official implementation of the paper "[INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning](https://arxiv.org/abs/2401.11667)" (ICASSP 2024).

## Installation

The experiments were written for Python 3.10 and PyTorch 2.0.0 with CUDA 11.8.

```bash
pip install -r requirements.txt
```

If your platform cannot install CUDA 11.8 wheels from `requirements.txt`, install
the matching PyTorch/torchvision build for your machine first, then install the
remaining packages from the file.

## Dataset Preparation

Pass your dataset root with `--dataroot`.

**CIFAR-100**

- Place the extracted `cifar-100-python/` folder under `--dataroot` (it is
  downloaded automatically on first use).

**ImageNet-R**

- Download ImageNet-R from the official release:
  https://github.com/hendrycks/imagenet-r
- Place the dataset under `--dataroot/data/imagenet-r/`.
- The split files in `dataloaders/splits/*.yaml` (24,000 train / 6,000 test,
  the standard split) are resolved relative to `--dataroot`.

## Reproducing the Paper Setting

The paper setting is the default: hard argmax routing over the task keys, with
the routing query taken from the CLS token of a full frozen-backbone pass, and
no test-time augmentation.

Split CIFAR-100 (10 tasks):

```bash
python -u run.py --config configs/cifar-100_prompt.yaml --gpuid 0 \
  --dataroot /path/to/data \
  --learner_type prompt --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --repeat 3 \
  --log_dir outputs/CIFAR100/incprompt-paper
```

Split ImageNet-R (10 tasks):

```bash
python -u run.py --config configs/imnet-r_prompt.yaml --gpuid 0 \
  --dataroot /path/to/data \
  --learner_type prompt --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --repeat 3 \
  --log_dir outputs/ImageNet_R/incprompt-paper
```

The helper scripts `experiments/cifar-100.sh` and `experiments/imagenet-r.sh`
run the same commands.

### Hyperparameters

| Hyperparameter | Value | Where |
|---|---|---|
| Backbone | Frozen timm 0.6.7 `vit_base_patch16_224(pretrained=True)` checkpoint (ViT-B/16; ImageNet-21k pretraining followed by ImageNet-1k fine-tuning in timm's default weights) | `models/zoo.py` |
| Prompt length | 20 (split into 10 prefix-key + 10 prefix-value tokens) | `--prompt_param 30 20 6` |
| Prompt depth | 6 (prompts attached to transformer blocks 0–5) | `--prompt_param 30 20 6` |
| Key regularization weight (λ_reg) | 1e-4 | `--reg_item` |
| Triplet margin (α) | 0.002 | `--triplet_margin` |
| Optimizer | Adam, lr 1e-3, cosine schedule, 20 epochs, batch 32 | `configs/*.yaml` |

Notes:

- The first value of `--prompt_param` (pool size, `30`) is unused by INCPrompt;
  it is kept for command-line compatibility with L2P-style configs.
- Training/evaluation transforms follow the CODA-Prompt framework (including
  its identity input normalization) for comparability within this framework.

### Evaluation Protocol

- Metrics: final average accuracy over all tasks after the last task, and
  average forgetting, computed in `trainer.py`.
- **No test-time augmentation by default** (`--tta_views 1`). The baseline
  numbers cited in the paper (L2P, DualPrompt, etc.) come from their original
  publications, which do not use TTA, so any comparison should keep TTA off.
  `--tta_views N` enables seeded, deterministic N-view TTA for separate
  analysis only.
- `--repeat N` runs N seeds (0..N-1) and reports mean±std. Note that seed 0
  keeps the original class order; seeds > 0 shuffle the class order when
  `rand_split: True`.
- Use a single GPU. Routing state and routing statistics are tracked on the
  prompt module during forward, which `DataParallel` does not preserve across
  replicas.

## Optional Routing Settings

The following routing settings are available through command-line flags and are
disabled by default.

**Shallow-pass routing query** (`--routing_query_source shallow_pass`,
`--routing_query_depth k`): computes the routing query from the CLS token after
the first `k` prompt-free transformer blocks instead of a full second backbone
pass, reducing the routing overhead from 12 extra blocks to `k`. Optionally,
`--routing_distill_weight 0.2` adds a training-time KL distillation that aligns
the shallow router with the full-backbone router.

**Uncertainty-gated top-k prompt fusion** (`--routing_mode topk_uncertainty`):
keeps hard argmax routing for confident queries, but when the routing
confidence falls below `--routing_conf_threshold` (or entropy rises above
`--routing_entropy_threshold` with `--routing_gate_type confidence_or_entropy`),
blends the top-`--routing_top_k` task prompts with residual weighting
(`--routing_fusion_mode residual --routing_residual_alpha 0.5`) instead of
committing to a possibly misrouted single task.

Example:

```bash
python -u run.py --config configs/cifar-100_prompt.yaml --gpuid 0 \
  --dataroot /path/to/data \
  --learner_type prompt --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --routing_mode topk_uncertainty \
  --routing_query_source shallow_pass --routing_query_depth 3 \
  --routing_distill_weight 0.2 \
  --log_dir outputs/CIFAR100/incprompt-extended
```

Routing diagnostics (per-task selection counts, routing confusion matrix,
confidence/entropy summaries) are written to
`<log_dir>/models/repeat-*/task-*/routing_analysis/`.

`--routing_query_source shared_block` is kept as a compatibility alias for
`shallow_pass`.

## Notes

- `INCPrompt` is the canonical learner name. The learner name `L2P` is accepted
  as a CLI compatibility alias for INCPrompt; it does **not** run the L2P
  baseline.
- This repository contains the INCPrompt method only. The baseline numbers in
  the paper are quoted from the original publications and are not reproduced by
  standalone baseline implementations here.

## Acknowledgements

This implementation is built on the
[CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt) code base (MIT License,
Copyright (c) 2023 GT-RIPL), whose training framework, data loaders and
prompted-ViT integration we adapt; `models/vit.py` is based on the ViT
implementation from [BLIP](https://github.com/salesforce/BLIP). We also thank
the authors of [L2P and DualPrompt](https://github.com/google-research/l2p)
for establishing the evaluation protocol. See `NOTICE` for license details.

Accepted by the 49th IEEE International Conference on Acoustics, Speech, and
Signal Processing (ICASSP 2024).

## Citation

```bibtex
@article{wang2024incprompt,
  title={INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning},
  author={Wang, Zhiyuan and Qu, Xiaoyang and Xiao, Jing and Chen, Bokui and Wang, Jianzong},
  journal={arXiv preprint arXiv:2401.11667},
  year={2024}
}
```
