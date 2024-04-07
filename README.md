# INCPrompt: Task-Aware Incremental Prompting for Rehearsal-Free Class-Incremental Learning

This repository contains the official implementation of the paper "[INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning](https://arxiv.org/abs/2401.11667)".

## Dataset Preparation

Pass your dataset root with `--dataroot`.

Expected layout:

**CIFAR-100**

- Place the extracted `cifar-100-python/` folder under `--dataroot`.

**ImageNet-R**

- Place the dataset under `--dataroot/data/imagenet-r/`.
- The split files in `dataloaders/splits/*.yaml` are resolved relative to `--dataroot`.

## Default Inference Configuration

The repository now defaults to a more stable inference setting:

- `--routing_mode topk_uncertainty`
- `--routing_gate_type confidence`
- `--routing_conf_threshold 0.35`
- `--routing_top_k 3`
- `--routing_query_source shared_block`

`shared_block` reuses the prompt branch input as the routing query source, which avoids the extra ViT backbone pass that the legacy routing path used.

If you want the original hard routing behaviour, run with:

```bash
python run.py ... --routing_mode hard
```

If you want the legacy two-pass routing query for ablation or strict compatibility, add:

```bash
python run.py ... --routing_query_source backbone_pass
```

## Recommended 10-Task Run

The current recommended CIFAR-100 10-task command is:

```bash
python -u run.py \
  --config configs/cifar-100_prompt.yaml \
  --gpuid 0 \
  --dataroot /path/to/data \
  --log_dir outputs/cifar100-10task-default \
  --learner_type prompt \
  --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --routing_mode topk_uncertainty \
  --routing_gate_type confidence \
  --routing_conf_threshold 0.35 \
  --routing_top_k 3 \
  --routing_query_source shared_block \
  --repeat 1 \
  --overwrite 1 \
  --eval_during_train 0 \
  --eval_local 0
```

On Windows PowerShell, replace `/path/to/data` with your dataset root, for example:

```powershell
python -u run.py --config configs/cifar-100_prompt.yaml --gpuid 0 --dataroot E:\project\INCPrompt\data --log_dir outputs\cifar100-10task-default --learner_type prompt --learner_name INCPrompt --prompt_param 30 20 6 --routing_mode topk_uncertainty --routing_gate_type confidence --routing_conf_threshold 0.35 --routing_top_k 3 --routing_query_source shared_block --repeat 1 --overwrite 1 --eval_during_train 0 --eval_local 0
```

## Reproducing the Paper Setting

The paper reports INCPrompt with prompt length `20` and prompt depth `6`.

Split CIFAR-100:

```bash
python run.py --config configs/cifar-100_prompt.yaml --learner_type prompt --learner_name INCPrompt --prompt_param 30 20 6 --routing_mode hard --routing_query_source backbone_pass --log_dir outputs/CIFAR100/incprompt-paper
```

Split ImageNet-R:

```bash
python run.py --config configs/imnet-r_prompt.yaml --learner_type prompt --learner_name INCPrompt --prompt_param 30 20 6 --routing_mode hard --routing_query_source backbone_pass --log_dir outputs/ImageNet_R/incprompt-paper
```

For the optimized default inference path:

```bash
python run.py --config configs/cifar-100_prompt.yaml --learner_type prompt --learner_name INCPrompt --prompt_param 30 20 6 --log_dir outputs/CIFAR100/incprompt
```

You can also use the helper scripts in `experiments/`.

## Hyperparameters

Hyperparameters can be modified in the YAML files under `configs/`.

## Notes

- `INCPrompt` is the canonical learner name.
- The legacy alias `L2P` is still available for backward compatibility with older scripts or checkpoints.
- This repository currently contains the INCPrompt method only. The baseline comparison numbers in the paper are not reproduced here by standalone baseline implementations.

## Citation

```bibtex
@article{wang2024incprompt,
  title={INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning},
  author={Wang, Zhiyuan and Qu, Xiaoyang and Xiao, Jing and Chen, Bokui and Wang, Jianzong},
  journal={arXiv preprint arXiv:2401.11667},
  year={2024}
}
```

## Acknowledgement

Accepted by the 49th IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2024).
