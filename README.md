# Project Title
INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning


# INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning

This is the official implementation of paper "[INCPrompt: Task-Aware incremental Prompting for Rehearsal-Free Class-incremental Learning](https://arxiv.org/abs/2401.11667)". Our approach progressive Incremental Prompting (INCPrompt) aims to mitigate forgetting problem existed in the continual learning of Vision Transformer. 

### Dataset Preparation

If you already have CIFAR-100 or ImageNet-R, pass your dataset path to  `--data-path`.


The datasets aren't ready, change the download argument in `datasets.py` as follows

**CIFAR-100**
```
datasets.CIFAR100(download=True)
```

**ImageNet-R**
```
Imagenet_R(download=True)
```

### Replicate our results

If you want to replicate our results in our paper, you can directly run the command line for that table. For example, if you want to replicate the results of ZSCL, then run `sh run_experiment.sh`.


### Training and Evaluation

run `sh run_experiment.sh`.

### Hyperparameters

You can modify the Hyperparameters at the yaml file in the configs fold.


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
