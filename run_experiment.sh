#!/bin/bash

set -e

# Paper setting (ICASSP 2024) on Split ImageNet-R: hard argmax routing on a
# full frozen-backbone query, no test-time augmentation (repository defaults).
python run.py \
  --config configs/imnet-r_prompt.yaml \
  --learner_type prompt \
  --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --log_dir outputs/ImageNet_R/incprompt-paper
