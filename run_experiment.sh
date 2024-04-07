#!/bin/bash

set -e

# Default optimized command on Split ImageNet-R.
python run.py \
  --config configs/imnet-r_prompt.yaml \
  --learner_type prompt \
  --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --routing_mode topk_uncertainty \
  --routing_gate_type confidence \
  --routing_conf_threshold 0.35 \
  --routing_top_k 3 \
  --routing_query_source shared_block \
  --log_dir outputs/ImageNet_R/incprompt
