# bash experiments/imagenet-r.sh
# INCPrompt on Split ImageNet-R (10 tasks)

DATASET=ImageNet_R
OUTDIR=outputs/${DATASET}/10-task

# Routing state/statistics are tracked on the prompt module during forward,
# which DataParallel does not preserve across replicas: use a single GPU.
GPUID='0'
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=3
OVERWRITE=0

mkdir -p $OUTDIR

# Paper setting (ICASSP 2024): hard argmax routing on a full frozen-backbone
# query, no test-time augmentation. These are also the repository defaults.
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --routing_mode hard \
  --routing_query_source backbone_pass \
  --log_dir ${OUTDIR}/incprompt-paper

# Optional non-paper setting: shallow-pass routing query
# with router distillation and uncertainty-gated top-k prompt fusion.
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name INCPrompt \
#   --prompt_param 30 20 6 \
#   --routing_mode topk_uncertainty \
#   --routing_gate_type confidence \
#   --routing_conf_threshold 0.35 \
#   --routing_top_k 3 \
#   --routing_query_source shallow_pass \
#   --routing_query_depth 3 \
#   --routing_distill_weight 0.2 \
#   --log_dir ${OUTDIR}/incprompt-extended
