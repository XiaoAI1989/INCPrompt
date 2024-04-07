# bash experiments/imagenet-r.sh
# INCPrompt on Split ImageNet-R

DATASET=ImageNet_R
OUTDIR=outputs/${DATASET}/10-task

GPUID='0 1 2 3'
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=1
OVERWRITE=0

mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name INCPrompt \
  --prompt_param 30 20 6 \
  --routing_mode topk_uncertainty \
  --routing_gate_type confidence \
  --routing_conf_threshold 0.35 \
  --routing_top_k 3 \
  --routing_query_source shared_block \
  --log_dir ${OUTDIR}/incprompt
