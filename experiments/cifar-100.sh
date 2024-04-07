# bash experiments/cifar-100.sh
# INCPrompt on Split CIFAR-100

DATASET=CIFAR100
OUTDIR=outputs/${DATASET}/10-task

GPUID='0 1 2 3'
CONFIG=configs/cifar-100_prompt.yaml
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
