from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

from trainer import Trainer


def create_args():
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/ablation length",
                        help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='prompt', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='INCPrompt', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--reg_item', type=float, default=0.1, help="The weight of regularization loss.")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N',
                        help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[30, 20, 6],
                        help="Prompt pool size, prompt length, prompt depth.")
    parser.add_argument('--routing_mode', type=str, default='topk_uncertainty',
                        choices=['hard', 'topk_uncertainty'],
                        help='Inference-time routing strategy.')
    parser.add_argument('--routing_top_k', type=int, default=3,
                        help='Number of prompts to fuse when uncertainty-aware routing is triggered.')
    parser.add_argument('--routing_conf_threshold', type=float, default=0.35,
                        help='Trigger top-k routing when top-1 routing confidence falls below this threshold.')
    parser.add_argument('--routing_entropy_threshold', type=float, default=0.75,
                        help='Trigger top-k routing when normalized routing entropy exceeds this threshold.')
    parser.add_argument('--routing_temperature', type=float, default=0.2,
                        help='Temperature used to convert key similarities into routing probabilities.')
    parser.add_argument('--routing_gate_type', type=str, default='confidence',
                        choices=['confidence', 'confidence_or_entropy'],
                        help='Condition used to trigger uncertainty-aware routing.')
    parser.add_argument('--routing_query_source', type=str, default='shared_block',
                        choices=['shared_block', 'backbone_pass'],
                        help='How to obtain routing queries. shared_block avoids a second ViT backbone pass.')
    parser.add_argument('--routing_fusion_mode', type=str, default='residual',
                        choices=['replace', 'residual'],
                        help='How uncertainty-triggered top-k prompts are fused at inference time.')
    parser.add_argument('--routing_residual_alpha', type=float, default=0.5,
                        help='Maximum residual blending strength used by residual top-k fusion.')
    parser.add_argument('--routing_distill_weight', type=float, default=0.2,
                        help='Weight of training-time router distillation from backbone_pass to shared_block.')
    parser.add_argument('--routing_distill_temperature', type=float, default=1.0,
                        help='Temperature used for router distillation targets.')
    parser.add_argument('--dataroot', type=str, default=None, help='Dataset root override.')
    parser.add_argument('--max_task', type=int, default=None, help='Maximum number of tasks to run.')
    parser.add_argument('--eval_during_train', type=int, default=None,
                        help='Whether to run seen-task evaluation after each training task. 1/0.')
    parser.add_argument('--eval_local', type=int, default=None,
                        help='Whether to compute pt-local metrics during evaluation. 1/0.')

    # Config Arg
    parser.add_argument('--config', type=str, default="configs/imnet-r_prompt.yaml",
                        help="yaml experiment config input")

    return parser


def get_args(argv):
    parser = create_args()
    args = parser.parse_args(argv)
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    override_args = {key: value for key, value in vars(args).items() if value is not None}
    config.update(override_args)
    return argparse.Namespace(**config)


# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic = True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)

    metric_keys = ['acc', 'time', ]
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys:
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # load results
    if args.overwrite:
        start_r = 0
    else:
        try:
            for mkey in metric_keys:
                for skey in save_keys:
                    if (not (mkey in global_only)) or (skey == 'global'):
                        save_file = args.log_dir + '/results-' + mkey + '/' + skey + '.yaml'
                        if os.path.exists(save_file):
                            with open(save_file, 'r') as yaml_file:
                                yaml_result = yaml.safe_load(yaml_file)
                                avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])

            # next repeat needed
            start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]

            # extend if more repeats left
            if start_r < args.repeat:
                max_task = avg_metrics['acc']['global'].shape[0]
                for mkey in metric_keys:
                    avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'],
                                                            np.zeros((max_task, args.repeat - start_r)), axis=-1)
                    if (not (mkey in global_only)):
                        avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'],
                                                            np.zeros((max_task, max_task, args.repeat - start_r)),
                                                            axis=-1)
                        avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'],
                                                                  np.zeros((max_task, max_task, args.repeat - start_r)),
                                                                  axis=-1)

        except:
            start_r = 0
    # start_r = 0
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r + 1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # set up a trainer
        trainer = Trainer(args, seed, metric_keys, save_keys)

        # init total run metrics storage
        max_task = trainer.max_task
        if r == 0:
            for mkey in metric_keys:
                avg_metrics[mkey]['global'] = np.zeros((max_task, args.repeat))
                if (not (mkey in global_only)):
                    avg_metrics[mkey]['pt'] = np.zeros((max_task, max_task, args.repeat))
                    avg_metrics[mkey]['pt-local'] = np.zeros((max_task, max_task, args.repeat))

        # train model
        avg_metrics = trainer.train(avg_metrics)

        # evaluate model
        avg_metrics, avg_f = trainer.evaluate(avg_metrics)

        # save results
        for mkey in metric_keys:
            m_dir = args.log_dir + '/results-' + mkey + '/'
            if not os.path.exists(m_dir): os.makedirs(m_dir)
            for skey in save_keys:
                if (not (mkey in global_only)) or (skey == 'global'):
                    save_file = m_dir + skey + '.yaml'
                    result = avg_metrics[mkey][skey]
                    yaml_results = {}
                    if len(result.shape) > 2:
                        yaml_results['mean'] = result[:, :, :r + 1].mean(axis=2).tolist()
                        if r > 1: yaml_results['std'] = result[:, :, :r + 1].std(axis=2).tolist()
                        yaml_results['history'] = result[:, :, :r + 1].tolist()
                    else:
                        yaml_results['mean'] = result[:, :r + 1].mean(axis=1).tolist()
                        if r > 1: yaml_results['std'] = result[:, :r + 1].std(axis=1).tolist()
                        yaml_results['history'] = result[:, :r + 1].tolist()
                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # Print the summary so far
        print('===Summary of experiment repeats:', r + 1, '/', args.repeat, '===')
        for mkey in metric_keys:
            print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1, :r + 1].mean(), 'std:',
                  avg_metrics[mkey]['global'][-1, :r + 1].std())
        print(f"the forgetting rate is: {avg_f}")
