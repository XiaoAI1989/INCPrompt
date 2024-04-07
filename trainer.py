import os
import random
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import yaml
from torch.utils.data import DataLoader

import dataloaders
import learners
from dataloaders import utils as data_utils


@dataclass(frozen=True)
class DatasetSpec:
    dataset_cls: type
    num_classes: int
    dataset_size: list
    top_k: int = 1


class Trainer:
    def __init__(self, args, seed, metric_keys, save_keys):
        self.args = args
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.model_top_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0
        self.eval_during_train = bool(getattr(args, 'eval_during_train', 1))
        self.eval_local = bool(getattr(args, 'eval_local', 1))
        self.pin_memory = bool(args.gpuid[0] >= 0)

        self.dataset_spec = self._select_dataset(args.dataset)
        self.dataset_size = self.dataset_spec.dataset_size
        self.top_k = self.dataset_spec.top_k
        self.num_classes = self.dataset_spec.num_classes

        if args.upper_bound_flag:
            args.other_split_size = self.num_classes
            args.first_split_size = self.num_classes

        self.tasks, self.tasks_logits = self._build_tasks()
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i + 1) for i in range(self.num_tasks)]
        self.max_task = self._resolve_max_task(args.max_task)

        self.train_dataset, self.test_dataset = self._create_datasets()
        self.learner_type = args.learner_type
        self.learner_name = args.learner_name
        self.learner_config = self._build_learner_config()
        self.learner = self._create_learner()

    def _select_dataset(self, dataset_name):
        if dataset_name == 'CIFAR100':
            return DatasetSpec(dataloaders.iCIFAR100, 100, [32, 32, 3])
        if dataset_name == 'ImageNet_R':
            return DatasetSpec(dataloaders.iIMAGENET_R, 200, [224, 224, 3])
        raise ValueError('Dataset not implemented!')

    def _build_tasks(self):
        class_order = np.arange(self.num_classes).tolist()
        class_order_logits = np.arange(self.num_classes).tolist()
        if self.seed > 0 and self.args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')

        tasks = []
        tasks_logits = []
        start = 0
        while start < self.num_classes and (self.args.max_task == -1 or len(tasks) < self.args.max_task):
            split_size = self.args.other_split_size if start > 0 else self.args.first_split_size
            tasks.append(class_order[start:start + split_size])
            tasks_logits.append(class_order_logits[start:start + split_size])
            start += split_size
        return tasks, tasks_logits

    def _resolve_max_task(self, max_task):
        if max_task > 0:
            return min(max_task, len(self.tasks))
        return len(self.tasks)

    def _create_datasets(self):
        train_transform, test_transform = self._create_transforms()
        dataset_cls = self.dataset_spec.dataset_cls
        train_dataset = dataset_cls(
            self.args.dataroot,
            train=True,
            lab=True,
            tasks=self.tasks,
            download_flag=True,
            transform=train_transform,
            seed=self.seed,
            rand_split=self.args.rand_split,
            validation=self.args.validation,
        )
        test_dataset = dataset_cls(
            self.args.dataroot,
            train=False,
            tasks=self.tasks,
            download_flag=False,
            transform=test_transform,
            seed=self.seed,
            rand_split=self.args.rand_split,
            validation=self.args.validation,
        )
        return train_dataset, test_dataset

    def _create_transforms(self):
        resize_imnet = self.args.model_name.startswith('vit')
        train_transform = data_utils.get_transform(
            dataset=self.args.dataset,
            phase='train',
            aug=self.args.train_aug,
            resize_imnet=resize_imnet,
        )
        test_transform = data_utils.get_transform(
            dataset=self.args.dataset,
            phase='test',
            aug=self.args.train_aug,
            resize_imnet=resize_imnet,
        )
        return train_transform, test_transform

    def _build_learner_config(self):
        return {
            'num_classes': self.num_classes,
            'lr': self.args.lr,
            'reg_item': self.args.reg_item,
            'debug_mode': self.args.debug_mode == 1,
            'momentum': self.args.momentum,
            'weight_decay': self.args.weight_decay,
            'schedule': self.args.schedule,
            'schedule_type': self.args.schedule_type,
            'model_type': self.args.model_type,
            'model_name': self.args.model_name,
            'optimizer': self.args.optimizer,
            'gpuid': self.args.gpuid,
            'memory': self.args.memory,
            'temp': self.args.temp,
            'out_dim': self.num_classes,
            'overwrite': self.args.overwrite == 1,
            'DW': self.args.DW,
            'batch_size': self.args.batch_size,
            'upper_bound_flag': self.args.upper_bound_flag,
            'tasks': self.tasks_logits,
            'top_k': self.top_k,
            'prompt_param': [self.num_tasks, self.args.prompt_param],
            'routing_config': {
                'mode': self.args.routing_mode,
                'gate_type': self.args.routing_gate_type,
                'top_k': self.args.routing_top_k,
                'confidence_threshold': self.args.routing_conf_threshold,
                'entropy_threshold': self.args.routing_entropy_threshold,
                'temperature': self.args.routing_temperature,
                'query_source': self.args.routing_query_source,
                'fusion_mode': self.args.routing_fusion_mode,
                'residual_alpha': self.args.routing_residual_alpha,
                'distill_weight': self.args.routing_distill_weight,
                'distill_temperature': self.args.routing_distill_temperature,
            },
        }

    def _create_learner(self):
        learner_cls = learners.__dict__[self.learner_type].__dict__[self.learner_name]
        return learner_cls(self.learner_config)

    @staticmethod
    def _model_ref(model):
        return model.module if hasattr(model, 'module') else model

    def _set_task_id(self, task_id):
        self._model_ref(self.learner.model).task_id = task_id

    def _advance_prompt_task_count(self):
        model = self._model_ref(self.learner.model)
        if getattr(model, 'prompt', None) is not None:
            model.prompt.process_task_count()

    def _get_prompt_module(self):
        model = self._model_ref(self.learner.model)
        return getattr(model, 'prompt', None)

    def _create_loader(self, dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=int(self.workers),
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.workers),
        )

    def _model_save_dir(self, task_index):
        return os.path.join(
            self.model_top_dir,
            'models',
            f'repeat-{self.seed + 1}',
            f'task-{self.task_names[task_index]}',
        )

    def _is_task_complete(self, task_index):
        model_save_dir = self._model_save_dir(task_index)
        return os.path.exists(model_save_dir + 'class.pth')

    def _find_resume_point(self):
        """Return the index of the first incomplete task (0 means start from scratch)."""
        for task_index in range(self.max_task):
            if not self._is_task_complete(task_index):
                return task_index
        return self.max_task

    def _fast_forward_to(self, resume_task):
        """Restore learner state for all completed tasks without loading data."""
        for task_index in range(resume_task):
            task = self.tasks_logits[task_index]
            self.add_dim = len(task)
            self._set_task_id(task_index)
            self.learner.add_valid_output_dim(self.add_dim)
            if task_index > 0:
                self._advance_prompt_task_count()
            self.learner.last_valid_out_dim = self.learner.valid_out_dim
            self.learner.task_count += 1
            print(f'[Resume] Task {self.task_names[task_index]} already complete, skipping.')

        # Load model weights from the last completed task
        if resume_task > 0:
            last_dir = self._model_save_dir(resume_task - 1)
            self.learner.load_model(last_dir)
            print(f'[Resume] Loaded model from task {self.task_names[resume_task - 1]}')

    def _prepare_training_task(self, task_index):
        task = self.tasks_logits[task_index]
        if self.oracle_flag:
            self.train_dataset.load_dataset(task_index, train=False)
            self.learner = self._create_learner()
            self.add_dim += len(task)
        else:
            self.train_dataset.load_dataset(task_index, train=True)
            self.add_dim = len(task)

        self._set_task_id(task_index)
        self.learner.add_valid_output_dim(self.add_dim)
        self.train_dataset.append_coreset(only=False)

    def _prepare_evaluation_task(self, task_index):
        if task_index > 0:
            self._advance_prompt_task_count()

        self.learner.task_count = task_index
        self.learner.add_valid_output_dim(len(self.tasks_logits[task_index]))
        self.learner.pre_steps()
        self.learner.load_model(self._model_save_dir(task_index))
        self._set_task_id(task_index)

    def task_eval(self, task_index, local=False, task='acc'):
        val_name = self.task_names[task_index]
        print('validation split name:', val_name)
        self.test_dataset.load_dataset(task_index, train=True)
        test_loader = self._create_loader(self.test_dataset, shuffle=False, drop_last=False)
        if local:
            return self.learner.validation(test_loader, task_in=self.tasks_logits[task_index], task_metric=task)
        return self.learner.validation(test_loader, task_metric=task)

    def _evaluate_seen_average(self, task_index):
        scores = [self.task_eval(eval_index) for eval_index in range(task_index + 1)]
        return float(np.mean(np.asarray(scores)))

    def _init_temp_table(self):
        return {metric_key: [] for metric_key in self.metric_keys}

    def _save_temp_metrics(self, temp_table):
        temp_dir = os.path.join(self.log_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for metric_key in ['acc']:
            save_path = os.path.join(temp_dir, f'{metric_key}.csv')
            np.savetxt(save_path, np.asarray(temp_table[metric_key]), delimiter=',', fmt='%.2f')

    def train(self, avg_metrics):
        temp_table = self._init_temp_table()

        # Resume: skip already-completed tasks
        resume_task = 0
        if not self.learner.overwrite:
            resume_task = self._find_resume_point()
            if resume_task > 0:
                print(f'[Resume] Found {resume_task} completed task(s), fast-forwarding...')
                self._fast_forward_to(resume_task)
            if resume_task >= self.max_task:
                print('[Resume] All tasks already complete.')
                return avg_metrics

        for task_index in range(resume_task, self.max_task):
            self.current_t_index = task_index
            print('======================', self.task_names[task_index], '=======================')

            self._prepare_training_task(task_index)
            if task_index > 0:
                self._advance_prompt_task_count()

            train_loader = self._create_loader(self.train_dataset, shuffle=True, drop_last=True)
            self.test_dataset.load_dataset(task_index, train=False)
            test_loader = self._create_loader(self.test_dataset, shuffle=False, drop_last=False)

            model_save_dir = self._model_save_dir(task_index)
            os.makedirs(model_save_dir, exist_ok=True)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)
            self.learner.save_model(model_save_dir)

            if self.eval_during_train:
                temp_table['acc'].append(self._evaluate_seen_average(task_index))
                self._save_temp_metrics(temp_table)

            if avg_train_time is not None:
                avg_metrics['time']['global'][task_index] = avg_train_time

        return avg_metrics

    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        avg_acc_history = [0] * self.max_task
        for train_task_index in range(self.max_task):
            train_name = self.task_names[train_task_index]
            cls_acc_sum = 0
            for val_task_index in range(train_task_index + 1):
                val_name = self.task_names[val_task_index]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[val_task_index, train_task_index, self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[val_task_index, train_task_index, self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[train_task_index] = cls_acc_sum / (train_task_index + 1)

        avg_acc_all[:, self.seed] = avg_acc_history
        return {'global': avg_acc_all, 'pt': avg_acc_pt, 'pt-local': avg_acc_pt_local}

    def _init_metric_tables(self):
        metric_table = {metric_key: {} for metric_key in self.metric_keys}
        metric_table_local = {metric_key: {} for metric_key in self.metric_keys}
        return metric_table, metric_table_local

    def _record_task_metrics(self, metric_table, metric_table_local, task_index):
        current_task_name = self.task_names[task_index]
        metric_table['acc'].setdefault(current_task_name, OrderedDict())
        metric_table_local['acc'].setdefault(current_task_name, OrderedDict())

        for eval_index in range(task_index + 1):
            val_name = self.task_names[eval_index]
            metric_table['acc'].setdefault(val_name, OrderedDict())
            metric_table_local['acc'].setdefault(val_name, OrderedDict())
            metric_table['acc'][val_name][current_task_name] = self.task_eval(eval_index)
            if self.eval_local:
                metric_table_local['acc'][val_name][current_task_name] = self.task_eval(eval_index, local=True)
            else:
                metric_table_local['acc'][val_name][current_task_name] = metric_table['acc'][val_name][current_task_name]

    def _reset_routing_analysis(self):
        prompt_module = self._get_prompt_module()
        if prompt_module is not None and hasattr(prompt_module, 'reset_routing_stats'):
            prompt_module.reset_routing_stats()

    def _save_routing_analysis(self, task_index):
        prompt_module = self._get_prompt_module()
        if prompt_module is None or not hasattr(prompt_module, 'get_routing_summary'):
            return

        summary = prompt_module.get_routing_summary()
        if not summary:
            return

        routing_tag = self.args.routing_mode
        if self.args.routing_mode == 'topk_uncertainty':
            routing_tag = (
                f"{routing_tag}_{self.args.routing_gate_type}_k{self.args.routing_top_k}"
                f"_c{self.args.routing_conf_threshold}"
                f"_e{self.args.routing_entropy_threshold}"
                f"_t{self.args.routing_temperature}"
                f"_{self.args.routing_fusion_mode}"
                f"_ra{self.args.routing_residual_alpha}"
                f"_dw{self.args.routing_distill_weight}"
                f"_dt{self.args.routing_distill_temperature}"
            )
        analysis_dir = os.path.join(self._model_save_dir(task_index), 'routing_analysis', routing_tag)
        os.makedirs(analysis_dir, exist_ok=True)

        with open(os.path.join(analysis_dir, 'summary.yaml'), 'w', encoding='utf-8') as summary_file:
            yaml.safe_dump(summary, summary_file, sort_keys=False)

        confusion = np.asarray(summary['confusion_matrix'], dtype=np.int64)
        np.savetxt(os.path.join(analysis_dir, 'confusion_matrix.csv'), confusion, delimiter=',', fmt='%d')
        np.savetxt(
            os.path.join(analysis_dir, 'selection_counts.csv'),
            np.asarray(summary['selection_counts'], dtype=np.int64),
            delimiter=',',
            fmt='%d',
        )

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(confusion, cmap='Blues')
            ax.set_title('Routing Confusion Matrix')
            ax.set_xlabel('Predicted Route Task')
            ax.set_ylabel('True Task')
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(os.path.join(analysis_dir, 'confusion_matrix.png'), dpi=200)
            plt.close(fig)
        except Exception:
            pass

    def _compute_average_forgetting(self, acc_table):
        if self.max_task <= 1:
            return 0.0

        total_forgetting = 0
        final_task_name = self.task_names[self.max_task - 1]
        for task_index in range(self.max_task - 1):
            task_name = self.task_names[task_index]
            acc_org = acc_table[task_name][task_name]
            acc_final = acc_table[task_name][final_task_name]
            total_forgetting += acc_org - acc_final
        return total_forgetting / (self.max_task - 1)

    def evaluate(self, avg_metrics):
        self.learner = self._create_learner()
        metric_table, metric_table_local = self._init_metric_tables()

        for task_index in range(self.max_task):
            self._prepare_evaluation_task(task_index)
            self._reset_routing_analysis()
            self._record_task_metrics(metric_table, metric_table_local, task_index)
            self._save_routing_analysis(task_index)

        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'], metric_table_local['acc'])
        avg_f = self._compute_average_forgetting(metric_table['acc'])
        return avg_metrics, avg_f
