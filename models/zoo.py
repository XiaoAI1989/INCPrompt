from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit import VisionTransformer


@dataclass(frozen=True)
class PromptSpec:
    num_tasks: int
    prompt_length: int
    prompt_layers: list

    @classmethod
    def from_prompt_param(cls, prompt_param):
        num_tasks, prompt_args = prompt_param
        prompt_length = int(prompt_args[1])
        prompt_depth = int(prompt_args[2])
        prompt_layers = list(range(prompt_depth)) if prompt_depth > 0 else [0]
        return cls(num_tasks=int(num_tasks), prompt_length=prompt_length, prompt_layers=prompt_layers)


class TaskPromptGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, prompt_length, embed_dim):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_length * embed_dim),
            nn.ReLU(),
        )

    def forward(self, query):
        prompt = self.network(query)
        return prompt.view(-1, self.prompt_length, self.embed_dim)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = F.softmax(query @ key.transpose(-2, -1) / math.sqrt(x.size(-1)), dim=-1)
        return attention @ value


class ResidualAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.attention = SelfAttention(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.shortcut = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        hidden = self.attention(hidden)
        hidden = hidden + self.shortcut(x)
        return F.relu(self.linear2(hidden))


class AttentionResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_blocks)]
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = F.relu(x)
        x = self.output(x)
        return F.normalize(x, p=2, dim=-1)


class TaskAwarePromptPool(nn.Module):
    def __init__(
        self,
        embed_dim,
        prompt_spec,
        input_dim=768,
        hidden_dim=384,
        reg_item=1e-4,
        margin=0.002,
        triplet_weight=1.0,
        routing_config=None,
    ):
        super().__init__()
        routing_config = routing_config or {}

        self.embed_dim = embed_dim
        self.prompt_spec = prompt_spec
        self.reg_item = reg_item
        self.margin = margin
        self.triplet_weight = triplet_weight
        self.task_count = 0

        self.routing_mode = routing_config.get('mode', 'topk_uncertainty')
        self.routing_gate_type = routing_config.get('gate_type', 'confidence')
        self.routing_top_k = max(1, int(routing_config.get('top_k', 3)))
        self.routing_confidence_threshold = float(routing_config.get('confidence_threshold', 0.35))
        self.routing_entropy_threshold = float(routing_config.get('entropy_threshold', 0.75))
        self.routing_temperature = max(float(routing_config.get('temperature', 0.2)), 1e-6)
        self.routing_query_source = routing_config.get('query_source', 'shared_block')
        self.routing_fusion_mode = routing_config.get('fusion_mode', 'residual')
        self.routing_residual_alpha = float(routing_config.get('residual_alpha', 0.5))
        self.routing_distill_weight = float(routing_config.get('distill_weight', 0.2))
        self.routing_distill_temperature = max(float(routing_config.get('distill_temperature', 1.0)), 1e-6)

        self.layer_to_slot = {layer: slot for slot, layer in enumerate(self.prompt_spec.prompt_layers)}
        self.prompt_generators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TaskPromptGenerator(input_dim, hidden_dim, self.prompt_spec.prompt_length, embed_dim)
                        for _ in self.prompt_spec.prompt_layers
                    ]
                )
                for _ in range(self.prompt_spec.num_tasks)
            ]
        )
        self.key_learners = nn.ModuleList(
            [AttentionResidualNet(input_dim, hidden_dim, embed_dim) for _ in range(self.prompt_spec.num_tasks)]
        )
        self.query_norm = nn.LayerNorm(input_dim)

        self.current_routing_info = None
        self.current_query = None
        self.reset_routing_stats()

    def process_task_count(self):
        self.task_count += 1

    def reset_routing_stats(self):
        num_tasks = self.prompt_spec.num_tasks
        self.routing_stats = {
            'num_samples': 0,
            'num_fused': 0,
            'num_misrouted': 0,
            'num_topk_hits': 0,
            'entropy_sum': 0.0,
            'confidence_sum': 0.0,
            'blend_sum': 0.0,
            'selection_counts': torch.zeros(num_tasks, dtype=torch.long),
            'confusion': torch.zeros((num_tasks, num_tasks), dtype=torch.long),
        }

    def get_routing_summary(self):
        total = int(self.routing_stats['num_samples'])
        if total == 0:
            return None

        return {
            'routing_mode': self.routing_mode,
            'routing_top_k': self.routing_top_k,
            'routing_gate_type': self.routing_gate_type,
            'routing_query_source': self.routing_query_source,
            'routing_fusion_mode': self.routing_fusion_mode,
            'routing_residual_alpha': self.routing_residual_alpha,
            'routing_distill_weight': self.routing_distill_weight,
            'routing_distill_temperature': self.routing_distill_temperature,
            'confidence_threshold': self.routing_confidence_threshold,
            'entropy_threshold': self.routing_entropy_threshold,
            'temperature': self.routing_temperature,
            'num_samples': total,
            'num_fused': int(self.routing_stats['num_fused']),
            'fused_rate': float(self.routing_stats['num_fused']) / total,
            'misroute_rate': float(self.routing_stats['num_misrouted']) / total,
            'topk_hit_rate': float(self.routing_stats['num_topk_hits']) / total,
            'avg_entropy': float(self.routing_stats['entropy_sum']) / total,
            'avg_confidence': float(self.routing_stats['confidence_sum']) / total,
            'avg_blend_factor': float(self.routing_stats['blend_sum']) / total,
            'selection_counts': self.routing_stats['selection_counts'].tolist(),
            'confusion_matrix': self.routing_stats['confusion'].tolist(),
        }

    def record_batch_statistics(self, true_task_ids):
        if self.current_routing_info is None:
            return

        true_task_ids = true_task_ids.detach().cpu().long().view(-1)
        predicted_tasks = self.current_routing_info['top1_indices'].detach().cpu().long()
        use_topk = self.current_routing_info['use_topk'].detach().cpu()
        entropies = self.current_routing_info['entropies'].detach().cpu()
        confidences = self.current_routing_info['confidences'].detach().cpu()
        topk_indices = self.current_routing_info['topk_indices'].detach().cpu().long()
        blend_factors = self.current_routing_info['blend_factors'].detach().cpu()

        batch_size = int(true_task_ids.numel())
        self.routing_stats['num_samples'] += batch_size
        self.routing_stats['num_fused'] += int(use_topk.sum().item())
        self.routing_stats['num_misrouted'] += int((predicted_tasks != true_task_ids).sum().item())
        self.routing_stats['entropy_sum'] += float(entropies.sum().item())
        self.routing_stats['confidence_sum'] += float(confidences.sum().item())
        self.routing_stats['blend_sum'] += float(blend_factors.sum().item())
        self.routing_stats['selection_counts'] += torch.bincount(
            predicted_tasks,
            minlength=self.prompt_spec.num_tasks,
        )

        topk_hits = (topk_indices == true_task_ids.unsqueeze(1)).any(dim=1)
        self.routing_stats['num_topk_hits'] += int(topk_hits.sum().item())

        for true_task, predicted_task in zip(true_task_ids.tolist(), predicted_tasks.tolist()):
            self.routing_stats['confusion'][true_task, predicted_task] += 1

    def forward(self, x_query, layer_index, x_block, train=False, task_id=None, teacher_query=None):
        if layer_index not in self.layer_to_slot:
            return None, x_block.new_zeros(1), x_block

        slot = self.layer_to_slot[layer_index]
        first_prompt_layer = layer_index == self.prompt_spec.prompt_layers[0]
        if first_prompt_layer:
            self.current_query = self._resolve_query(x_query, x_block)
        elif self.current_query is None:
            raise RuntimeError('Routing query is unavailable before the first prompt layer.')
        query = self.current_query

        if train:
            if task_id is None:
                raise ValueError('task_id is required during training.')
            self.current_routing_info = None
            prompt_tokens = self.prompt_generators[task_id][slot](query)
            if first_prompt_layer:
                prompt_loss = self._compute_key_loss(query, task_id)
                prompt_loss = prompt_loss + self._compute_router_distillation_loss(query, teacher_query, task_id)
            else:
                prompt_loss = query.new_zeros(1)
        else:
            if first_prompt_layer:
                self.current_routing_info = self._route_queries(query)
            prompt_tokens = self._generate_inference_prompt(query, slot)
            prompt_loss = query.new_zeros(1)

        return self._split_prompt(prompt_tokens), prompt_loss, x_block

    def _resolve_query(self, x_query, x_block):
        if x_query is not None:
            return F.normalize(x_query, dim=1).detach()
        if self.routing_query_source != 'shared_block':
            raise ValueError(f'Unsupported routing query source: {self.routing_query_source}')
        cls_token = self.query_norm(x_block[:, 0, :].detach())
        return F.normalize(cls_token, dim=1)

    def should_compute_teacher_query(self, train):
        return train and self.routing_distill_weight > 0 and self.routing_query_source != 'backbone_pass'

    def _active_task_count(self, task_id=None):
        seen_tasks = self.task_count + 1
        if task_id is not None:
            seen_tasks = max(seen_tasks, task_id + 1)
        return min(self.prompt_spec.num_tasks, max(1, seen_tasks))

    def _compute_similarity_matrix(self, query, task_limit):
        task_keys = torch.stack([self.key_learners[idx](query) for idx in range(task_limit)], dim=0)
        return F.cosine_similarity(query.unsqueeze(0), task_keys, dim=-1).transpose(0, 1)

    def _routing_blend_factors(self, confidences, use_topk):
        blend_factors = torch.zeros_like(confidences)
        if self.routing_fusion_mode != 'residual':
            blend_factors[use_topk] = 1.0
            return blend_factors

        threshold = max(self.routing_confidence_threshold, 1e-6)
        uncertainty = ((threshold - confidences).clamp(min=0.0) / threshold)
        blend_factors[use_topk] = self.routing_residual_alpha * uncertainty[use_topk]
        return blend_factors.clamp_(0.0, 1.0)

    def _route_queries(self, query):
        task_limit = self._active_task_count()
        similarities = self._compute_similarity_matrix(query, task_limit)
        probabilities = F.softmax(similarities / self.routing_temperature, dim=-1)

        top_k = min(self.routing_top_k, task_limit)
        topk_probabilities, topk_indices = torch.topk(probabilities, k=top_k, dim=-1)
        confidences = topk_probabilities[:, 0]
        entropies = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)
        if task_limit > 1:
            entropies = entropies / math.log(task_limit)

        use_topk = torch.zeros_like(confidences, dtype=torch.bool)
        if self.routing_mode == 'topk_uncertainty':
            if self.routing_gate_type == 'confidence':
                use_topk = confidences < self.routing_confidence_threshold
            else:
                use_topk = (confidences < self.routing_confidence_threshold) | (
                    entropies > self.routing_entropy_threshold
                )
        blend_factors = self._routing_blend_factors(confidences, use_topk)

        return {
            'top1_indices': topk_indices[:, 0],
            'topk_indices': topk_indices,
            'topk_probabilities': topk_probabilities,
            'confidences': confidences,
            'entropies': entropies,
            'use_topk': use_topk,
            'blend_factors': blend_factors,
        }

    def _generate_inference_prompt(self, query, slot):
        routing_info = self.current_routing_info
        batch_size = query.size(0)
        prompt = query.new_empty(batch_size, self.prompt_spec.prompt_length, self.embed_dim)

        hard_mask = ~routing_info['use_topk']
        if hard_mask.any():
            hard_indices = routing_info['top1_indices'][hard_mask]
            for task_index in hard_indices.unique(sorted=True).tolist():
                mask = hard_mask.clone()
                mask[hard_mask] = hard_indices == task_index
                prompt[mask] = self.prompt_generators[task_index][slot](query[mask])

        if routing_info['use_topk'].any():
            fused_indices = routing_info['use_topk'].nonzero(as_tuple=False).view(-1)
            for sample_index in fused_indices.tolist():
                weights = routing_info['topk_probabilities'][sample_index]
                weights = weights / weights.sum().clamp_min(1e-8)
                top1_task_index = int(routing_info['top1_indices'][sample_index].item())
                top1_prompt = self.prompt_generators[top1_task_index][slot](query[sample_index:sample_index + 1])
                fused_prompt = None
                for weight, task_index in zip(weights, routing_info['topk_indices'][sample_index]):
                    task_prompt = self.prompt_generators[int(task_index.item())][slot](
                        query[sample_index:sample_index + 1]
                    )
                    if fused_prompt is None:
                        fused_prompt = weight * task_prompt
                    else:
                        fused_prompt = fused_prompt + weight * task_prompt
                blend = routing_info['blend_factors'][sample_index]
                prompt[sample_index:sample_index + 1] = top1_prompt + blend * (fused_prompt - top1_prompt)

        return prompt

    def _compute_router_distillation_loss(self, student_query, teacher_query, task_id):
        if self.routing_distill_weight <= 0 or teacher_query is None:
            return student_query.new_zeros(1)

        task_limit = self._active_task_count(task_id)
        if task_limit <= 1:
            return student_query.new_zeros(1)

        student_similarity = self._compute_similarity_matrix(student_query, task_limit)
        with torch.no_grad():
            teacher_similarity = self._compute_similarity_matrix(F.normalize(teacher_query, dim=1).detach(), task_limit)

        temperature = self.routing_distill_temperature
        student_log_probs = F.log_softmax(student_similarity / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_similarity / temperature, dim=-1)
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        return self.routing_distill_weight * distill_loss

    def _compute_key_loss(self, query, task_id):
        self._freeze_key_learners(task_id)
        positive_key = self.key_learners[task_id](query)
        reg_loss = torch.norm(positive_key, p=1)

        if task_id == 0:
            return self.reg_item * reg_loss

        negative_keys = torch.stack([self.key_learners[i](query) for i in range(task_id)], dim=0)
        negative_similarities = F.cosine_similarity(positive_key.unsqueeze(0), negative_keys, dim=-1)
        hard_negative_indices = torch.argmax(negative_similarities, dim=0)
        sample_indices = torch.arange(query.size(0), device=query.device)
        hard_negatives = negative_keys[hard_negative_indices, sample_indices]

        positive_distance = torch.norm(positive_key - query, dim=-1)
        negative_distance = torch.norm(query - hard_negatives, dim=-1)
        triplet_loss = torch.relu(positive_distance - negative_distance + self.margin).mean()
        return self.reg_item * reg_loss + self.triplet_weight * triplet_loss

    def _freeze_key_learners(self, active_task_id):
        for task_index, key_learner in enumerate(self.key_learners):
            requires_grad = task_index == active_task_id
            for param in key_learner.parameters():
                param.requires_grad = requires_grad

    def _split_prompt(self, prompt_tokens):
        half = self.prompt_spec.prompt_length // 2
        prompt_key = prompt_tokens[:, :half, :].contiguous()
        prompt_value = prompt_tokens[:, half:, :].contiguous()
        return [prompt_key, prompt_value]


class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, routing_config=None):
        super().__init__()
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.feat = self._build_backbone(pt)
        self.last = nn.Linear(768, num_classes)
        self.prompt = self._build_prompt_module(prompt_param, routing_config)

    def _build_backbone(self, pretrained):
        backbone = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ckpt_layer=0,
            drop_path_rate=0,
        )
        if pretrained:
            from timm.models import vit_base_patch16_224

            state_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del state_dict['head.weight']
            del state_dict['head.bias']
            backbone.load_state_dict(state_dict)
        return backbone

    def _build_prompt_module(self, prompt_param, routing_config):
        if self.prompt_flag not in {'l2p', 'incprompt'}:
            return None
        prompt_spec = PromptSpec.from_prompt_param(prompt_param)
        return TaskAwarePromptPool(
            embed_dim=768,
            prompt_spec=prompt_spec,
            routing_config=routing_config,
        )

    def forward(self, x, pen=False, train=False):
        prompt_loss = x.new_zeros(1)
        if self.prompt is not None:
            query = None
            teacher_query = None
            if getattr(self.prompt, 'routing_query_source', 'shared_block') == 'backbone_pass':
                teacher_query = self._encode_query(x)
                query = teacher_query
            elif self.prompt.should_compute_teacher_query(train=train):
                teacher_query = self._encode_query(x)
            prompt_task_id = self.task_id if train else None
            tokens, prompt_loss = self.feat(
                x,
                prompt=self.prompt,
                q=query,
                train=train,
                task_id=prompt_task_id,
                teacher_q=teacher_query,
            )
        else:
            tokens, _ = self.feat(x)

        features = tokens[:, 0, :].reshape(tokens.size(0), -1)
        logits = features if pen else self.last(features)
        if self.prompt is not None and train:
            return logits, prompt_loss
        return logits

    def _encode_query(self, x):
        with torch.no_grad():
            tokens, _ = self.feat(x)
        return tokens[:, 0, :]


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None, routing_config=None):
    return ViTZoo(
        num_classes=out_dim,
        pt=True,
        prompt_flag=prompt_flag,
        prompt_param=prompt_param,
        routing_config=routing_config,
    )
