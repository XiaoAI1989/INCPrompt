import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit import VisionTransformer


class L2P(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, input_dim=768, hidden_dim=384, reg_item=1e-4,
                 epsilon=1e-7, margin=0.002, tri_item=1):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.reg_item = reg_item
        self.epsilon = epsilon
        self.task_learners = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, (self.e_p_length) * emb_d),
                    nn.ReLU()
                ) for _ in range(len(self.e_layers))
            ]) for _ in range(n_tasks)
        ])

        self.key_learners = nn.ModuleList([AttentionResidualNet(input_dim, hidden_dim, emb_d) for _ in range(n_tasks)])
        self.max_sim_indices = None
        self.margin = margin
        self.tri_item = tri_item

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:

            self.e_layers = [i for i in range(int(prompt_param[2]))]
        else:
            self.e_layers = [0]

        # prompt length (k+v)
        self.e_p_length = int(prompt_param[1])

    def process_task_count(self):
        self.task_count += 1

    def freeze_all_but_nth_model(self, n):
        for i, model in enumerate(self.key_learners):
            for param in model.parameters():
                if i == n:
                    param.requires_grad = True  # 解冻第 n 个模型
                else:
                    param.requires_grad = False

    def forward(self, x_query, l, x_block, train=False, task_id=None):
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_query.shape

            q = nn.functional.normalize(x_query, dim=1).detach()
            logits = q.unsqueeze(0)
            if train:
                if l == 0:
                    positive_key = []
                    negative_keys = []
                    for i, key_learner in enumerate(self.key_learners):

                        if i < task_id:
                            key = key_learner(q)
                            negative_keys.append(key)
                        elif i == task_id:
                            P_key = key_learner(q)
                            positive_key.append(logits)
                    self.freeze_all_but_nth_model(task_id)
                    # Use the outputs from the other task learners as the negative samples
                    P_flat = P_key.view(P_key.size(0), -1).unsqueeze(0)
                    pos_sim = nn.functional.cosine_similarity(logits, P_flat, dim=-1).mean()

                    logits_norm = F.normalize(logits, p=2, dim=-1)
                    P_key_norm = torch.norm(P_key, p=1)
                    reg_loss = 0
                    reg_loss = P_key_norm
                    # 首先，我们需要找到N_keys中与正样本余弦相似度最高的那个负样本
                    triplet_loss = 0

                    if task_id != 0:
                        N_keys = torch.stack(negative_keys)

                        neg_sim = nn.functional.cosine_similarity(P_flat, N_keys, dim=-1)

                        if task_id > 1:
                            top_neg_sim_values, top_indices = torch.topk(neg_sim, 1, dim=0)

                            top_N_keys = N_keys[top_indices]
                        else:
                            top_N_keys = N_keys

                        # 然后，我们计算锚点到正样本和负样本的距离
                        pos_dist = torch.norm(P_flat - logits_norm, dim=-1)
                        neg_dist = torch.norm(logits_norm - top_N_keys, dim=-1)

                        # 计算三元组损失
                        triplet_loss = torch.relu(pos_dist - neg_dist + self.margin).mean()
                        pos_sim = 0
                    prompt_loss = self.reg_item * reg_loss + self.tri_item * triplet_loss - 0.3 * pos_sim

                P_ = self.task_learners[task_id][l](q).clone().view(-1, self.e_p_length, self.emb_d)

            else:
                if l == 0:
                    samples = []
                    for i, task_learner in enumerate(self.task_learners):
                        # P_ = task_learner[l](q).clone().view(-1, self.e_p_length, self.emb_d)
                        P_key = self.key_learners[i](q)
                        samples.append(P_key)
                    key = torch.stack(samples, dim=0)

                    sim = nn.functional.cosine_similarity(logits, key, dim=-1)
                    max_sim_indices = torch.argmax(sim, dim=0)
                    self.max_sim_indices = max_sim_indices
                    from collections import Counter

                    # 假设 max_sim_indices 是一个一维 numpy 数组
                    # 先将 numpy 数组转化为 Python 列表
                    max_sim_indices_list = self.max_sim_indices.tolist()
                    # 使用 Counter 计算每个索引出现的次数
                    index_counts = Counter(max_sim_indices_list)
                    # 计算总数，用于计算比例
                    total_count = len(max_sim_indices_list)

                    # 创建一个字典来存储每个索引及其比例
                    index_proportions = {}

                    for index, count in index_counts.items():
                        index_proportions[index] = count / total_count

                P_ = []
                # 对于每一个数据，在对应的模型中获取P_
                for i in range(self.max_sim_indices.shape[0]):
                    idx = self.max_sim_indices[i].item()
                    P_.append(self.task_learners[idx][l](q[i]).reshape(-1, self.e_p_length, self.emb_d))
                P_ = torch.cat(P_, dim=0)  # 将所有的P_堆叠为一个张量

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :].clone().reshape((B, -1, self.emb_d))
            Ev = P_[:, i:, :].clone().reshape((B, -1, self.emb_d))

        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None
            prompt_loss = 0

        # return
        if train and l == 0:
            return p_return, prompt_loss, x_block
        else:
            return p_return, 0, x_block


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                          num_heads=12, ckpt_layer=0,
                                          drop_path_rate=0
                                          )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight'];
            del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])

        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)
                q = q[:, 0, :]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

    def process_task_count(self):
        self.task_count += 1


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(x.size(-1)), dim=-1)
        out = attn_weights @ v
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualAttentionBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.attn = SelfAttention(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        if input_dim != hidden_dim:
            self.shortcut = nn.Linear(input_dim, hidden_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = nn.functional.relu(self.linear1(x))
        out = self.attn(out)
        out += self.shortcut(x)
        out = nn.functional.relu(self.linear2(out))
        return out


class AttentionResidualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(AttentionResidualNet, self).__init__()
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_blocks)]
        )
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = F.relu(out)
        out = self.final_layer(out)
        out = F.normalize(out, p=2, dim=-1)
        return out
