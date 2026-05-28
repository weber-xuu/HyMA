import random
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import os
import datasets
from tqdm import tqdm
import argparse
import wandb
import swanlab
import time 

from eval import evaluate

import utils

import numpy as np

from models.hyperbolic_losses import (
    exact_entailment_cone_violation,
    label_cone_loss,
    lorentz_distance,
    query_to_label_tree_alignment_loss,
)
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data', type=str, default='rcv1')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--arch', type=str, default='/data/xwb/bert-base-uncased')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT') # 'GAT', 'GCN', 'graphormer' none
    parser.add_argument('--orth_method', type=str, default='gram_schmidt') # gram_schmidt', 'svd', 'householder', 'none'
    parser.add_argument('--mlm_mask_strategy', type=str, default='legacy') # legacy', 'legacy_bert', 'legacy_tokenizer_aware', 'filtered'
    
    parser.add_argument('--classifier_head', type=str, default='hybrid') # euclidean', 'hyperbolic', 'hybrid'
    parser.add_argument('--hyperbolic_dim', type=int, default=256)
    parser.add_argument('--hyperbolic_alpha', type=float, default=0.40) #  hybrid 分类头里“双曲分支权重”的全局系数
    parser.add_argument('--depth_aware_hyper_alpha', default=True, action='store_true') # 是给 hybrid 分类头用的“分层融合系数”，如果开了这个选项，模型会为每个 depth 学习一个 alpha

    parser.add_argument('--hyperbolic_curvature_init', type=float, default=1.0)  # 双曲曲率 c 的全局初值；推荐先用 1.0，稳定且是当前最安全起点
    parser.add_argument('--curvature_init_by_depth', type=str, default=None)  # 每层曲率初值，逗号分隔；默认 None 表示自动使用 --hyperbolic_curvature_init 并按实际 depth 广播，只有想手工指定每层初值时才填写
    parser.add_argument('--per_depth_curvature', default=True, action='store_true')  # 是否让每个 depth 单独学习曲率；推荐正式实验开 True，先排错或做简化基线时关 False

    parser.add_argument('--hyperbolic_logit_scale_init', type=float, default=1.0)  # 双曲距离转 logits 的初始缩放；推荐 1.0，过大容易让早期 logits 过尖
    parser.add_argument('--hyperbolic_radius_clip', type=float, default=1.5)  # exp map 前切空间向量范数的平滑 bounded-norm 上界；推荐 2.0，若仍外圈饱和可降到 1.5
    parser.add_argument('--graph_space', type=str, default='euclidean')  # 图编码器传播空间，'euclidean' 更稳，'tangent_hyperbolic' 几何更一致；推荐先用 'euclidean'

    parser.add_argument('--cone_loss_weight', type=float, default=0.08)  # lambda_cone；当前默认提高到 0.1，避免早期 q2path 过强时 cone 太快失效
    parser.add_argument('--path_loss_weight', type=float, default=0.055)  # lambda_path；当前默认降到 0.05，先压住多正标签对齐项对几何的过强牵引
    parser.add_argument('--cone_radius_margin', type=float, default=0.0)  # cone 违反的安全间隔；推荐先用 0.0，只有想让父子约束更严格时再调到 0.01~0.05
    parser.add_argument('--cone_min_radius', type=float, default=0.03)  # cone 顶点最小半径；默认降到 0.03，避免小半径标签时父锥过宽、cone 过早归零
    parser.add_argument('--hyperbolic_lr', type=float, default=3e-5)  # 双曲头学习率；0.0 表示回退到 --lr，推荐设为 5e-5 左右（若主 lr=1e-4）
    parser.add_argument('--graph_lr', type=float, default=1e-4)  # 图编码器学习率；0.0 表示回退到 --lr，推荐与 --lr 相同或略小，如 1e-4 或 5e-5
    parser.add_argument('--curvature_lr', type=float, default=5e-6)  # 曲率参数学习率；0.0 表示自动设为 min(lr, hyperbolic_lr)*0.1，推荐显式设 1e-5

    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--swanlab', default=False, action='store_true')
    

    return parser


class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)

def get_exponential_with_warmup_scheduler(optimizer, warmup_steps, total_steps, gamma):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            decay_steps = (current_step - warmup_steps) // warmup_steps
            return gamma ** decay_steps
    return LambdaLR(optimizer, lr_lambda)


def get_dataset_name(data_path):
    return os.path.basename(os.path.normpath(data_path))


def load_label_maps(data_path):
    raw_label_dict = utils.torch_load_compat(os.path.join(data_path, 'value_dict.pt'))
    if len(raw_label_dict) == 0:
        return {}, {}

    sample_key = next(iter(raw_label_dict.keys()))
    if isinstance(sample_key, str):
        label2id = {str(k): int(v) for k, v in raw_label_dict.items()}
        id2label = {int(v): str(k) for k, v in raw_label_dict.items()}
    else:
        id2label = {int(k): v for k, v in raw_label_dict.items()}
        label2id = {v: int(k) for k, v in id2label.items()}

    return id2label, label2id


def build_raw_data_files(data_path, dataset_name):
    if dataset_name == 'bgc':
        return {
            'train': os.path.join(data_path, 'train_data.jsonl'),
            'dev': os.path.join(data_path, 'dev_data.jsonl'),
            'test': os.path.join(data_path, 'test_data.jsonl'),
        }

    return {
        'train': os.path.join(data_path, '{}_train.json'.format(dataset_name)),
        'dev': os.path.join(data_path, '{}_dev.json'.format(dataset_name)),
        'test': os.path.join(data_path, '{}_test.json'.format(dataset_name)),
    }


def normalize_label_ids(labels, label2id):
    label_ids = []
    for label in labels:
        if isinstance(label, str):
            label_ids.append(label2id[label])
        else:
            label_ids.append(int(label))
    return label_ids


def normalize_graph_space(graph_space):
    normalized = (graph_space or 'euclidean').strip().lower()
    mapping = {
        'euclidean': 'euclidean',
        'euc': 'euclidean',
        'tangent_hyperbolic': 'tangent_hyperbolic',
        'tangent': 'tangent_hyperbolic',
        'hyperbolic_tangent': 'tangent_hyperbolic',
    }
    if normalized not in mapping:
        supported = ', '.join(['euclidean', 'tangent_hyperbolic'])
        raise ValueError(f'Unsupported graph_space: {graph_space}. Supported values: {supported}.')
    return mapping[normalized]


def parse_curvature_init_by_depth(curvature_init_by_depth, num_depths):
    if curvature_init_by_depth is None:
        return None
    if isinstance(curvature_init_by_depth, (list, tuple)):
        values = [float(item) for item in curvature_init_by_depth]
    else:
        text = str(curvature_init_by_depth).strip()
        if not text:
            return None
        values = [float(item.strip()) for item in text.split(',') if item.strip()]
    if not values:
        return None
    if len(values) == 1:
        return values * num_depths
    if len(values) != num_depths:
        raise ValueError(
            f'curvature_init_by_depth length {len(values)} must be 1 or equal to num_depths {num_depths}.'
        )
    return values

if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    args.graph = utils.normalize_graph_type(args.graph)
    args.graph_space = normalize_graph_space(args.graph_space)
    print(args)
    utils.seed_torch(args.seed)

    requested_device = str(args.device)
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))

    if args.swanlab:
        swanlab.init(config=args, project='IPSA-GOF')
    logger = utils.init_logger(os.path.join('checkpoints', args.name, 'run.log'))
    logger.info(args)
    if args.device != requested_device:
        logger.info('requested device {} is unavailable; fallback to {}'.format(requested_device, args.device))
    batch_size = args.batch

    label_dict, label2id = load_label_maps(data_path)
    num_class = len(label_dict)
    slot2value = utils.torch_load_compat(os.path.join(data_path, 'slot.pt'))
    children_by_parent = {}
    for parent, children in slot2value.items():
        parent_id = int(parent)
        if parent_id >= num_class:
            continue
        filtered_children = []
        for child in children:
            child_id = int(child)
            if child_id < num_class:
                filtered_children.append(child_id)
        children_by_parent[parent_id] = filtered_children
    value2slot = {}
    # num_class = 0
    for s in slot2value:
        if s >= num_class:
            continue
        for v in slot2value[s]:
            # value2slot[v] = s
            # if num_class < v:
            #     num_class = v
            # 过滤掉无效的子节点索引
            if v < num_class:
                value2slot[v] = s

    # num_class += 1
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1


    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth


    depth_dict = {i: get_depth(i) for i in range(num_class)} 
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    curvature_init_by_depth = parse_curvature_init_by_depth(args.curvature_init_by_depth, max_depth)
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
    direct_cone_edges = [(value2slot[child], child) for child in range(num_class) if value2slot[child] != -1]
    ancestor_closure_by_label = {}
    for label_id in range(num_class):
        closure = []
        current = label_id
        while current != -1:
            closure.append(current)
            current = value2slot[current]
        ancestor_closure_by_label[label_id] = closure
    cone_edges = []
    for child in range(num_class):
        for ancestor in ancestor_closure_by_label[child][1:]:
            cone_edges.append((ancestor, child))

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))
    
    logger.info('num_class: {}'.format(num_class))
    logger.info('label dict: {}'.format(label_dict))
    logger.info('slot2value: {}'.format(slot2value))
    logger.info('value2slot: {}'.format(value2slot))
    logger.info('depth2label: {}'.format(depth2label))
    logger.info('path_list: {}'.format(path_list))

    if args.model == 'prompt':
        cache_dir = utils.get_prompt_cache_dir(data_path, args.model, args.arch)
        prefix = [] 
        for i in range(max_depth):
            prefix.append(tokenizer.vocab_size + num_class + i)
            prefix.append(tokenizer.vocab_size + num_class + max_depth)
        prefix.append(tokenizer.sep_token_id) # prefix = [30663, 30665, 30664, 30665, 102]

        def build_path_labels(label_ids):
            path_labels = [[0 for _ in range(num_class)] for _ in range(max_depth)]
            expanded_labels = set()
            for label_id in label_ids:
                expanded_labels.update(ancestor_closure_by_label[int(label_id)])
            for label_id in expanded_labels:
                path_labels[depth_dict[label_id]][label_id] = 1
            return [x for y in path_labels for x in y]

        def data_map_function(batch, tokenizer): 
            new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': [], 'path_labels': []}
            for l, t in zip(batch['label'], batch['token']):
                label_ids = normalize_label_ids(l, label2id)
                new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)]) 
                for d in range(max_depth): 
                    for i in depth2label[d]: 
                        new_batch['labels'][-1][d][i] = 0
                    for i in label_ids:
                        if new_batch['labels'][-1][d][i] == 0:
                            new_batch['labels'][-1][d][i] = 1
                new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y] 
                new_batch['path_labels'].append(build_path_labels(label_ids))

                tokens = tokenizer(t, truncation=True)
                new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix) 
                new_batch['input_ids'][-1].extend(
                    [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1]))) 
                new_batch['attention_mask'].append(
                    tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                new_batch['token_type_ids'].append([0] * 512)

            return new_batch

        def path_label_map_function(batch):
            return {
                'path_labels': [
                    build_path_labels(normalize_label_ids(label_list, label2id))
                    for label_list in batch['label']
                ]
            }

        if args.data != 'bgc' and os.path.exists(cache_dir):
            dataset = datasets.load_from_disk(cache_dir)
            if 'path_labels' not in dataset['train'].column_names:
                if 'label' not in dataset['train'].column_names:
                    logger.info('cached dataset missing raw labels; rebuilding processed dataset with path_labels')
                    dataset = datasets.load_dataset('json', data_files=build_raw_data_files(data_path, args.data))
                    dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
                    dataset.save_to_disk(cache_dir)
                else:
                    logger.info('cached dataset missing path_labels; augmenting in memory')
                    dataset = dataset.map(path_label_map_function, batched=True)
        else:
            dataset = datasets.load_dataset('json', data_files=build_raw_data_files(data_path, args.data))
            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            if args.data != 'bgc':
                dataset.save_to_disk(cache_dir)
        label_freq = utils.compute_label_frequency(dataset['train']['label'], label2id)
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'path_labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        logger.info("train_data num is: {}".format(len(dataset['train'])))
        logger.info("dev_data num is: {}".format(len(dataset['dev'])))
        logger.info("test_data num is: {}".format(len(dataset['test'])))

        from models.prompt import Prompt

    else:
        raise NotImplementedError
    if args.low_res:
        if os.path.exists(os.path.join(data_path, 'low.json')):
            index = json.load(open(os.path.join(data_path, 'low.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(data_path, 'low.json'), 'w'))
        dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])
    model = Prompt.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, orth_method=args.orth_method,
                                   mlm_mask_strategy=args.mlm_mask_strategy,
                                   classifier_head=args.classifier_head,
                                   hyperbolic_dim=args.hyperbolic_dim,
                                   hyperbolic_alpha=args.hyperbolic_alpha,
                                   depth_aware_hyper_alpha=args.depth_aware_hyper_alpha,
                                   hyperbolic_curvature_init=args.hyperbolic_curvature_init,
                                   curvature_init_by_depth=curvature_init_by_depth,
                                   per_depth_curvature=args.per_depth_curvature,
                                   hyperbolic_logit_scale_init=args.hyperbolic_logit_scale_init,
                                   hyperbolic_radius_clip=args.hyperbolic_radius_clip,
                                   graph_space=args.graph_space,
                                   data_path=data_path, depth2label=depth2label) 
    model.init_embedding()
    logger.info(model)
    logger.info('backbone: {}, graph: {}, graph_space: {}, orth_method: {}, mlm_mask_strategy: {}, effective_mlm_mask_strategy: {}, classifier_head: {}, hyperbolic_dim: {}, hyperbolic_alpha: {}, depth_aware_hyper_alpha: {}, per_depth_curvature: {}, curvature_init_by_depth: {}'.format(
        model.bert.__class__.__name__, args.graph, args.graph_space, args.orth_method, args.mlm_mask_strategy,
        model.get_effective_mlm_mask_strategy(), model.get_effective_classifier_head(),
        args.hyperbolic_dim, args.hyperbolic_alpha, args.depth_aware_hyper_alpha, args.per_depth_curvature, curvature_init_by_depth
    ))
    logger.info(
        'structure losses: cone_weight: {}, path_weight: {}, cone_radius_margin: {}, cone_min_radius: {}, hyperbolic_lr: {}, graph_lr: {}, curvature_lr: {}'.format(
            args.cone_loss_weight,
            args.path_loss_weight,
            args.cone_radius_margin,
            args.cone_min_radius,
            args.hyperbolic_lr,
            args.graph_lr,
            args.curvature_lr,
        )
    )
    logger.info('cone scope: all ancestor-descendant pairs for training; direct parent-child metrics logged separately')
    logger.info('active objective: L = L_mlm + L_cls + lambda_cone * L_cone + lambda_path * L_q2path')
    logger.info('hyperbolic structure: cone uses structural label prototypes; q2path uses a semantic-structure blended label target; classifier hyperplanes remain decoupled')
    logger.info(f"Total params: {sum(param.numel() for param in model.parameters()) / 1000000.0}M. ")

    model.to(device)
    if (args.cone_loss_weight > 0 or args.path_loss_weight > 0) and not hasattr(model, 'hyperbolic_head'):
        raise ValueError('cone_loss_weight > 0 or path_loss_weight > 0 requires classifier_head to be hyperbolic or hybrid.')

    structure_device = model.device
    if direct_cone_edges:
        direct_cone_parent_indices = torch.tensor(
            [parent for parent, _ in direct_cone_edges],
            device=structure_device,
            dtype=torch.long,
        )
        direct_cone_child_indices = torch.tensor(
            [child for _, child in direct_cone_edges],
            device=structure_device,
            dtype=torch.long,
        )
    else:
        direct_cone_parent_indices = torch.empty(0, device=structure_device, dtype=torch.long)
        direct_cone_child_indices = torch.empty(0, device=structure_device, dtype=torch.long)

    if cone_edges:
        cone_parent_indices = torch.tensor([parent for parent, _ in cone_edges], device=structure_device, dtype=torch.long)
        cone_child_indices = torch.tensor([child for _, child in cone_edges], device=structure_device, dtype=torch.long)
    else:
        cone_parent_indices = torch.empty(0, device=structure_device, dtype=torch.long)
        cone_child_indices = torch.empty(0, device=structure_device, dtype=torch.long)

    depth_label_tensors = [
        torch.tensor(depth2label[depth_idx], device=structure_device, dtype=torch.long)
        for depth_idx in range(max_depth)
    ]
    label_parent_indices = torch.tensor(
        [value2slot[label_id] for label_id in range(num_class)],
        device=structure_device,
        dtype=torch.long,
    )
    label_depth_ids = torch.tensor(
        [depth_dict[label_id] for label_id in range(num_class)],
        device=structure_device,
        dtype=torch.long,
    )
    logger.info(
        'structure graph stats: direct_cone_edges: {}, ancestor_cone_edges: {}'.format(
            len(direct_cone_edges),
            len(cone_edges),
        )
    )

    def build_optimizer_param_groups():
        embedding_layer = model.get_input_embeddings()
        graph_module = getattr(embedding_layer, 'graph', None) if hasattr(embedding_layer, 'graph') else None
        hyperbolic_head_module = getattr(model, 'hyperbolic_head', None)

        graph_param_ids = set()
        if graph_module is not None:
            graph_param_ids = {id(param) for param in graph_module.parameters() if param.requires_grad}

        hyper_param_ids = set()
        if hyperbolic_head_module is not None:
            hyper_param_ids = {id(param) for param in hyperbolic_head_module.parameters() if param.requires_grad}

        curvature_param_ids = set()
        for name, param in model.named_parameters():
            if param.requires_grad and 'raw_curvature' in name:
                curvature_param_ids.add(id(param))

        depth_alpha_param_ids = set()
        if hasattr(model, 'raw_hyperbolic_alpha_by_depth'):
            depth_alpha_param_ids = {id(model.raw_hyperbolic_alpha_by_depth)}

        branch_scale_param_ids = set()
        for attr_name in ('raw_euclidean_branch_scale_by_depth', 'raw_hyperbolic_branch_scale_by_depth'):
            if hasattr(model, attr_name):
                branch_scale_param_ids.add(id(getattr(model, attr_name)))

        backbone_params = []
        graph_params = []
        hyper_params = []
        curvature_params = []

        for _, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in curvature_param_ids:
                curvature_params.append(param)
            elif param_id in depth_alpha_param_ids or param_id in branch_scale_param_ids or param_id in hyper_param_ids:
                hyper_params.append(param)
            elif param_id in graph_param_ids:
                graph_params.append(param)
            else:
                backbone_params.append(param)

        hyperbolic_lr = args.hyperbolic_lr if args.hyperbolic_lr > 0 else args.lr
        graph_lr = args.graph_lr if args.graph_lr > 0 else args.lr
        curvature_lr = args.curvature_lr if args.curvature_lr > 0 else min(args.lr, hyperbolic_lr) * 0.1

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': args.lr, 'group_name': 'backbone'})
        if graph_params:
            param_groups.append({'params': graph_params, 'lr': graph_lr, 'group_name': 'graph'})
        if hyper_params:
            param_groups.append({'params': hyper_params, 'lr': hyperbolic_lr, 'group_name': 'hyperbolic'})
        if curvature_params:
            param_groups.append({'params': curvature_params, 'lr': curvature_lr, 'group_name': 'curvature'})
        return param_groups

    if args.wandb:
        wandb.watch(model) 

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    model.to(device)
    optimizer_param_groups = build_optimizer_param_groups()
    optimizer = Adam(optimizer_param_groups, lr=args.lr)
    logger.info('optimizer groups: {}'.format(
        ', '.join(
            '{}(lr={}, params={})'.format(
                group['group_name'],
                group['lr'],
                sum(param.numel() for param in group['params']),
            )
            for group in optimizer_param_groups
        )
    ))
    lr_initial = args.lr
    lr_final = 1e-7
    gamma49 = (lr_final / lr_initial) ** (1 / 49)
    if args.data == 'WebOfScience':
        warmup_steps = (30070 // args.batch) + 1
        total_steps = warmup_steps * 50
    elif args.data == 'rcv1':
        warmup_steps = (20834 // args.batch) + 1
        total_steps = warmup_steps * 50
    elif args.data == 'bgc':
        warmup_steps = (len(dataset['train']) // args.batch) + 1
        total_steps = warmup_steps * 50
    else: # nyt
        warmup_steps = (23391 // args.batch) + 1
        total_steps = warmup_steps * 50
    scheduler = get_exponential_with_warmup_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, gamma=gamma49)



    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    # early_stop_count = 0
    update_step = 0
    loss = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))
    
    def numParams(net):
        num = 0
        for param in net.parameters():
            if param.requires_grad:
                num += int(np.prod(param.size()))
        return num

    def compute_geometry_diagnostics(output, path_aux=None):
        if not hasattr(output, 'hyperbolic_query_tangent_by_depth'):
            return {}

        diagnostics = {}
        clip_value = float(args.hyperbolic_radius_clip)
        near_cap_threshold = 0.95 * clip_value if clip_value > 0 else None

        raw_label_tangent = getattr(output, 'hyperbolic_raw_label_tangent', output.hyperbolic_label_tangent)
        raw_query_tangent_by_depth = getattr(
            output,
            'hyperbolic_raw_query_tangent_by_depth',
            output.hyperbolic_query_tangent_by_depth,
        )
        raw_label_radius = getattr(output, 'hyperbolic_raw_label_radius', None)
        if raw_label_radius is None:
            raw_label_radius = torch.norm(raw_label_tangent, dim=-1)
        raw_query_radius = getattr(output, 'hyperbolic_raw_query_radius_by_depth', None)
        if raw_query_radius is None:
            raw_query_radius = torch.norm(raw_query_tangent_by_depth, dim=-1)
        label_radius = getattr(output, 'hyperbolic_label_radius', None)
        if label_radius is None:
            label_radius = torch.norm(output.hyperbolic_label_tangent, dim=-1)
        query_radius = getattr(output, 'hyperbolic_query_radius_by_depth', None)
        if query_radius is None:
            query_radius = torch.norm(output.hyperbolic_query_tangent_by_depth, dim=-1)
        label_radius_floor = getattr(output, 'hyperbolic_label_radius_floor', None)
        raw_label_radius_residual = getattr(output, 'hyperbolic_raw_label_radius_residual', None)
        query_radius_floor = getattr(output, 'hyperbolic_query_radius_floor_by_depth', None)
        raw_query_radius_residual = getattr(output, 'hyperbolic_raw_query_radius_residual_by_depth', None)

        for depth_idx, depth_labels in enumerate(depth_label_tensors):
            if depth_labels.numel() == 0:
                continue
            depth_raw_label_radius = raw_label_radius.index_select(0, depth_labels)
            depth_label_radius = label_radius.index_select(0, depth_labels)
            diagnostics[f'raw_label_radius_mean_d{depth_idx}'] = depth_raw_label_radius.mean().item()
            diagnostics[f'raw_label_radius_std_d{depth_idx}'] = depth_raw_label_radius.std(unbiased=False).item()
            diagnostics[f'label_radius_mean_d{depth_idx}'] = depth_label_radius.mean().item()
            diagnostics[f'label_radius_std_d{depth_idx}'] = depth_label_radius.std(unbiased=False).item()
            if label_radius_floor is not None:
                diagnostics[f'label_radius_floor_mean_d{depth_idx}'] = (
                    label_radius_floor.index_select(0, depth_labels).mean().item()
                )
            if raw_label_radius_residual is not None:
                diagnostics[f'raw_label_radius_residual_mean_d{depth_idx}'] = (
                    raw_label_radius_residual.index_select(0, depth_labels).mean().item()
                )

            depth_raw_query_radius = raw_query_radius[:, depth_idx]
            depth_query_radius = query_radius[:, depth_idx]
            diagnostics[f'raw_query_radius_mean_d{depth_idx}'] = depth_raw_query_radius.mean().item()
            diagnostics[f'raw_query_radius_std_d{depth_idx}'] = depth_raw_query_radius.std(unbiased=False).item()
            diagnostics[f'query_radius_mean_d{depth_idx}'] = depth_query_radius.mean().item()
            diagnostics[f'query_radius_std_d{depth_idx}'] = depth_query_radius.std(unbiased=False).item()
            if query_radius_floor is not None:
                diagnostics[f'query_radius_floor_mean_d{depth_idx}'] = (
                    query_radius_floor[:, depth_idx].mean().item()
                )
            if raw_query_radius_residual is not None:
                diagnostics[f'raw_query_radius_residual_mean_d{depth_idx}'] = (
                    raw_query_radius_residual[:, depth_idx].mean().item()
                )
            diagnostics[f'curvature_d{depth_idx}'] = output.hyperbolic_curvature_by_depth[depth_idx].item()
            if hasattr(output, 'hyperbolic_raw_query_radius_delta_by_depth'):
                diagnostics[f'raw_query_radius_delta_mean_d{depth_idx}'] = (
                    output.hyperbolic_raw_query_radius_delta_by_depth[:, depth_idx].mean().item()
                )
            if hasattr(output, 'hyperbolic_raw_query_direction_by_depth'):
                diagnostics[f'query_direction_norm_mean_d{depth_idx}'] = (
                    torch.norm(output.hyperbolic_raw_query_direction_by_depth[:, depth_idx, :], dim=-1).mean().item()
                )

            if clip_value > 0:
                diagnostics[f'label_radius_bound_active_ratio_d{depth_idx}'] = (
                    depth_raw_label_radius > clip_value
                ).float().mean().item()
                diagnostics[f'query_radius_bound_active_ratio_d{depth_idx}'] = (
                    depth_raw_query_radius > clip_value
                ).float().mean().item()
                diagnostics[f'label_radius_near_cap_ratio_d{depth_idx}'] = (
                    depth_label_radius >= near_cap_threshold
                ).float().mean().item()
                diagnostics[f'query_radius_near_cap_ratio_d{depth_idx}'] = (
                    depth_query_radius >= near_cap_threshold
                ).float().mean().item()

        if query_radius.size(1) > 1:
            monotonic_violation = torch.clamp(query_radius[:, :-1] - query_radius[:, 1:], min=0.0)
            diagnostics['query_radius_monotonic_violation_mean'] = monotonic_violation.mean().item()

        if hasattr(output, 'hyperbolic_alpha_by_depth'):
            for depth_idx, alpha in enumerate(output.hyperbolic_alpha_by_depth):
                diagnostics[f'hyper_alpha_d{depth_idx}'] = alpha.item()
        if hasattr(output, 'euclidean_branch_scale_by_depth'):
            for depth_idx, scale in enumerate(output.euclidean_branch_scale_by_depth):
                diagnostics[f'euclidean_branch_scale_d{depth_idx}'] = scale.item()
        if hasattr(output, 'hyperbolic_branch_scale_by_depth'):
            for depth_idx, scale in enumerate(output.hyperbolic_branch_scale_by_depth):
                diagnostics[f'hyperbolic_branch_scale_d{depth_idx}'] = scale.item()
        if hasattr(output, 'hyperbolic_logit_scale'):
            diagnostics['hyperbolic_logit_scale'] = output.hyperbolic_logit_scale.item()

        if direct_cone_parent_indices.numel() > 0:
            direct_parent_points = output.hyperbolic_label_points.index_select(0, direct_cone_parent_indices)
            direct_child_points = output.hyperbolic_label_points.index_select(0, direct_cone_child_indices)
            direct_parent_curvatures = output.hyperbolic_label_curvature.index_select(0, direct_cone_parent_indices)
            parent_child_distance = lorentz_distance(
                direct_parent_points,
                direct_child_points,
                curv=direct_parent_curvatures,
            )
            diagnostics['parent_child_dist_mean'] = parent_child_distance.mean().item()
            diagnostics['parent_child_dist_std'] = parent_child_distance.std(unbiased=False).item()

            direct_cone_violation, direct_cone_angle, direct_cone_aperture = exact_entailment_cone_violation(
                direct_parent_points,
                direct_child_points,
                curv=direct_parent_curvatures,
                min_radius=args.cone_min_radius,
                margin=args.cone_radius_margin,
            )
            diagnostics['direct_cone_violation_ratio'] = (direct_cone_violation > 0).float().mean().item()
            diagnostics['direct_cone_violation_mean'] = direct_cone_violation.mean().item()
            diagnostics['direct_cone_angle_mean'] = direct_cone_angle.mean().item()
            diagnostics['direct_cone_aperture_mean'] = direct_cone_aperture.mean().item()
            diagnostics['direct_cone_slack_mean'] = (direct_cone_aperture - direct_cone_angle).mean().item()

        if cone_parent_indices.numel() > 0:
            parent_points = output.hyperbolic_label_points.index_select(0, cone_parent_indices)
            child_points = output.hyperbolic_label_points.index_select(0, cone_child_indices)
            parent_curvatures = output.hyperbolic_label_curvature.index_select(0, cone_parent_indices)
            cone_violation, cone_angle, cone_aperture = exact_entailment_cone_violation(
                parent_points,
                child_points,
                curv=parent_curvatures,
                min_radius=args.cone_min_radius,
                margin=args.cone_radius_margin,
            )
            diagnostics['cone_violation_ratio'] = (cone_violation > 0).float().mean().item()
            diagnostics['cone_violation_mean'] = cone_violation.mean().item()
            diagnostics['cone_angle_mean'] = cone_angle.mean().item()
            diagnostics['cone_aperture_mean'] = cone_aperture.mean().item()
            diagnostics['cone_slack_mean'] = (cone_aperture - cone_angle).mean().item()

        if path_aux is not None:
            scalar_keys = (
                'alignment_loss',
                'query_chain_loss',
                'query_label_align_dist_mean',
                'query_label_min_dist_mean',
                'query_chain_violation_mean',
                'valid_depth_ratio',
                'positive_count_mean',
                'path_count_mean',
                'path_length_mean',
            )
            for key in scalar_keys:
                if key in path_aux:
                    diagnostics[key] = float(path_aux[key].item() if hasattr(path_aux[key], 'item') else path_aux[key])

            if 'query_label_min_dist_mean' in path_aux:
                diagnostics['query_gold_path_dist_mean'] = diagnostics['query_label_min_dist_mean']
            elif 'query_label_align_dist_mean' in path_aux:
                diagnostics['query_gold_path_dist_mean'] = diagnostics['query_label_align_dist_mean']

            valid_depth_mask = path_aux.get('valid_depth_mask')
            positive_count_by_depth = path_aux.get('positive_count_by_depth')
            alignment_mean_by_depth = path_aux.get('alignment_mean_by_depth')
            min_distance_mean_by_depth = path_aux.get('min_distance_mean_by_depth')
            if valid_depth_mask is not None:
                for depth_idx in range(valid_depth_mask.size(1)):
                    diagnostics[f'valid_depth_ratio_d{depth_idx}'] = (
                        valid_depth_mask[:, depth_idx].float().mean().item()
                    )
            if positive_count_by_depth is not None and valid_depth_mask is not None:
                for depth_idx in range(valid_depth_mask.size(1)):
                    valid_samples = valid_depth_mask[:, depth_idx]
                    if not valid_samples.any():
                        continue
                    diagnostics[f'positive_count_mean_d{depth_idx}'] = (
                        positive_count_by_depth[valid_samples, depth_idx].mean().item()
                    )
            if alignment_mean_by_depth is not None:
                for depth_idx, depth_value in enumerate(alignment_mean_by_depth):
                    if float(depth_value) > 0:
                        diagnostics[f'alignment_softmin_mean_d{depth_idx}'] = float(depth_value.item())
            if min_distance_mean_by_depth is not None:
                for depth_idx, depth_value in enumerate(min_distance_mean_by_depth):
                    if float(depth_value) > 0:
                        diagnostics[f'query_positive_min_dist_mean_d{depth_idx}'] = float(depth_value.item())

            edge_mask = path_aux.get('valid_edge_mask')
            if edge_mask is not None:
                diagnostics['valid_query_chain_edge_ratio'] = edge_mask.float().mean().item()

        return diagnostics

    print("numParams:", numParams(model))

    for epoch in range(args.epoch):
        logger.info("------------ epoch {} ------------".format(epoch + 1))
        epoch_geometry_sums = {}
        epoch_geometry_count = 0
        start_time = time.time()
        print("start_time:", start_time)
        model.train()
        with tqdm(train) as p_bar:
            for batch in p_bar: # batch包含'input_ids''attention_mask''labels'
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'labels': batch['labels'],
                }
                output, masked_lm_loss, multiclass_loss = model(**model_inputs) 

                loss_total = output['loss']
                cone_loss = loss_total.new_zeros(())
                cone_violation_ratio = loss_total.new_zeros(())
                path_tree_loss = loss_total.new_zeros(())
                path_aux = None
                if args.cone_loss_weight > 0:
                    cone_loss, cone_violation_ratio = label_cone_loss(
                        output.hyperbolic_label_points,
                        cone_parent_indices,
                        cone_child_indices,
                        curv=output.hyperbolic_label_curvature.index_select(0, cone_parent_indices),
                        margin=args.cone_radius_margin,
                        min_radius=args.cone_min_radius,
                    )
                    loss_total = loss_total + args.cone_loss_weight * cone_loss
                if args.path_loss_weight > 0:
                    path_tree_loss, path_aux = query_to_label_tree_alignment_loss(
                        output.hyperbolic_query_tangent_by_depth,
                        output.hyperbolic_query_points_by_depth,
                        output.hyperbolic_alignment_label_points,
                        batch['path_labels'],
                        depth_label_tensors,
                        label_parent_indices=label_parent_indices,
                        label_depth_ids=label_depth_ids,
                        depth_curvatures=output.hyperbolic_curvature_by_depth,
                        margin=args.cone_radius_margin,
                        min_radius=args.cone_min_radius,
                    )
                    loss_total = loss_total + args.path_loss_weight * path_tree_loss

                loss_total.backward()
                loss += loss_total.item()
                update_step += 1
                batch_geometry = compute_geometry_diagnostics(output, path_aux=path_aux)
                for key, value in batch_geometry.items():
                    epoch_geometry_sums[key] = epoch_geometry_sums.get(key, 0.0) + float(value)
                epoch_geometry_count += 1
                if update_step % args.update == 0:
                    if args.swanlab:
                        swanlab.log({
                            'loss': loss,
                            'masked_lm_loss': masked_lm_loss.item(),
                            'multiclass_loss': multiclass_loss.item(),
                            'cone_loss': cone_loss.item(),
                            'cone_violation_ratio': cone_violation_ratio.item(),
                            'path_tree_loss': path_tree_loss.item(),
                            **batch_geometry,
                        })

                    if args.cone_loss_weight > 0 or args.path_loss_weight > 0:
                        p_bar.set_description(
                            'loss:{:.4f} cone:{:.4f} vr:{:.2f} q2path:{:.4f}'.format(
                                loss,
                                cone_loss.item(),
                                cone_violation_ratio.item(),
                                path_tree_loss.item(),
                            ))
                    else:
                        p_bar.set_description(
                            'loss:{:.4f}'.format(loss, ))
                    optimizer.step()
                    scheduler.step()  
                    optimizer.zero_grad()
                    loss = 0
                    update_step = 0

        end_time = time.time()
        print("end_time:", end_time)
        elapsed_train_time = end_time - start_time
        print(f"epoch train time:{elapsed_train_time}s")
        if epoch_geometry_count > 0:
            averaged_geometry = {
                key: value / epoch_geometry_count
                for key, value in sorted(epoch_geometry_sums.items())
            }
            logger.info('[geometry] {}'.format(utils.format_metrics(averaged_geometry)))

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev) as pbar:
            start_time = time.time()
            print("start_time:", start_time)
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, ) # output_ids是每个batch预测得到的标签，logits(8,2,141)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
            end_time = time.time()
            print("end_time:", end_time)
            elapsed_eval_time = end_time - start_time
            print(f"epoch eval time:{elapsed_eval_time}s")

            # if args.wandb:
            #     wandb.log({'label_ausc': np.sum(label_s)})

        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        logger.info(' macro: {:.4f}, micro: {:.4f}'.format(macro_f1, micro_f1))
        print('macro', macro_f1, 'micro', micro_f1)

        if args.swanlab:
            swanlab.log({'val_macro': macro_f1, 'val_micro': micro_f1})


        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            logger.info(f"New best macro F1: {best_score_macro:.4f}. Checkpoint saved.")
            # early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            logger.info(f"New best micro F1: {best_score_micro:.4f}. Checkpoint saved.")

        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
        if args.swanlab:
            swanlab.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # test
    test = DataLoader(dataset['test'], batch_size=32, shuffle=False)
    model.eval()


    def test_function(extra):
        checkpoint = utils.torch_load_compat(
            os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra)),
            map_location='cpu',
        )
        logger.info(f'Test load checkpoint: {checkpoint}')
        model.load_state_dict(checkpoint['param'])
        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)

        metric_prefix = 'test' + extra
        evaluation = utils.run_detailed_evaluation(
            pred,
            gold,
            label_dict,
            value2slot,
            slot2value,
            label_freq,
            depth2label,
            logger=logger,
            prefix=metric_prefix,
        )
        scores = evaluation['standard']
        path_scores = evaluation['path']
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        print("---------------------")
        print(utils.format_metrics(
            path_scores,
            preferred_keys=[
                'P_acc',
                'p_precision',
                'p_recall',
                'p_micro_f1',
                'p_macro_f1',
                'p_ori_micro_f1',
                'p_ori_macro_f1',
                'c_precision',
                'c_recall',
                'c_micro_f1',
                'c_macro_f1',
            ],
        ))

        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            for line in evaluation['summary_lines']:
                print(line, file=f)
            for line in evaluation['analysis']['report_lines']:
                print(line, file=f)
        if args.swanlab:
            swanlab.log({
                metric_prefix + '_macro': macro_f1,
                metric_prefix + '_micro': micro_f1,
                metric_prefix + '_p_micro_f1': path_scores['p_micro_f1'],
                metric_prefix + '_p_macro_f1': path_scores['p_macro_f1'],
                metric_prefix + '_c_micro_f1': path_scores['c_micro_f1'],
                metric_prefix + '_c_macro_f1': path_scores['c_macro_f1'],
                metric_prefix + '_path_acc': path_scores['P_acc'],
            })


    test_function('_macro')
    test_function('_micro')

    wandb.finish()
