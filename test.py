import torch
import os
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

from train import (
    parse,
    load_label_maps,
    build_raw_data_files,
    normalize_label_ids,
    normalize_graph_space,
    parse_curvature_init_by_depth,
)
import utils
import random
import numpy as np

if __name__ == '__main__':
    utils.seed_torch(3)
    parser = parse()
    parser.add_argument('--extra', type=str, default='_macro')
    args = parser.parse_args()

    checkpoint_dir = os.path.join('checkpoints', args.name)
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join('checkpoints', f'{args.data}-{args.name}')
    logger = utils.init_logger(os.path.join(checkpoint_dir, 'test{}.log'.format(args.extra)))
    checkpoint = utils.torch_load_compat(
        os.path.join(checkpoint_dir, 'checkpoint_best{}.pt'.format(args.extra)),
        map_location='cpu',
    )
    batch_size = args.batch
    data_path = args.data
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    args = parser.parse_args(namespace=args)
    args.graph = utils.normalize_graph_type(args.graph)
    args.graph_space = normalize_graph_space(args.graph_space)
    print(args)
    data_path = os.path.join('data', args.data if hasattr(args, 'data') else data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)

    label_dict, label2id = load_label_maps(data_path)

    slot2value = utils.torch_load_compat(os.path.join(data_path, 'slot.pt'))
    value2slot = {}
    for s in slot2value:
        if s >= len(label_dict):
            continue
        for v in slot2value[s]:
            if v < len(label_dict):
                value2slot[v] = s
    num_class = len(label_dict)
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

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))

    if args.model == 'prompt':
        cache_dir = utils.get_prompt_cache_dir(data_path, args.model, args.arch)
        if args.data != 'bgc' and os.path.exists(cache_dir):
            dataset = datasets.load_from_disk(cache_dir)
        else:
            dataset = datasets.load_dataset('json', data_files=build_raw_data_files(data_path, args.data))

            prefix = []
            for i in range(max_depth):
                prefix.append(tokenizer.vocab_size + num_class + i)
                prefix.append(tokenizer.vocab_size + num_class + max_depth)
            prefix.append(tokenizer.sep_token_id)


            def data_map_function(batch, tokenizer):
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
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

                    tokens = tokenizer(t, truncation=True)
                    new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix)
                    new_batch['input_ids'][-1].extend(
                        [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                    new_batch['attention_mask'].append(
                        tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                    new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                    new_batch['token_type_ids'].append([0] * 512)

                return new_batch


            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            if args.data != 'bgc':
                dataset.save_to_disk(cache_dir)
        label_freq = utils.compute_label_frequency(dataset['train']['label'], label2id)
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        from models.prompt import Prompt

    else:
        raise NotImplementedError

    checkpoint = utils.torch_load_compat(
        os.path.join(checkpoint_dir, 'checkpoint_best{}.pt'.format(extra)),
        map_location='cpu',
    )
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
    model.load_state_dict(checkpoint['param'])
    model.to('cuda')

    test = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    model.eval()
    pred = []
    gold = []
    father_count = 0
    father_false = 0

    with torch.no_grad(), tqdm(test) as pbar:
        for batch in pbar:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label,)
            for out, g in zip(output_ids, batch['labels']):
                for i in set(out):
                    if value2slot[i] != -1:
                        if value2slot[i] in out:
                            father_count += 1
                        else:
                            father_false += 1
                pred.append(set([i for i in out]))
                gold.append([])
                g = g.view(-1, num_class)
                for ll in g:
                    for i, l in enumerate(ll):
                        if l == 1:
                            gold[-1].append(i)

    parent_path_acc = father_count / (father_false + father_count) if (father_false + father_count) > 0 else 0.0
    logger.info('[test{}] parent_path_acc: {:.4f}'.format(extra, parent_path_acc))
    evaluation = utils.run_detailed_evaluation(
        pred,
        gold,
        label_dict,
        value2slot,
        slot2value,
        label_freq,
        depth2label,
        logger=logger,
        prefix='test' + extra,
    )
    scores = evaluation['standard']
    path_scores = evaluation['path']
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print('parent_path_acc', parent_path_acc)
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

    with open(os.path.join(checkpoint_dir, 'result{}.txt'.format(extra)), 'w') as f:
        print('parent_path_acc: {:.4f}'.format(parent_path_acc), file=f)
        for line in evaluation['summary_lines']:
            print(line, file=f)
        for line in evaluation['analysis']['report_lines']:
            print(line, file=f)
