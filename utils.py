import hashlib
import logging
import numbers
import os
import random
import re

import numpy as np
import torch


def seed_torch(seed=1029):
    print('Set seed to', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def torch_load_compat(*args, **kwargs):
    """Preserve pre-2.6 torch.load behavior for trusted local artifacts."""
    kwargs.setdefault('weights_only', False)
    try:
        return torch.load(*args, **kwargs)
    except TypeError:
        kwargs.pop('weights_only', None)
        return torch.load(*args, **kwargs)


def constraint(batch_id, input_ids, label_dict):
    last_token = input_ids[-1].item()
    if last_token not in label_dict:
        ret = [2]
    else:
        ret = [i + 3 for i in label_dict[input_ids[-1].item() - 3]]
    return ret

def init_logger(log_path):
    logging.basicConfig(filemode='w')
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8', mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s -   %(message)s', 
                                  datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_label_dict(data_path):
    for file_name in ('value_dict.pt', 'label_dict.pt'):
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            raw_label_dict = torch_load_compat(file_path)
            if len(raw_label_dict) == 0:
                return {}

            id_to_label = {}
            label_to_id = {}
            for key, value in raw_label_dict.items():
                try:
                    id_to_label[int(key)] = str(value)
                    continue
                except (TypeError, ValueError):
                    pass

                try:
                    label_to_id[int(value)] = str(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'Unsupported label dictionary format in {file_path}: '
                        f'expected id->label or label->id entries, got {key!r}->{value!r}.'
                    ) from exc

            return id_to_label if id_to_label else label_to_id
    raise FileNotFoundError(f'No label dictionary found under {data_path}.')


def compute_label_frequency(label_sequences, label2id):
    label_freq = {}
    for labels in label_sequences:
        for label in labels:
            if isinstance(label, str):
                label_id = label2id[label]
            else:
                label_id = int(label)
            label_freq[label_id] = label_freq.get(label_id, 0) + 1
    return label_freq


def _is_scalar_metric(value):
    return isinstance(value, (numbers.Real, np.integer, np.floating))


def format_metrics(metrics, preferred_keys=None, excluded_keys=('full',)):
    ordered_keys = []
    seen = set(excluded_keys)

    if preferred_keys is not None:
        for key in preferred_keys:
            if key in metrics and key not in seen and _is_scalar_metric(metrics[key]):
                ordered_keys.append(key)
                seen.add(key)

    for key, value in metrics.items():
        if key in seen or not _is_scalar_metric(value):
            continue
        ordered_keys.append(key)
        seen.add(key)

    parts = []
    for key in ordered_keys:
        value = metrics[key]
        if isinstance(value, (float, np.floating)):
            parts.append(f'{key}: {float(value):.4f}')
        elif isinstance(value, (int, np.integer)):
            parts.append(f'{key}: {int(value)}')
        else:
            parts.append(f'{key}: {value}')
    return ', '.join(parts)


def log_metrics(logger, title, metrics, preferred_keys=None, excluded_keys=('full',)):
    line = format_metrics(metrics, preferred_keys=preferred_keys, excluded_keys=excluded_keys)
    if logger is not None:
        logger.info(f'{title}: {line}')
    return f'{title}: {line}'


def run_detailed_evaluation(
    predictions,
    gold_labels,
    id2label,
    value2slot,
    slot2value,
    label_freq,
    depth2label,
    logger=None,
    prefix='test',
):
    from eval import evaluate, evaluate_based_on_path
    from label_analysis import analyze_labels

    gold_sets = [set(labels) for labels in gold_labels]
    standard_scores = evaluate(predictions, gold_labels, id2label)
    path_scores = evaluate_based_on_path(predictions, gold_labels, id2label, value2slot, slot2value)

    standard_line = log_metrics(
        logger,
        f'[{prefix}] standard metrics',
        standard_scores,
        preferred_keys=['precision', 'recall', 'micro_f1', 'macro_f1'],
    )
    path_line = log_metrics(
        logger,
        f'[{prefix}] path metrics',
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
    )

    if logger is not None:
        logger.info(f'[{prefix}] label analysis report:')
    label_report = analyze_labels(
        predictions,
        gold_sets,
        label_freq,
        depth2label,
        id2label,
        logger=logger,
    )

    return {
        'standard': standard_scores,
        'path': path_scores,
        'analysis': label_report,
        'summary_lines': [standard_line, path_line],
    }


def normalize_graph_type(graph_type):
    normalized = (graph_type or '').strip().lower()
    mapping = {
        '': '',
        'none': '',
        'gat': 'GAT',
        'gcn': 'GCN',
        'graphormer': 'graphormer',
        'hypergraph': 'hypergraph',
    }
    if normalized not in mapping:
        supported = ', '.join(['GAT', 'GCN', 'graphormer', 'hypergraph'])
        raise ValueError(f'Unsupported graph type: {graph_type}. Supported values: {supported}.')
    return mapping[normalized]


def get_prompt_cache_dir(data_path, model_name, arch):
    arch_key = os.path.normpath(arch)
    base_name = os.path.basename(arch_key.rstrip('/\\')) or arch_key
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', base_name)
    digest = hashlib.md5(arch_key.encode('utf-8')).hexdigest()[:8]
    return os.path.join(data_path, f'{model_name}_{safe_name}_{digest}')
