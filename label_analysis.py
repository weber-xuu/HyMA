"""
Label Analysis Module for Long-Tail Evaluation

Provides detailed analysis of model performance across:
1. Per-label metrics (Precision, Recall, F1)
2. Frequency-based grouping (5 tiers: >80%, 60-80%, 40-60%, 20-40%, <20%)
3. Hierarchy-based grouping (per-level metrics)
"""

import numpy as np
from typing import List, Dict, Set, Optional


def _precision_recall_f1(tp, fp, fn):
    """Calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calculate_per_label_metrics(predictions: List[Set[int]], 
                                  gold_labels: List[Set[int]], 
                                  num_labels: int):
    """
    Calculate per-label TP, FP, FN, Precision, Recall, F1.
    
    Args:
        predictions: List of predicted label sets
        gold_labels: List of ground truth label sets
        num_labels: Total number of labels
    
    Returns:
        Dict with per-label metrics
    """
    # Initialize counters
    tp_count = [0] * num_labels
    fp_count = [0] * num_labels
    fn_count = [0] * num_labels
    support = [0] * num_labels  # True label frequency in test set
    
    for pred, gold in zip(predictions, gold_labels):
        for label_id in range(num_labels):
            if label_id in gold:
                support[label_id] += 1
                if label_id in pred:
                    tp_count[label_id] += 1
                else:
                    fn_count[label_id] += 1
            elif label_id in pred:
                fp_count[label_id] += 1
    
    # Calculate metrics
    metrics = {}
    for label_id in range(num_labels):
        p, r, f = _precision_recall_f1(tp_count[label_id], fp_count[label_id], fn_count[label_id])
        metrics[label_id] = {
            'tp': tp_count[label_id],
            'fp': fp_count[label_id],
            'fn': fn_count[label_id],
            'precision': p,
            'recall': r,
            'f1': f,
            'support': support[label_id]
        }
    
    return metrics


def calculate_group_metrics(label_ids: List[int], per_label_metrics: Dict):
    """
    Calculate aggregated metrics for a group of labels.
    
    Args:
        label_ids: List of label IDs in this group
        per_label_metrics: Per-label metrics from calculate_per_label_metrics
    
    Returns:
        Dict with Macro-P, Macro-R, Macro-F1, Micro-F1
    """
    if not label_ids:
        return {
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'num_labels': 0
        }
    
    # Macro metrics: average of per-label metrics
    precisions = [per_label_metrics[lid]['precision'] for lid in label_ids]
    recalls = [per_label_metrics[lid]['recall'] for lid in label_ids]
    f1s = [per_label_metrics[lid]['f1'] for lid in label_ids]
    
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    # Micro metrics: aggregate TP, FP, FN
    tp_total = sum(per_label_metrics[lid]['tp'] for lid in label_ids)
    fp_total = sum(per_label_metrics[lid]['fp'] for lid in label_ids)
    fn_total = sum(per_label_metrics[lid]['fn'] for lid in label_ids)
    
    micro_p, micro_r, micro_f1 = _precision_recall_f1(tp_total, fp_total, fn_total)
    
    return {
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_f1': micro_f1,
        'num_labels': len(label_ids)
    }


def get_frequency_groups(label_freq: Dict[int, int], num_labels: int):
    """
    Group labels into 5 frequency tiers based on percentiles.
    
    Args:
        label_freq: Dict of label_id -> frequency in training set
        num_labels: Total number of labels
    
    Returns:
        Dict of group_name -> list of label_ids
    """
    full_label_freq = {label_id: int(label_freq.get(label_id, 0)) for label_id in range(num_labels)}
    # Sort labels by frequency (descending)
    sorted_labels = sorted(full_label_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate group sizes (based on percentiles)
    group_names = ['>80%', '60-80%', '40-60%', '20-40%', '<20%']
    percentiles = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    groups = {}
    prev_idx = 0
    for i, (name, percentile) in enumerate(zip(group_names, percentiles)):
        end_idx = int(num_labels * percentile)
        group_labels = [label_id for label_id, _ in sorted_labels[prev_idx:end_idx]]
        groups[name] = group_labels
        prev_idx = end_idx
    
    return groups


def get_hierarchy_groups(depth2label: Dict[int, List[int]]):
    """
    Group labels by hierarchy level.
    
    Args:
        depth2label: Dict of depth_level -> list of label_ids
    
    Returns:
        Dict of level -> list of label_ids (sorted by level)
    """
    return {level: labels for level, labels in sorted(depth2label.items())}


def analyze_labels(predictions: List[Set[int]], 
                   gold_labels: List[Set[int]], 
                   label_freq: Dict[int, int],
                   depth2label: Dict[int, List[int]],
                   id2label: Dict[int, str],
                   logger: Optional[object] = None):
    """
    Main analysis function: calculates and logs all metrics.
    
    Args:
        predictions: List of predicted label sets
        gold_labels: List of ground truth label sets
        label_freq: Label frequency in training set
        depth2label: Hierarchy mapping
        id2label: Label ID to name mapping
        logger: Logger object for output
    """
    num_labels = len(id2label)
    report_lines = []

    def emit(line: str = ""):
        report_lines.append(line)
        if logger is not None:
            logger.info(line)
    
    # 1. Per-label metrics
    emit("\n" + "="*80)
    emit("=== PER-LABEL METRICS ===")
    emit("="*80)
    per_label_metrics = calculate_per_label_metrics(predictions, gold_labels, num_labels)
    
    emit(f"{'Label_ID':<10} {'Label_Name':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    emit("-" * 110)
    for label_id in sorted(per_label_metrics.keys()):
        metrics = per_label_metrics[label_id]
        label_name = id2label.get(label_id, f"Label_{label_id}")
        emit(f"{label_id:<10} {label_name:<40} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
             f"{metrics['f1']:<12.4f} {metrics['support']:<10}")
    
    # 2. Frequency-based metrics
    emit("\n" + "="*80)
    emit("=== FREQUENCY-BASED METRICS ===")
    emit("="*80)
    freq_groups = get_frequency_groups(label_freq, num_labels)
    frequency_group_metrics = {}
    
    emit(f"{'Group':<15} {'#Labels':<10} {'Macro-P':<12} {'Macro-R':<12} {'Macro-F1':<12} {'Micro-F1':<12}")
    emit("-" * 80)
    for group_name in ['>80%', '60-80%', '40-60%', '20-40%', '<20%']:
        group_labels = freq_groups[group_name]
        group_metrics = calculate_group_metrics(group_labels, per_label_metrics)
        frequency_group_metrics[group_name] = group_metrics
        emit(f"{group_name:<15} {group_metrics['num_labels']:<10} "
             f"{group_metrics['macro_precision']:<12.4f} {group_metrics['macro_recall']:<12.4f} "
             f"{group_metrics['macro_f1']:<12.4f} {group_metrics['micro_f1']:<12.4f}")
    
    # 3. Hierarchy-based metrics
    emit("\n" + "="*80)
    emit("=== HIERARCHY-BASED METRICS ===")
    emit("="*80)
    hierarchy_groups = get_hierarchy_groups(depth2label)
    hierarchy_group_metrics = {}
    
    emit(f"{'Level':<10} {'#Labels':<10} {'Macro-P':<12} {'Macro-R':<12} {'Macro-F1':<12} {'Micro-F1':<12}")
    emit("-" * 80)
    for level in sorted(hierarchy_groups.keys()):
        level_labels = hierarchy_groups[level]
        level_metrics = calculate_group_metrics(level_labels, per_label_metrics)
        hierarchy_group_metrics[level] = level_metrics
        emit(f"{level:<10} {level_metrics['num_labels']:<10} "
             f"{level_metrics['macro_precision']:<12.4f} {level_metrics['macro_recall']:<12.4f} "
             f"{level_metrics['macro_f1']:<12.4f} {level_metrics['micro_f1']:<12.4f}")
    
    emit("\n" + "="*80)
    return {
        'per_label_metrics': per_label_metrics,
        'frequency_groups': freq_groups,
        'frequency_group_metrics': frequency_group_metrics,
        'hierarchy_groups': hierarchy_groups,
        'hierarchy_group_metrics': hierarchy_group_metrics,
        'report_lines': report_lines,
    }
