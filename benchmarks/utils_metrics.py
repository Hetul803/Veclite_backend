"""
Metrics calculation utilities for evaluation.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


def recall_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Recall@K."""
    if len(ground_truth) == 0:
        return 0.0
    pred_set = set(predicted[:k])
    gt_set = set(ground_truth)
    return len(pred_set & gt_set) / len(gt_set)


def mrr_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank@K."""
    if len(ground_truth) == 0:
        return 0.0
    gt_set = set(ground_truth)
    for rank, pred_id in enumerate(predicted[:k], 1):
        if pred_id in gt_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: List[str], ground_truth: Dict[str, float], k: int) -> float:
    """
    Calculate nDCG@K.
    
    Args:
        predicted: List of predicted document IDs
        ground_truth: Dict mapping doc_id -> relevance score
        k: Top K to consider
    """
    if len(ground_truth) == 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for rank, pred_id in enumerate(predicted[:k], 1):
        rel = ground_truth.get(pred_id, 0.0)
        dcg += rel / np.log2(rank + 1)
    
    # Calculate IDCG (ideal DCG)
    ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_metrics(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    qrels: Optional[List[Dict[str, float]]] = None,
    k_values: List[int] = [10, 100]
) -> Dict[str, float]:
    """
    Calculate all metrics for a set of queries.
    
    Args:
        predictions: List of predicted doc ID lists (one per query)
        ground_truths: List of ground truth doc ID lists (one per query)
        qrels: Optional list of relevance dicts (doc_id -> score) for nDCG
        k_values: List of K values to compute metrics for
    
    Returns:
        Dict of metric_name -> value
    """
    metrics = {}
    
    # Recall@K
    for k in k_values:
        recalls = [recall_at_k(pred, gt, k) for pred, gt in zip(predictions, ground_truths)]
        metrics[f"recall@{k}"] = np.mean(recalls)
    
    # MRR@10
    mrrs = [mrr_at_k(pred, gt, 10) for pred, gt in zip(predictions, ground_truths)]
    metrics["mrr@10"] = np.mean(mrrs)
    
    # nDCG@10 (if qrels available)
    if qrels is not None:
        ndcgs = [ndcg_at_k(pred, qrel, 10) for pred, qrel in zip(predictions, qrels)]
        metrics["ndcg@10"] = np.mean(ndcgs)
    
    return metrics


def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    if len(latencies) == 0:
        return {}
    
    latencies = np.array(latencies)
    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "mean": np.mean(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
    }


def calculate_qps(latencies: List[float], concurrent: int) -> float:
    """Calculate queries per second under concurrency."""
    if len(latencies) == 0:
        return 0.0
    total_time = np.sum(latencies) / 1000.0  # Convert ms to seconds
    if total_time == 0:
        return 0.0
    return (len(latencies) * concurrent) / total_time

