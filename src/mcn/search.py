# src/mcn/search.py
"""
Search: Deterministic routing + exact rerank.
No ANN on children - only exact dot product.
"""
import numpy as np
from typing import Tuple, List, Dict, Optional


def route(
    super_vectors: np.ndarray,
    query: np.ndarray,
    beam_size: int
) -> np.ndarray:
    """
    Route query to top beam_size clusters using exact dot product.
    
    Args:
        super_vectors: (M, dim) float32 normalized super vectors
        query: (dim,) float32 normalized query vector
        beam_size: Number of top clusters to return
        
    Returns:
        top_super_ids: (beam_size,) int32 cluster IDs
    """
    if super_vectors.shape[0] == 0:
        return np.array([], dtype="int32")
    
    # Ensure query is normalized
    query = _l2norm(query.reshape(1, -1)).flatten()
    
    # Matrix-vector product: (M, dim) @ (dim,) -> (M,)
    scores = (super_vectors @ query).flatten()
    
    # Get top beam_size
    beam_size = min(beam_size, len(scores))
    top_indices = np.argpartition(-scores, beam_size - 1)[:beam_size]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    
    return top_indices.astype("int32")


def expand(
    cluster_offsets: np.ndarray,
    cluster_child_ids: np.ndarray,
    top_super_ids: np.ndarray
) -> np.ndarray:
    """
    Expand top clusters to get candidate child IDs.
    
    Args:
        cluster_offsets: (M+1,) int32 CSR-like offsets
        cluster_child_ids: (N,) int32 flattened child IDs
        top_super_ids: (beam_size,) int32 cluster IDs
        
    Returns:
        cand_child_ids: (C,) int32 candidate child IDs
    """
    cand_ids = []
    
    for cluster_id in top_super_ids:
        if 0 <= cluster_id < len(cluster_offsets) - 1:
            start = cluster_offsets[cluster_id]
            end = cluster_offsets[cluster_id + 1]
            cand_ids.extend(cluster_child_ids[start:end].tolist())
    
    return np.array(cand_ids, dtype="int32") if cand_ids else np.array([], dtype="int32")


def rerank(
    child_vectors: np.ndarray,
    cand_child_ids: np.ndarray,
    query: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact rerank on candidate children.
    
    Args:
        child_vectors: (N, dim) float32 child vectors (full store)
        cand_child_ids: (C,) int32 candidate child IDs
        query: (dim,) float32 normalized query vector
        k: Number of top results to return
        
    Returns:
        top_child_ids: (k,) int32 top child IDs
        scores: (k,) float32 similarity scores
    """
    if len(cand_child_ids) == 0:
        return np.array([], dtype="int32"), np.array([], dtype="float32")
    
    # Ensure query is normalized
    query = _l2norm(query.reshape(1, -1)).flatten()
    
    # Get candidate vectors
    valid_ids = cand_child_ids[(cand_child_ids >= 0) & (cand_child_ids < len(child_vectors))]
    if len(valid_ids) == 0:
        return np.array([], dtype="int32"), np.array([], dtype="float32")
    
    cand_vectors = child_vectors[valid_ids]
    
    # Normalize candidate vectors
    cand_vectors = _l2norm(cand_vectors)
    
    # Exact dot product: (C, dim) @ (dim,) -> (C,)
    scores = (cand_vectors @ query).flatten()
    
    # Get top-k
    k = min(k, len(scores))
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    
    top_child_ids = valid_ids[top_indices]
    top_scores = scores[top_indices]
    
    return top_child_ids.astype("int32"), top_scores.astype("float32")


def _l2norm(x: np.ndarray) -> np.ndarray:
    """L2 normalize vectors."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms

