# src/mcn/cluster.py
"""
Clustering: Build clusters from child vectors and create super vectors.
Deterministic, production-fast clustering with hard caps.
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
from .config import MCNConfig


def build_clusters(
    child_vectors: np.ndarray,
    child_meta: List[Dict],
    config: MCNConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build clusters from child vectors.
    
    Args:
        child_vectors: (N, dim) normalized float32 vectors
        child_meta: List of N metadata dicts (must contain original_idx)
        config: MCNConfig
        
    Returns:
        super_vectors: (M, dim) float32 normalized super vectors
        cluster_offsets: (M+1,) int32 CSR-like offsets
        cluster_child_ids: (N,) int32 flattened child IDs per cluster
    """
    n = len(child_vectors)
    if n == 0:
        return (
            np.empty((0, config.dim), dtype="float32"),
            np.array([0], dtype="int32"),
            np.array([], dtype="int32")
        )
    
    # Normalize vectors (ensure unit length)
    child_vectors = _l2norm(child_vectors)
    
    # Calculate number of clusters
    n_clusters = max(1, n // config.target_cluster_size)
    n_clusters = min(n_clusters, n // 2) if n >= 2 else 1
    
    # Cluster using MiniBatchKMeans
    from sklearn.cluster import MiniBatchKMeans
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(256, n),
        n_init=1,
        random_state=42,
        max_iter=50
    )
    labels = kmeans.fit_predict(child_vectors)
    
    # Group by cluster
    cluster_groups: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(i)
    
    # Split oversized clusters
    final_clusters = _split_oversized_clusters(
        cluster_groups, child_vectors, config.max_cluster_size
    )
    
    # Build super vectors and CSR structure
    super_vectors = []
    cluster_offsets = [0]
    cluster_child_ids = []
    
    for cluster_id, indices in sorted(final_clusters.items()):
        if len(indices) == 0:
            continue
        
        cluster_vecs = child_vectors[indices]
        
        # Compute centroid (normalized mean)
        centroid = _compute_centroid(cluster_vecs)
        
        # Optional refinement (GA) if needed
        if config.use_refine:
            similarities = (centroid @ cluster_vecs.T).flatten()
            min_sim = float(np.min(similarities))
            if min_sim < config.refine_threshold:
                centroid = _refine_cluster_rep(
                    centroid, cluster_vecs, config
                )
        
        super_vectors.append(centroid)
        cluster_child_ids.extend(indices)
        cluster_offsets.append(len(cluster_child_ids))
    
    # Convert to arrays
    super_vectors_array = np.vstack(super_vectors).astype("float32")
    cluster_offsets_array = np.array(cluster_offsets, dtype="int32")
    cluster_child_ids_array = np.array(cluster_child_ids, dtype="int32")
    
    return super_vectors_array, cluster_offsets_array, cluster_child_ids_array


def _split_oversized_clusters(
    cluster_groups: Dict[int, List[int]],
    vectors: np.ndarray,
    max_size: int
) -> Dict[int, List[int]]:
    """
    Recursively split clusters exceeding max_size.
    """
    final_clusters = {}
    next_id = max(cluster_groups.keys()) + 1 if cluster_groups else 0
    
    def split_recursive(indices: List[int], cluster_id: int):
        nonlocal next_id
        
        if len(indices) <= max_size:
            final_clusters[cluster_id] = indices
            return
        
        # Split using MiniBatchKMeans
        from sklearn.cluster import MiniBatchKMeans
        cluster_vecs = vectors[indices]
        n_sub = (len(indices) + max_size - 1) // max_size
        n_sub = max(2, min(n_sub, len(indices) // 2))
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_sub,
            batch_size=min(256, len(indices)),
            n_init=1,
            random_state=42,
            max_iter=20
        )
        sub_labels = kmeans.fit_predict(cluster_vecs)
        
        for sub_id in range(n_sub):
            sub_indices = [indices[i] for i in range(len(indices)) if sub_labels[i] == sub_id]
            if len(sub_indices) > 0:
                split_recursive(sub_indices, next_id)
                next_id += 1
    
    for orig_id, indices in cluster_groups.items():
        split_recursive(indices, orig_id)
    
    return final_clusters


def _compute_centroids(vectors: np.ndarray) -> np.ndarray:
    """Compute normalized centroids for clusters."""
    centroid = vectors.mean(axis=0)
    return _l2norm(centroid.reshape(1, -1)).flatten()


def _compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """Compute normalized centroid (single cluster)."""
    return _compute_centroids(vectors)


def _refine_cluster_rep(
    initial_rep: np.ndarray,
    cluster_vectors: np.ndarray,
    config: MCNConfig
) -> np.ndarray:
    """
    Refine cluster representative using genetic algorithm.
    Hard timeout: max_refine_gen generations.
    """
    n = len(cluster_vectors)
    if n == 0:
        return initial_rep
    
    # Initialize population
    population = [initial_rep.copy()]
    for _ in range(config.population_size - 1):
        mutated = initial_rep + np.random.normal(0, 0.1, size=initial_rep.shape)
        population.append(_l2norm(mutated.reshape(1, -1)).flatten())
    
    best_vector = initial_rep.copy()
    best_score = float(np.min((initial_rep @ cluster_vectors.T).flatten()))
    
    # Evolve for max generations
    for generation in range(config.max_refine_gen):
        # Evaluate fitness (min similarity)
        fitness_scores = []
        for candidate in population:
            candidate_norm = _l2norm(candidate.reshape(1, -1)).flatten()
            similarities = (candidate_norm @ cluster_vectors.T).flatten()
            fitness_scores.append(float(np.min(similarities)))
        
        # Find best
        best_idx = int(np.argmax(fitness_scores))
        if fitness_scores[best_idx] > best_score:
            best_vector = population[best_idx].copy()
            best_score = fitness_scores[best_idx]
        
        # Early stopping if good enough
        if best_score >= config.refine_threshold:
            break
        
        # Evolve population
        elite_count = max(1, len(population) // 5)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        new_population = [population[i].copy() for i in elite_indices]
        
        while len(new_population) < len(population):
            if np.random.random() < 0.5:
                # Mutation
                parent = population[np.random.choice(elite_indices)]
                child = parent + np.random.normal(0, 0.1, size=parent.shape)
            else:
                # Crossover
                p1, p2 = np.random.choice(elite_indices, size=2, replace=False)
                alpha = np.random.random()
                child = alpha * population[p1] + (1 - alpha) * population[p2]
            new_population.append(_l2norm(child.reshape(1, -1)).flatten())
        
        population = new_population[:len(population)]
    
    return best_vector


def _l2norm(x: np.ndarray) -> np.ndarray:
    """L2 normalize vectors."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms

