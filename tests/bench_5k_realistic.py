# tests/bench_5k_realistic.py
"""
Deterministic benchmark for MCN v1.
Tests Recall@10 >= 0.90 on realistic clustered datasets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from typing import Tuple, List
from mcn import MCNLayer


def generate_faq_dataset(n_vectors: int = 5000, dim: int = 384, n_clusters: int = 100, seed: int = 42):
    """
    Generate realistic FAQ/chatbot dataset:
    - Clustered semantic duplicates
    - Paraphrase noise
    - 2-5% outliers
    """
    np.random.seed(seed)
    
    # Generate cluster centroids
    centroids = np.random.randn(n_clusters, dim).astype("float32")
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    
    # Assign vectors to clusters
    vectors = []
    labels = []
    cluster_sizes = np.random.multinomial(n_vectors, np.ones(n_clusters) / n_clusters)
    
    for cluster_id, size in enumerate(cluster_sizes):
        # Generate vectors around centroid with noise
        cluster_vecs = centroids[cluster_id:cluster_id+1] + np.random.randn(size, dim).astype("float32") * 0.1
        cluster_vecs = cluster_vecs / (np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-10)
        vectors.append(cluster_vecs)
        labels.extend([cluster_id] * size)
    
    vectors = np.vstack(vectors)
    
    # Add 2-5% outliers (random vectors)
    n_outliers = int(n_vectors * 0.03)
    outlier_indices = np.random.choice(len(vectors), n_outliers, replace=False)
    outliers = np.random.randn(n_outliers, dim).astype("float32")
    outliers = outliers / (np.linalg.norm(outliers, axis=1, keepdims=True) + 1e-10)
    vectors[outlier_indices] = outliers
    
    # Shuffle
    perm = np.random.permutation(len(vectors))
    vectors = vectors[perm]
    labels = [labels[i] for i in perm]
    
    # Create metadata
    metadata = []
    for i, (vec, label) in enumerate(zip(vectors, labels)):
        metadata.append({
            "original_idx": i,
            "id": f"vec_{i}",
            "cluster_id": label,
            "is_outlier": i in outlier_indices
        })
    
    return vectors, metadata, labels


def brute_force_search(query: np.ndarray, vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute-force cosine similarity search (ground truth).
    """
    # Normalize
    query = query / (np.linalg.norm(query) + 1e-10)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    
    # Dot product
    scores = (query @ vectors_norm.T).flatten()
    
    # Top-k
    top_k = min(k, len(scores))
    top_indices = np.argsort(-scores)[:top_k]
    top_scores = scores[top_indices]
    
    return top_indices, top_scores


def compute_recall(predicted_ids: List[int], ground_truth_ids: np.ndarray, k: int = 10) -> float:
    """
    Compute Recall@k.
    """
    if len(predicted_ids) == 0:
        return 0.0
    
    predicted_set = set(predicted_ids[:k])
    ground_truth_set = set(ground_truth_ids[:k].tolist())
    
    if len(ground_truth_set) == 0:
        return 0.0
    
    intersection = predicted_set & ground_truth_set
    return len(intersection) / len(ground_truth_set)


def run_benchmark():
    """Run deterministic benchmark."""
    print("=" * 80)
    print("MCN v1 Benchmark: 5k Realistic Dataset")
    print("=" * 80)
    
    # Generate dataset
    print("\n[1/5] Generating dataset...")
    dim = 384
    n_vectors = 5000
    vectors, metadata, labels = generate_faq_dataset(n_vectors, dim, n_clusters=100, seed=42)
    print(f"Generated {n_vectors} vectors in {len(set(labels))} clusters")
    
    # Initialize MCN
    print("\n[2/5] Initializing MCN...")
    mcn = MCNLayer(
        dim=dim,
        hot_buffer_size=50,
        use_background_thread=False
    )
    
    # Ingest vectors
    print("\n[3/5] Ingesting vectors...")
    ingest_start = time.time()
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch_vecs = vectors[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]
        mcn.add(batch_vecs, batch_meta)
    ingest_time = time.time() - ingest_start
    print(f"Ingested {n_vectors} vectors in {ingest_time:.2f}s")
    
    # Finalize index
    print("\n[4/5] Finalizing index...")
    finalize_start = time.time()
    mcn.finalize_index(expected_count=n_vectors, timeout_s=120.0)
    finalize_time = time.time() - finalize_start
    print(f"Index finalized in {finalize_time:.2f}s")
    print(f"  - Child store: {mcn.child_store.size()} vectors")
    print(f"  - Clusters: {mcn.get_cold_index_size()} super vectors")
    
    # Generate test queries
    print("\n[5/5] Running search benchmark...")
    n_queries = 200
    query_indices = np.random.choice(n_vectors, n_queries, replace=False)
    queries = vectors[query_indices]
    
    # Run searches
    latencies = []
    recall_10_scores = []
    recall_1_scores = []
    
    for i, (query_idx, query) in enumerate(zip(query_indices, queries)):
        # MCN search
        search_start = time.time()
        results, scores = mcn.search(query, k=10)
        search_time = (time.time() - search_start) * 1000  # ms
        latencies.append(search_time)
        
        # Ground truth
        gt_indices, gt_scores = brute_force_search(query, vectors, k=10)
        
        # Extract IDs from results
        pred_ids = [r.get("original_idx") for r in results if r.get("original_idx") is not None]
        
        # Compute recall
        recall_10 = compute_recall(pred_ids, gt_indices, k=10)
        recall_1 = compute_recall(pred_ids, gt_indices, k=1)
        recall_10_scores.append(recall_10)
        recall_1_scores.append(recall_1)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_queries} queries...")
    
    # Compute statistics
    avg_recall_10 = np.mean(recall_10_scores)
    avg_recall_1 = np.mean(recall_1_scores)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    avg_latency = np.mean(latencies)
    
    # Memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Compression ratio
    compression_ratio = n_vectors / max(1, mcn.get_cold_index_size())
    
    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Dataset: {n_vectors} vectors, {dim} dimensions")
    print(f"Clusters: {mcn.get_cold_index_size()} super vectors")
    print(f"Compression ratio: {compression_ratio:.1f}:1")
    print(f"\nRecall Metrics:")
    print(f"  Recall@10: {avg_recall_10:.4f} (target: >= 0.90)")
    print(f"  Recall@1:  {avg_recall_1:.4f}")
    print(f"\nLatency Metrics (ms):")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  p50:     {p50_latency:.2f}ms")
    print(f"  p95:     {p95_latency:.2f}ms")
    print(f"\nBuild Metrics:")
    print(f"  Ingest time: {ingest_time:.2f}s")
    print(f"  Finalize time: {finalize_time:.2f}s")
    print(f"  Memory usage: {memory_mb:.1f} MB")
    print("=" * 80)
    
    # Validation
    if avg_recall_10 >= 0.90:
        print("\n✅ PASS: Recall@10 >= 0.90")
        return 0
    else:
        print(f"\n❌ FAIL: Recall@10 = {avg_recall_10:.4f} < 0.90")
        return 1


if __name__ == "__main__":
    exit_code = run_benchmark()
    sys.exit(exit_code)

