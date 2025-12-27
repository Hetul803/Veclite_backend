#!/usr/bin/env python3
"""
Large-scale synthetic test for MCN v1.

Tests on 10k, 50k, 100k vectors to validate scalability.
"""
import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psutil
from mcn import MCNLayer


def generate_clustered_data(n_vectors: int, dim: int = 384, n_clusters: int = None, seed: int = 42):
    """Generate realistic clustered data."""
    np.random.seed(seed)
    
    if n_clusters is None:
        n_clusters = max(100, n_vectors // 15)
    
    # Generate cluster centroids
    centroids = np.random.randn(n_clusters, dim).astype("float32")
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    
    # Assign vectors to clusters
    vectors = []
    cluster_sizes = np.random.multinomial(n_vectors, np.ones(n_clusters) / n_clusters)
    
    for cluster_id, size in enumerate(cluster_sizes):
        cluster_vecs = centroids[cluster_id:cluster_id+1] + np.random.randn(size, dim).astype("float32") * 0.1
        cluster_vecs = cluster_vecs / (np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-10)
        vectors.append(cluster_vecs)
    
    vectors = np.vstack(vectors)
    
    # Shuffle
    perm = np.random.permutation(len(vectors))
    vectors = vectors[perm]
    
    return vectors


def brute_force_search(query: np.ndarray, vectors: np.ndarray, k: int = 10):
    """Brute-force cosine similarity search."""
    query = query / (np.linalg.norm(query) + 1e-10)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    scores = (query @ vectors_norm.T).flatten()
    top_indices = np.argsort(-scores)[:k]
    return top_indices, scores[top_indices]


def evaluate_scale(n_vectors: int, dim: int = 384, n_queries: int = 200):
    """Evaluate MCN at a specific scale."""
    print(f"\n{'='*80}")
    print(f"Large-Scale Test: {n_vectors:,} vectors, {dim} dimensions")
    print(f"{'='*80}\n")
    
    # Generate data
    print(f"[1/5] Generating {n_vectors:,} vectors...")
    vectors = generate_clustered_data(n_vectors, dim)
    print(f"Generated {len(vectors):,} vectors")
    
    # Generate queries
    print(f"[2/5] Generating {n_queries} queries...")
    queries = np.random.randn(n_queries, dim).astype("float32")
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    
    # Compute ground truth (sample first 10 queries for speed)
    print(f"[3/5] Computing ground truth (first 10 queries)...")
    ground_truth = []
    for query in queries[:10]:
        gt_indices, _ = brute_force_search(query, vectors, k=10)
        ground_truth.append(gt_indices)
    
    # Initialize MCN
    print(f"[4/5] Building MCN index...")
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    mcn = MCNLayer(dim=dim, hot_buffer_size=50, use_background_thread=False)
    
    # Ingest
    ingest_start = time.time()
    batch_size = 1000
    metadata = [{"original_idx": i, "id": f"vec_{i}"} for i in range(len(vectors))]
    
    for i in range(0, len(vectors), batch_size):
        batch_vecs = vectors[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]
        mcn.add(batch_vecs, batch_meta)
        if (i + batch_size) % 10000 == 0:
            print(f"  Ingested {i + batch_size:,}/{len(vectors):,} vectors...")
    
    ingest_time = time.time() - ingest_start
    
    # Finalize
    print("  Finalizing index...")
    finalize_start = time.time()
    mcn.finalize_index(expected_count=len(vectors), timeout_s=600.0)
    finalize_time = time.time() - finalize_start
    
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    
    # Stats
    n_clusters = mcn.get_cold_index_size()
    compression_ratio = len(vectors) / max(1, n_clusters)
    
    print(f"  Index built: {n_clusters:,} clusters ({compression_ratio:.2f}:1 compression)")
    print(f"  Build time: {finalize_time:.2f}s")
    print(f"  Memory: {mem_used:.1f} MB")
    
    # Search
    print(f"[5/5] Running {n_queries} queries...")
    latencies = []
    recall_10_scores = []
    
    for i, query in enumerate(queries):
        search_start = time.time()
        results, scores = mcn.search(query, k=10)
        search_time = (time.time() - search_start) * 1000
        latencies.append(search_time)
        
        # Compute recall (only for first 10 with ground truth)
        if i < 10:
            pred_indices = [r.get("original_idx") for r in results[:10] if r.get("original_idx") is not None]
            pred_set = set(pred_indices)
            gt_set = set(ground_truth[i][:10])
            recall = len(pred_set & gt_set) / max(1, len(gt_set))
            recall_10_scores.append(recall)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_queries} queries...")
    
    # Statistics
    avg_recall = np.mean(recall_10_scores) if recall_10_scores else 0.0
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    avg_latency = np.mean(latencies)
    
    # Storage estimate
    child_store_size = len(vectors) * dim * 4
    super_vectors_size = n_clusters * dim * 4
    csr_size = (n_clusters + 1) * 4 + len(vectors) * 4
    total_storage_mb = (child_store_size + super_vectors_size + csr_size) / 1024 / 1024
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {n_vectors:,} vectors")
    print(f"{'='*80}")
    print(f"Recall@10: {avg_recall:.4f} (from {len(recall_10_scores)} queries with ground truth)")
    print(f"Latency:")
    print(f"  p50: {p50_latency:.2f}ms")
    print(f"  p95: {p95_latency:.2f}ms")
    print(f"  p99: {p99_latency:.2f}ms")
    print(f"  avg: {avg_latency:.2f}ms")
    print(f"Build:")
    print(f"  Ingest: {ingest_time:.2f}s")
    print(f"  Finalize: {finalize_time:.2f}s")
    print(f"  Total: {ingest_time + finalize_time:.2f}s")
    print(f"Resources:")
    print(f"  Memory: {mem_used:.1f} MB")
    print(f"  Storage: {total_storage_mb:.2f} MB")
    print(f"  Compression: {compression_ratio:.2f}:1")
    print(f"  Clusters: {n_clusters:,}")
    print(f"{'='*80}\n")
    
    return {
        "n_vectors": n_vectors,
        "recall_10": avg_recall,
        "latency_p50": p50_latency,
        "latency_p95": p95_latency,
        "latency_p99": p99_latency,
        "latency_avg": avg_latency,
        "build_time": ingest_time + finalize_time,
        "memory_mb": mem_used,
        "storage_mb": total_storage_mb,
        "compression_ratio": compression_ratio,
        "n_clusters": n_clusters
    }


def main():
    parser = argparse.ArgumentParser(description="Large-scale MCN evaluation")
    parser.add_argument("--vectors", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--queries", type=int, default=200, help="Number of queries")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimension")
    parser.add_argument("--all", action="store_true", help="Run all scales (10k, 50k, 100k)")
    
    args = parser.parse_args()
    
    if args.all:
        # Run all scales
        scales = [10000, 50000, 100000]
        all_results = []
        
        for scale in scales:
            try:
                result = evaluate_scale(scale, args.dim, args.queries)
                all_results.append(result)
            except Exception as e:
                print(f"Error at scale {scale}: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*80)
        print("SCALABILITY SUMMARY")
        print("="*80)
        print(f"{'Scale':<10} {'Recall@10':<12} {'p95 (ms)':<12} {'Memory (MB)':<15} {'Compression':<12}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['n_vectors']:>8,}   {r['recall_10']:>10.4f}   {r['latency_p95']:>10.2f}   {r['memory_mb']:>13.1f}   {r['compression_ratio']:>10.2f}:1")
        print("="*80)
    else:
        # Run single scale
        evaluate_scale(args.vectors, args.dim, args.queries)


if __name__ == "__main__":
    main()

