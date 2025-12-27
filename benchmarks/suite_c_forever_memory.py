"""
Test Suite C: Forever-Memory Simulation

Simulates incremental ingest with periodic queries and background compaction.
Tracks staleness, latency drift, and compression over time.
"""
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_harness import EmbeddingCache, IndexBuilder, QueryRunner, MemoryTracker
from utils_metrics import calculate_metrics, calculate_latency_stats


def generate_synthetic_data(
    total_vectors: int,
    dim: int,
    seed: int = 42
) -> Tuple[np.ndarray, List[Dict]]:
    """Generate synthetic vectors for forever-memory simulation."""
    np.random.seed(seed)
    
    # Generate clustered data for realism
    n_clusters = max(100, total_vectors // 15)
    centroids = np.random.randn(n_clusters, dim).astype("float32")
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    
    vectors = []
    cluster_sizes = np.random.multinomial(total_vectors, np.ones(n_clusters) / n_clusters)
    
    for cluster_id, size in enumerate(cluster_sizes):
        cluster_vecs = centroids[cluster_id:cluster_id+1] + np.random.randn(size, dim).astype("float32") * 0.1
        cluster_vecs = cluster_vecs / (np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-10)
        vectors.append(cluster_vecs)
    
    vectors = np.vstack(vectors)
    
    # Shuffle
    perm = np.random.permutation(len(vectors))
    vectors = vectors[perm]
    
    # Create metadata
    metadata = [
        {"original_idx": i, "id": f"vec_{i}", "timestamp": i}
        for i in range(len(vectors))
    ]
    
    return vectors, metadata


def compute_ground_truth_brute_force(
    query: np.ndarray,
    vectors: np.ndarray,
    k: int = 10
) -> List[int]:
    """Compute ground truth for a single query."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    scores = (query_norm @ vectors_norm.T).flatten()
    top_indices = np.argsort(-scores)[:k].tolist()
    return top_indices


def run_suite_c(
    total_vectors: int = 100000,
    wave_size: int = 5000,
    dim: int = 384,
    cache_dir: Path = Path("./cache"),
    seed: int = 42,
    n_queries_per_step: int = 50,
    concurrent: int = 1
) -> Dict:
    """
    Run Test Suite C: Forever-memory simulation.
    
    Args:
        total_vectors: Total vectors to ingest
        wave_size: Vectors per ingestion wave
        dim: Vector dimension
        cache_dir: Cache directory
        seed: Random seed
        n_queries_per_step: Queries to run per time step
        concurrent: Concurrent queries
    
    Returns:
        Results dict
    """
    print(f"\n{'='*80}")
    print(f"Test Suite C: Forever-Memory Simulation")
    print(f"{'='*80}\n")
    
    # Generate all data upfront
    print("Generating synthetic data...")
    all_vectors, all_metadata = generate_synthetic_data(total_vectors, dim, seed)
    
    # Generate queries from different time periods
    np.random.seed(seed + 1)
    n_steps = total_vectors // wave_size
    queries_per_step = []
    
    for step in range(n_steps):
        # Generate queries from vectors ingested up to this step
        step_end = min((step + 1) * wave_size, len(all_vectors))
        step_vectors = all_vectors[:step_end]
        
        # Sample queries from this step's vectors
        query_indices = np.random.choice(len(step_vectors), n_queries_per_step, replace=False)
        queries_per_step.append(step_vectors[query_indices])
    
    # Initialize MCN
    from mcn import MCNLayer
    
    mcn = MCNLayer(dim=dim, hot_buffer_size=50, use_background_thread=False)
    builder = IndexBuilder(dim)
    runner = QueryRunner(dim)
    mem_tracker = MemoryTracker()
    
    # Results storage
    results = {
        "time_steps": [],
        "recall@10": [],
        "recall@100": [],
        "latency_p50": [],
        "latency_p95": [],
        "compression_ratio": [],
        "memory_mb": [],
        "build_time": [],
        "staleness_recall": {
            "t-5": [],
            "t-10": [],
            "t-20": []
        }
    }
    
    # Simulate time steps
    for step in range(n_steps):
        step_start_idx = step * wave_size
        step_end_idx = min((step + 1) * wave_size, len(all_vectors))
        
        print(f"\n{'='*80}")
        print(f"Time Step {step + 1}/{n_steps}: Ingesting vectors {step_start_idx:,} to {step_end_idx:,}")
        print(f"{'='*80}\n")
        
        # Ingest wave
        wave_vectors = all_vectors[step_start_idx:step_end_idx]
        wave_metadata = all_metadata[step_start_idx:step_end_idx]
        
        print(f"Ingesting {len(wave_vectors):,} vectors...")
        ingest_start = time.time()
        
        batch_size = 1000
        for i in range(0, len(wave_vectors), batch_size):
            batch_vecs = wave_vectors[i:i+batch_size]
            batch_meta = wave_metadata[i:i+batch_size]
            mcn.add(batch_vecs, batch_meta)
        
        ingest_time = time.time() - ingest_start
        
        # Finalize (compaction)
        print("Finalizing index (compaction)...")
        mem_before = mem_tracker.measure_absolute()
        finalize_start = time.time()
        mcn.finalize_index(expected_count=step_end_idx, timeout_s=600.0)
        finalize_time = time.time() - finalize_start
        mem_after = mem_tracker.measure_absolute()
        mem_used = mem_after - mem_before
        
        build_time = ingest_time + finalize_time
        
        # Get compression stats
        n_clusters = mcn.get_cold_index_size()
        compression_ratio = step_end_idx / max(1, n_clusters)
        
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Memory: {mem_used:.1f} MB")
        print(f"  Compression: {compression_ratio:.2f}:1 ({n_clusters:,} clusters)")
        
        # Run queries
        print(f"Running {n_queries_per_step} queries...")
        queries = queries_per_step[step]
        
        latencies = []
        predictions = []
        ground_truths = []
        
        for query in queries:
            # Compute ground truth from all vectors ingested so far
            gt_indices = compute_ground_truth_brute_force(query, all_vectors[:step_end_idx], k=10)
            ground_truths.append([str(i) for i in gt_indices])
            
            # Search MCN
            start = time.time()
            search_results, scores = mcn.search(query, k=100)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            # Convert to doc IDs
            pred_ids = [r.get("id", r.get("original_idx", i)) for i, r in enumerate(search_results[:100])]
            predictions.append(pred_ids)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths, None, k_values=[10, 100])
        latency_stats = calculate_latency_stats(latencies)
        
        # Calculate staleness recall (for queries from t-5, t-10, t-20 steps ago)
        staleness_recall = {"t-5": 0.0, "t-10": 0.0, "t-20": 0.0}
        
        for lookback, key in [(5, "t-5"), (10, "t-10"), (20, "t-20")]:
            if step >= lookback:
                # Use queries from step - lookback
                old_queries = queries_per_step[step - lookback]
                old_gt_start = max(0, (step - lookback) * wave_size)
                
                old_predictions = []
                old_ground_truths = []
                
                for query in old_queries[:10]:  # Sample 10 queries
                    # Ground truth from when query was created
                    old_gt_indices = compute_ground_truth_brute_force(
                        query, all_vectors[:old_gt_start + wave_size], k=10
                    )
                    old_ground_truths.append([str(i) for i in old_gt_indices])
                    
                    # Search current index
                    search_results, _ = mcn.search(query, k=10)
                    pred_ids = [r.get("id", r.get("original_idx", i)) for i, r in enumerate(search_results[:10])]
                    old_predictions.append(pred_ids)
                
                # Calculate recall
                from utils_metrics import recall_at_k
                recalls = [
                    recall_at_k(pred, gt, 10)
                    for pred, gt in zip(old_predictions, old_ground_truths)
                ]
                staleness_recall[key] = np.mean(recalls) if recalls else 0.0
        
        # Store results
        results["time_steps"].append(step + 1)
        results["recall@10"].append(metrics["recall@10"])
        results["recall@100"].append(metrics["recall@100"])
        results["latency_p50"].append(latency_stats["p50"])
        results["latency_p95"].append(latency_stats["p95"])
        results["compression_ratio"].append(compression_ratio)
        results["memory_mb"].append(mem_used)
        results["build_time"].append(build_time)
        results["staleness_recall"]["t-5"].append(staleness_recall["t-5"])
        results["staleness_recall"]["t-10"].append(staleness_recall["t-10"])
        results["staleness_recall"]["t-20"].append(staleness_recall["t-20"])
        
        print(f"  Recall@10: {metrics['recall@10']:.4f}")
        print(f"  p95 Latency: {latency_stats['p95']:.2f}ms")
        print(f"  Staleness Recall (t-5): {staleness_recall['t-5']:.4f}")
    
    return {
        "suite_name": "Forever-Memory",
        "total_vectors": total_vectors,
        "wave_size": wave_size,
        "n_steps": n_steps,
        "results": results,
        "seed": seed,
    }

