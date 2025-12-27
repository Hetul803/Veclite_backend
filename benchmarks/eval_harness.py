"""
Unified evaluation harness for MCN v1 and baselines.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import json
import numpy as np
import psutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from mcn import MCNLayer
from utils_metrics import calculate_metrics, calculate_latency_stats, calculate_qps


class EmbeddingCache:
    """Cache embeddings to disk to avoid recomputing."""
    
    def __init__(self, cache_dir: Path, model_name: str):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name.replace("/", "_")
        self.model = None
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{self.model_name}_{key}.npy"
    
    def get_model(self):
        """Lazy load model."""
        if self.model is None:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer(model_name, device="cpu")
        return self.model
    
    def encode(self, texts: List[str], cache_key: str, batch_size: int = 32) -> np.ndarray:
        """Encode texts, using cache if available."""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            print(f"  Loading embeddings from cache: {cache_key}")
            return np.load(cache_path)
        
        print(f"  Computing embeddings for {len(texts)} texts...")
        model = self.get_model()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norms
        
        # Save to cache
        np.save(cache_path, embeddings)
        print(f"  Saved embeddings to cache: {cache_key}")
        
        return embeddings


class IndexBuilder:
    """Build indexes for different systems."""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def build_mcn(
        self,
        vectors: np.ndarray,
        metadata: List[Dict],
        config: Optional[Dict] = None
    ) -> Tuple[MCNLayer, Dict[str, Any]]:
        """Build MCN index."""
        config = config or {}
        mcn = MCNLayer(
            dim=self.dim,
            hot_buffer_size=config.get("hot_buffer_size", 50),
            use_background_thread=False,
            **{k: v for k, v in config.items() if k != "hot_buffer_size"}
        )
        
        # Add vectors in batches
        batch_size = 1000
        for i in range(0, len(vectors), batch_size):
            batch_vecs = vectors[i:i+batch_size]
            batch_meta = metadata[i:i+batch_size]
            mcn.add(batch_vecs, batch_meta)
        
        # Finalize
        build_start = time.time()
        mcn.finalize_index(expected_count=len(vectors), timeout_s=600.0)
        build_time = time.time() - build_start
        
        # Get stats
        n_clusters = mcn.get_cold_index_size()
        compression_ratio = len(vectors) / max(1, n_clusters)
        
        stats = {
            "build_time": build_time,
            "n_clusters": n_clusters,
            "compression_ratio": compression_ratio,
        }
        
        return mcn, stats
    
    def build_faiss_exact(
        self,
        vectors: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build FAISS IndexFlatIP (exact cosine)."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        vectors_norm = vectors / norms
        
        build_start = time.time()
        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors_norm.astype("float32"))
        build_time = time.time() - build_start
        
        stats = {
            "build_time": build_time,
            "compression_ratio": 1.0,  # No compression
        }
        
        return index, stats
    
    def build_brute_force(
        self,
        vectors: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build brute-force index (just store vectors)."""
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        vectors_norm = vectors / norms
        
        stats = {
            "build_time": 0.0,  # No build needed
            "compression_ratio": 1.0,
        }
        
        return vectors_norm, stats


class QueryRunner:
    """Run queries against different indexes."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self._mcn_lock = threading.Lock()  # MCN may need read locks
    
    def search_mcn(
        self,
        index: MCNLayer,
        query: np.ndarray,
        k: int
    ) -> Tuple[List[Dict], np.ndarray, float]:
        """Search MCN index."""
        with self._mcn_lock:
            start = time.time()
            results, scores = index.search(query, k=k)
            latency = (time.time() - start) * 1000  # ms
        
        return results, scores, latency
    
    def search_faiss(
        self,
        index: Any,
        query: np.ndarray,
        k: int
    ) -> Tuple[List[int], np.ndarray, float]:
        """Search FAISS index."""
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        query_norm = query_norm.reshape(1, -1).astype("float32")
        
        start = time.time()
        scores, indices = index.search(query_norm, k)
        latency = (time.time() - start) * 1000  # ms
        
        return indices[0].tolist(), scores[0], latency
    
    def search_brute_force(
        self,
        vectors: np.ndarray,
        query: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Search brute-force."""
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        
        start = time.time()
        scores = (query_norm @ vectors.T).flatten()
        top_indices = np.argsort(-scores)[:k]
        latency = (time.time() - start) * 1000  # ms
        
        return top_indices, scores[top_indices], latency
    
    def run_queries_concurrent(
        self,
        search_fn: Callable,
        queries: np.ndarray,
        k: int,
        concurrent: int,
        metadata_map: Optional[Dict[int, Dict]] = None
    ) -> Tuple[List[List[str]], List[float]]:
        """
        Run queries concurrently.
        
        Returns:
            predictions: List of predicted doc ID lists
            latencies: List of latencies in ms
        """
        predictions = []
        latencies = []
        
        def run_single(query_idx: int):
            query = queries[query_idx]
            results, scores, latency = search_fn(query, k)
            
            # Convert to doc IDs
            if metadata_map is not None:
                # MCN returns metadata dicts
                if len(results) > 0 and isinstance(results[0], dict):
                    pred_ids = [str(r.get("id", r.get("original_idx", i))) for i, r in enumerate(results)]
                else:
                    # FAISS/brute-force returns indices
                    pred_ids = [str(metadata_map.get(int(idx), {}).get("id", idx)) for idx in results]
            else:
                # No metadata, use indices
                if len(results) > 0 and isinstance(results[0], dict):
                    pred_ids = [str(r.get("original_idx", i)) for i, r in enumerate(results)]
                else:
                    pred_ids = [str(int(idx)) for idx in results]
            
            return query_idx, pred_ids, latency
        
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = {executor.submit(run_single, i): i for i in range(len(queries))}
            
            results_dict = {}
            for future in as_completed(futures):
                query_idx, pred_ids, latency = future.result()
                results_dict[query_idx] = (pred_ids, latency)
        
        # Sort by query_idx
        for i in range(len(queries)):
            pred_ids, latency = results_dict[i]
            predictions.append(pred_ids)
            latencies.append(latency)
        
        return predictions, latencies


class MemoryTracker:
    """Track memory usage."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def measure(self) -> float:
        """Measure current memory usage in MB."""
        return (self.process.memory_info().rss / 1024 / 1024) - self.baseline
    
    def measure_absolute(self) -> float:
        """Measure absolute memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024


def evaluate_system(
    system_name: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truths: List[List[str]],
    metadata: List[Dict],
    qrels: Optional[List[Dict[str, float]]] = None,
    config: Optional[Dict] = None,
    concurrent: int = 1,
    max_brute_force: int = 20000
) -> Dict[str, Any]:
    """
    Evaluate a single system.
    
    Args:
        system_name: "mcn", "faiss", or "brute_force"
        vectors: Corpus vectors
        queries: Query vectors
        ground_truths: List of ground truth doc ID lists
        metadata: Metadata for each vector
        qrels: Optional relevance scores for nDCG
        config: System-specific config
        concurrent: Number of concurrent queries
        max_brute_force: Max vectors for brute-force (sanity check only)
    
    Returns:
        Dict with metrics and stats
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {system_name}")
    print(f"{'='*80}")
    
    dim = vectors.shape[1]
    builder = IndexBuilder(dim)
    runner = QueryRunner(dim)
    mem_tracker = MemoryTracker()
    
    # Limit brute-force to max_brute_force
    if system_name == "brute_force" and len(vectors) > max_brute_force:
        print(f"  Limiting brute-force to {max_brute_force} vectors (sanity check)")
        vectors = vectors[:max_brute_force]
        metadata = metadata[:max_brute_force]
        # Adjust ground truths
        ground_truths = [[gt for gt in gt_list if int(gt) < max_brute_force] for gt_list in ground_truths]
    
    # Build index
    print(f"  Building index for {len(vectors):,} vectors...")
    mem_before = mem_tracker.measure_absolute()
    
    if system_name == "mcn":
        index, build_stats = builder.build_mcn(vectors, metadata, config)
        def search_fn(q, k):
            return runner.search_mcn(index, q, k)
        metadata_map = {i: meta for i, meta in enumerate(metadata)}
    elif system_name == "faiss":
        index, build_stats = builder.build_faiss_exact(vectors)
        def search_fn(q, k):
            return runner.search_faiss(index, q, k)
        metadata_map = {i: meta for i, meta in enumerate(metadata)}
    elif system_name == "brute_force":
        index, build_stats = builder.build_brute_force(vectors)
        def search_fn(q, k):
            return runner.search_brute_force(index, q, k)
        metadata_map = {i: meta for i, meta in enumerate(metadata)}
    else:
        raise ValueError(f"Unknown system: {system_name}")
    
    mem_after = mem_tracker.measure_absolute()
    mem_used = mem_after - mem_before
    
    print(f"  Build time: {build_stats['build_time']:.2f}s")
    print(f"  Memory: {mem_used:.1f} MB")
    
    # Run queries
    print(f"  Running {len(queries)} queries (concurrent={concurrent})...")
    predictions, latencies = runner.run_queries_concurrent(
        search_fn, queries, k=100, concurrent=concurrent, metadata_map=metadata_map
    )
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths, qrels, k_values=[10, 100])
    latency_stats = calculate_latency_stats(latencies)
    qps = calculate_qps(latencies, concurrent)
    
    # Storage estimate
    if system_name == "mcn":
        storage_mb = (
            len(vectors) * dim * 4 +  # Child vectors
            build_stats["n_clusters"] * dim * 4 +  # Super vectors
            (build_stats["n_clusters"] + 1) * 4 + len(vectors) * 4  # CSR
        ) / 1024 / 1024
    elif system_name == "faiss":
        storage_mb = len(vectors) * dim * 4 / 1024 / 1024
    else:
        storage_mb = len(vectors) * dim * 4 / 1024 / 1024
    
    results = {
        "metrics": metrics,
        "latency_stats": latency_stats,
        "latencies": latencies,
        "build_time": build_stats["build_time"],
        "memory_mb": mem_used,
        "storage_mb": storage_mb,
        "compression_ratio": build_stats.get("compression_ratio", 1.0),
        "qps": qps,
        "n_vectors": len(vectors),
    }
    
    print(f"  Recall@10: {metrics['recall@10']:.4f}")
    print(f"  p95 Latency: {latency_stats['p95']:.2f}ms")
    print(f"  QPS: {qps:.1f}")
    
    return results

