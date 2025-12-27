#!/usr/bin/env python3
"""
TEST 1: Compression While Serving Queries (Online Mutation Test)

Goal: Verify MCN correctness and latency while new data is ingested + compressed
concurrently with queries.

Setup:
- Dataset: BEIR FiQA or SciFact
- Initial vectors: 100,000
- Query set: 200 queries (ground truth)
- Ingest waves: +10,000 vectors every 5 seconds (until +50k total)
- Compression happens at ingestion time, not later

Expected Outcome:
- Recall drop ≤ 1%
- Latency increase ≤ 20%
- Zero crashes
- No blocking during compression
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import json
import numpy as np
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("ERROR: BEIR not available. Install with: pip install beir")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("ERROR: sentence-transformers not available")
    sys.exit(1)

from mcn import MCNLayer
try:
    from benchmarks.eval_harness import EmbeddingCache
    from benchmarks.utils_metrics import calculate_metrics, calculate_latency_stats
except ImportError:
    # Fallback if benchmarks module not available
    import pickle
    from sentence_transformers import SentenceTransformer
    
    class EmbeddingCache:
        def __init__(self, cache_dir, model_name):
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.model = SentenceTransformer(model_name)
        
        def encode(self, texts, cache_key):
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            return embeddings
    
    def calculate_metrics(predictions, ground_truths, qrels, k_values=[10, 100]):
        """Calculate recall metrics."""
        metrics = {}
        for k in k_values:
            recall_sum = 0.0
            count = 0
            for pred, gt in zip(predictions, ground_truths):
                if len(gt) > 0:
                    pred_k = pred[:k]
                    hits = sum(1 for p in pred_k if p in gt)
                    recall_sum += hits / min(len(gt), k)
                    count += 1
            metrics[f"recall@{k}"] = recall_sum / max(count, 1)
        return metrics
    
    def calculate_latency_stats(latencies):
        """Calculate latency statistics."""
        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {
            "p50": sorted_lat[int(n * 0.50)],
            "p95": sorted_lat[int(n * 0.95)],
            "p99": sorted_lat[int(n * 0.99)],
            "mean": np.mean(sorted_lat),
        }


class OnlineMutationTest:
    """Test online mutation with concurrent queries and ingestion."""
    
    def __init__(
        self,
        dataset_name: str,
        initial_size: int = 100000,
        query_count: int = 200,
        wave_size: int = 10000,
        num_waves: int = 5,
        wave_interval: float = 5.0,
        query_qps: float = 20.0,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.initial_size = initial_size
        self.query_count = query_count
        self.wave_size = wave_size
        self.num_waves = num_waves
        self.wave_interval = wave_interval
        self.query_qps = query_qps
        self.seed = seed
        self.dim = 384
        
        # State
        self.mcn = None
        self.corpus_embeddings = None
        self.query_embeddings = None
        self.ground_truths = None
        self.qrels_list = None
        self.corpus_ids = None
        self.query_ids = None
        
        # Metrics tracking
        self.metrics_history = []
        self.latency_history = deque(maxlen=1000)
        self.error_count = 0
        self.total_queries = 0
        self.compression_times = []
        self.ingestion_times = []
        
        # Threading
        self.query_thread = None
        self.ingestion_thread = None
        self.stop_flag = threading.Event()
        self.metrics_lock = threading.Lock()
        
        # Baseline metrics (before mutation)
        self.baseline_recall = None
        self.baseline_p95_latency = None
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[Dict], List[str], List[str]]:
        """Load BEIR dataset and build embeddings."""
        print(f"\n[1/4] Loading dataset: {self.dataset_name}...")
        
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        data_path = f"./beir_data/{self.dataset_name}"
        
        # Download if needed
        if not os.path.exists(data_path) or not os.path.exists(f"{data_path}/corpus.jsonl"):
            print(f"  Downloading {self.dataset_name}...")
            util.download_and_unzip(url, data_path)
            # Fix nested directory
            if os.path.exists(f"{data_path}/{self.dataset_name}"):
                import shutil
                for f in os.listdir(f"{data_path}/{self.dataset_name}"):
                    src = f"{data_path}/{self.dataset_name}/{f}"
                    dst = f"{data_path}/{f}"
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
        
        # Load data
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        
        # Convert to lists
        corpus_ids_list = sorted(corpus.keys())
        corpus_texts_list = [corpus[doc_id]["text"] for doc_id in corpus_ids_list]
        
        print(f"  Loaded {len(corpus_texts_list):,} documents")
        
        # Subsample to initial size + waves (or use all available)
        np.random.seed(self.seed)
        total_needed = self.initial_size + (self.wave_size * self.num_waves)
        if len(corpus_texts_list) < total_needed:
            print(f"  Warning: Dataset has only {len(corpus_texts_list):,} docs (need {total_needed:,})")
            print(f"  Using all available documents")
        elif len(corpus_texts_list) > total_needed:
            indices = np.random.choice(len(corpus_texts_list), total_needed, replace=False)
            indices = sorted(indices)
            corpus_texts_list = [corpus_texts_list[i] for i in indices]
            corpus_ids_list = [corpus_ids_list[i] for i in indices]
            print(f"  Subsample to {len(corpus_texts_list):,} documents")
        
        # Load queries
        query_ids_list = sorted(queries.keys())
        query_texts_list = [queries[q_id] for q_id in query_ids_list]
        
        if len(query_texts_list) > self.query_count:
            query_ids_list = query_ids_list[:self.query_count]
            query_texts_list = query_texts_list[:self.query_count]
            qrels = {q_id: qrels[q_id] for q_id in query_ids_list if q_id in qrels}
        
        # Convert qrels
        qrels_list = []
        for q_id in query_ids_list:
            if q_id in qrels:
                qrels_list.append(qrels[q_id])
            else:
                qrels_list.append({})
        
        qrels_dict = {}
        for q_id in query_ids_list:
            if q_id in qrels:
                qrels_dict[q_id] = list(qrels[q_id].keys())
            else:
                qrels_dict[q_id] = []
        
        ground_truths = [qrels_dict.get(q_id, []) for q_id in query_ids_list]
        
        # Filter ground truths to only include docs in corpus
        corpus_ids_set = set(corpus_ids_list)
        ground_truths = [
            [gt for gt in gt_list if gt in corpus_ids_set]
            for gt_list in ground_truths
        ]
        
        print(f"  Loaded {len(query_texts_list)} queries")
        print(f"  Queries with ground truth: {sum(1 for gt in ground_truths if len(gt) > 0)}")
        
        # Build embeddings
        print("\n[2/4] Building embeddings (using cache)...")
        cache_dir = Path("./reports/online_mutation_test/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = EmbeddingCache(cache_dir, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        
        corpus_embeddings = cache.encode(corpus_texts_list, f"{self.dataset_name}_corpus_{total_needed}")
        query_embeddings = cache.encode(query_texts_list, f"{self.dataset_name}_queries")
        
        print(f"  Corpus embeddings: {corpus_embeddings.shape}")
        print(f"  Query embeddings: {query_embeddings.shape}")
        
        return corpus_embeddings, query_embeddings, ground_truths, qrels_list, corpus_ids_list, query_ids_list
    
    def build_initial_index(self):
        """Build initial MCN index with 100k vectors."""
        print(f"\n[3/4] Building initial index with {self.initial_size:,} vectors...")
        
        # Split corpus into initial + waves
        # Adjust initial_size if dataset is smaller
        actual_initial_size = min(self.initial_size, len(self.corpus_embeddings))
        if actual_initial_size < self.initial_size:
            print(f"  Warning: Dataset has only {len(self.corpus_embeddings):,} vectors, using {actual_initial_size:,} as initial size")
            self.initial_size = actual_initial_size
        
        initial_vectors = self.corpus_embeddings[:actual_initial_size]
        initial_metadata = [
            {"original_idx": i, "id": self.corpus_ids[i]}
            for i in range(actual_initial_size)
        ]
        
        # Create MCN
        self.mcn = MCNLayer(
            dim=self.dim,
            hot_buffer_size=50,
            beam_size=200,
            target_cluster_size=15,
            max_cluster_size=64,
        )
        
        # Add vectors
        print(f"  Adding {len(initial_vectors):,} vectors...")
        self.mcn.add(initial_vectors, initial_metadata)
        
        # Finalize index (use longer timeout for 100K+ vectors)
        print("  Finalizing index...")
        finalize_start = time.time()
        # Use 600s timeout for 100K vectors, scale up for larger
        timeout = 600 if actual_initial_size <= 100000 else 1800
        self.mcn.finalize_index(timeout_s=timeout)
        finalize_time = time.time() - finalize_start
        print(f"  Index finalized in {finalize_time:.2f}s")
        
        # Measure baseline metrics
        print("\n  Measuring baseline metrics...")
        baseline_metrics = self.measure_metrics()
        self.baseline_recall = baseline_metrics["recall@10"]
        self.baseline_p95_latency = baseline_metrics["latency_p95"]
        
        print(f"  Baseline Recall@10: {self.baseline_recall:.4f}")
        print(f"  Baseline p95 Latency: {self.baseline_p95_latency:.2f}ms")
    
    def measure_metrics(self) -> Dict:
        """Measure current metrics (recall, latency)."""
        if self.mcn is None:
            return {}
        
        # Run queries and measure latency
        latencies = []
        predictions = []
        
        for i, query in enumerate(self.query_embeddings):
            try:
                start = time.time()
                results, scores = self.mcn.search(query, 100)
                latency = (time.time() - start) * 1000  # ms
                latencies.append(latency)
                
                # Convert to doc IDs
                pred_ids = [str(r.get("id", r.get("original_idx", idx))) for idx, r in enumerate(results[:100])]
                predictions.append(pred_ids)
            except Exception as e:
                print(f"  Error in query {i}: {e}")
                self.error_count += 1
                latencies.append(0.0)
                predictions.append([])
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, self.ground_truths, self.qrels_list, k_values=[10, 100])
        latency_stats = calculate_latency_stats(latencies)
        
        return {
            "recall@10": metrics["recall@10"],
            "recall@100": metrics["recall@100"],
            "latency_p50": latency_stats["p50"],
            "latency_p95": latency_stats["p95"],
            "latency_p99": latency_stats["p99"],
            "latency_mean": latency_stats["mean"],
            "error_count": self.error_count,
        }
    
    def query_loop(self):
        """Continuous query loop at target QPS."""
        query_interval = 1.0 / self.query_qps
        query_idx = 0
        
        print(f"\n  Starting query loop at {self.query_qps} QPS...")
        
        while not self.stop_flag.is_set():
            query_start = time.time()
            
            # Get query (round-robin)
            query = self.query_embeddings[query_idx % len(self.query_embeddings)]
            
            try:
                # Search
                search_start = time.time()
                results, scores = self.mcn.search(query, 10)
                search_time = (time.time() - search_start) * 1000  # ms
                
                # Record latency
                with self.metrics_lock:
                    self.latency_history.append(search_time)
                    self.total_queries += 1
                
            except Exception as e:
                print(f"  Query error: {e}")
                with self.metrics_lock:
                    self.error_count += 1
            
            # Maintain QPS
            elapsed = time.time() - query_start
            sleep_time = max(0, query_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            query_idx += 1
    
    def ingestion_loop(self):
        """Ingestion loop: add vectors and compress every N seconds."""
        print(f"\n  Starting ingestion loop: {self.num_waves} waves of {self.wave_size:,} vectors...")
        
        # Get wave vectors (from remaining corpus)
        wave_start_idx = self.initial_size
        total_waves = self.num_waves
        
        for wave_num in range(total_waves):
            if self.stop_flag.is_set():
                break
            
            # Wait for interval (except first wave)
            if wave_num > 0:
                time.sleep(self.wave_interval)
            
            print(f"\n  Wave {wave_num + 1}/{total_waves}: Ingesting {self.wave_size:,} vectors...")
            
            # Get vectors for this wave
            wave_end_idx = wave_start_idx + self.wave_size
            wave_vectors = self.corpus_embeddings[wave_start_idx:wave_end_idx]
            wave_metadata = [
                {"original_idx": i, "id": self.corpus_ids[i]}
                for i in range(wave_start_idx, wave_end_idx)
            ]
            
            # Ingest
            ingest_start = time.time()
            try:
                self.mcn.add(wave_vectors, wave_metadata)
                ingest_time = time.time() - ingest_start
                self.ingestion_times.append(ingest_time)
                print(f"    Ingested in {ingest_time:.3f}s")
            except Exception as e:
                print(f"    Ingestion error: {e}")
                self.error_count += 1
                continue
            
            # Compress (finalize index)
            print(f"    Compressing index...")
            compress_start = time.time()
            try:
                # Use appropriate timeout based on total vectors
                total_vectors = wave_end_idx
                timeout = 600 if total_vectors <= 100000 else 1800
                self.mcn.finalize_index(timeout_s=timeout)
                compress_time = time.time() - compress_start
                self.compression_times.append(compress_time)
                print(f"    Compressed in {compress_time:.2f}s")
            except Exception as e:
                print(f"    Compression error: {e}")
                self.error_count += 1
                continue
            
            # Measure metrics after compression
            with self.metrics_lock:
                metrics = self.measure_metrics()
                metrics["wave"] = wave_num + 1
                metrics["total_vectors"] = wave_end_idx
                metrics["compression_time"] = compress_time
                metrics["ingestion_time"] = ingest_time
                self.metrics_history.append(metrics)
                
                print(f"    Recall@10: {metrics['recall@10']:.4f} (baseline: {self.baseline_recall:.4f})")
                print(f"    p95 Latency: {metrics['latency_p95']:.2f}ms (baseline: {self.baseline_p95_latency:.2f}ms)")
            
            wave_start_idx = wave_end_idx
        
        print(f"\n  Ingestion complete. Total waves: {total_waves}")
    
    def run(self):
        """Run the online mutation test."""
        print("="*80)
        print("TEST 1: Compression While Serving Queries (Online Mutation Test)")
        print("="*80)
        
        # Load dataset
        (self.corpus_embeddings, self.query_embeddings, self.ground_truths,
         self.qrels_list, self.corpus_ids, self.query_ids) = self.load_dataset()
        
        # Build initial index
        self.build_initial_index()
        
        # Start query loop
        print("\n[4/4] Starting concurrent query and ingestion...")
        self.query_thread = threading.Thread(target=self.query_loop, daemon=True)
        self.query_thread.start()
        
        # Wait a bit for queries to start
        time.sleep(1.0)
        
        # Start ingestion loop
        self.ingestion_thread = threading.Thread(target=self.ingestion_loop, daemon=True)
        self.ingestion_thread.start()
        
        # Wait for ingestion to complete
        self.ingestion_thread.join()
        
        # Let queries run a bit more
        time.sleep(2.0)
        
        # Stop query loop
        self.stop_flag.set()
        time.sleep(0.5)
        
        # Final metrics
        print("\n  Measuring final metrics...")
        final_metrics = self.measure_metrics()
        final_metrics["wave"] = "final"
        final_metrics["total_vectors"] = self.initial_size + (self.wave_size * self.num_waves)
        self.metrics_history.append(final_metrics)
        
        # Store final_metrics for report generation
        self.final_metrics = final_metrics
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate test report."""
        output_dir = Path("./reports/online_mutation_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get final metrics
        final_metrics = self.final_metrics if hasattr(self, 'final_metrics') else self.metrics_history[-1]
        
        # Calculate summary statistics
        recall_drops = []
        latency_increases = []
        
        for metrics in self.metrics_history:
            if metrics.get("wave") != "final" and self.baseline_recall:
                recall_drop = self.baseline_recall - metrics["recall@10"]
                recall_drops.append(recall_drop)
            
            if metrics.get("wave") != "final" and self.baseline_p95_latency:
                latency_increase = ((metrics["latency_p95"] - self.baseline_p95_latency) / 
                                   self.baseline_p95_latency) * 100
                latency_increases.append(latency_increase)
        
        max_recall_drop = max(recall_drops) if recall_drops else 0.0
        max_latency_increase = max(latency_increases) if latency_increases else 0.0
        avg_compression_time = np.mean(self.compression_times) if self.compression_times else 0.0
        avg_ingestion_time = np.mean(self.ingestion_times) if self.ingestion_times else 0.0
        
        # Generate report
        report = {
            "test_name": "Online Mutation Test",
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_name,
            "initial_size": self.initial_size,
            "wave_size": self.wave_size,
            "num_waves": self.num_waves,
            "wave_interval": self.wave_interval,
            "query_qps": self.query_qps,
            "baseline": {
                "recall@10": self.baseline_recall,
                "p95_latency_ms": self.baseline_p95_latency,
            },
            "final": {
                "recall@10": final_metrics["recall@10"],
                "p95_latency_ms": final_metrics["latency_p95"],
                "total_vectors": final_metrics["total_vectors"],
            },
            "summary": {
                "max_recall_drop": max_recall_drop,
                "max_latency_increase_percent": max_latency_increase,
                "total_queries": self.total_queries,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.total_queries, 1),
                "avg_compression_time_s": avg_compression_time,
                "avg_ingestion_time_s": avg_ingestion_time,
            },
            "metrics_history": self.metrics_history,
        }
        
        # Save JSON
        with open(output_dir / "results.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown
        md_lines = [
            "# TEST 1: Compression While Serving Queries (Online Mutation Test)",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset**: {self.dataset_name}",
            f"**Initial Vectors**: {self.initial_size:,}",
            f"**Wave Size**: {self.wave_size:,} vectors",
            f"**Number of Waves**: {self.num_waves}",
            f"**Wave Interval**: {self.wave_interval}s",
            f"**Query QPS**: {self.query_qps}",
            "",
            "## Test Results",
            "",
            "### Baseline (Before Mutation)",
            "",
            f"- Recall@10: {self.baseline_recall:.4f}",
            f"- p95 Latency: {self.baseline_p95_latency:.2f}ms",
            "",
            "### Final (After All Mutations)",
            "",
            f"- Recall@10: {final_metrics['recall@10']:.4f}",
            f"- p95 Latency: {final_metrics['latency_p95']:.2f}ms",
            f"- Total Vectors: {final_metrics['total_vectors']:,}",
            "",
            "### Summary Statistics",
            "",
            f"- **Max Recall Drop**: {max_recall_drop:.4f} ({'✅ PASS' if max_recall_drop <= 0.01 else '❌ FAIL'} - target ≤ 1%)",
            f"- **Max Latency Increase**: {max_latency_increase:.1f}% ({'✅ PASS' if max_latency_increase <= 20 else '❌ FAIL'} - target ≤ 20%)",
            f"- **Total Queries**: {self.total_queries:,}",
            f"- **Error Count**: {self.error_count} ({'✅ PASS' if self.error_count == 0 else '❌ FAIL'} - target = 0)",
            f"- **Error Rate**: {report['summary']['error_rate']:.4f}",
            f"- **Avg Compression Time**: {avg_compression_time:.2f}s",
            f"- **Avg Ingestion Time**: {avg_ingestion_time:.3f}s",
            "",
            "## Metrics Over Time",
            "",
            "| Wave | Total Vectors | Recall@10 | p95 Latency (ms) | Compression Time (s) |",
            "|------|---------------|-----------|-------------------|----------------------|",
        ]
        
        for metrics in self.metrics_history:
            if metrics.get("wave") != "final":
                md_lines.append(
                    f"| {metrics['wave']} | {metrics['total_vectors']:,} | "
                    f"{metrics['recall@10']:.4f} | {metrics['latency_p95']:.2f} | "
                    f"{metrics['compression_time']:.2f} |"
                )
        
        md_lines.append(
            f"| Final | {final_metrics['total_vectors']:,} | "
            f"{final_metrics['recall@10']:.4f} | {final_metrics['latency_p95']:.2f} | - |"
        )
        
        md_lines.extend([
            "",
            "## Test Assessment",
            "",
        ])
        
        # Assessment
        all_pass = (
            max_recall_drop <= 0.01 and
            max_latency_increase <= 20 and
            self.error_count == 0
        )
        
        if all_pass:
            md_lines.append("✅ **TEST PASSED**: All criteria met")
        else:
            md_lines.append("❌ **TEST FAILED**: Some criteria not met")
        
        md_lines.extend([
            "",
            f"- Recall drop ≤ 1%: {'✅' if max_recall_drop <= 0.01 else '❌'} (actual: {max_recall_drop:.4f})",
            f"- Latency increase ≤ 20%: {'✅' if max_latency_increase <= 20 else '❌'} (actual: {max_latency_increase:.1f}%)",
            f"- Zero crashes: {'✅' if self.error_count == 0 else '❌'} (actual: {self.error_count})",
            f"- No blocking during compression: {'✅' if avg_compression_time < 10 else '⚠️'} (avg: {avg_compression_time:.2f}s)",
            "",
        ])
        
        with open(output_dir / "ONLINE_MUTATION_REPORT.md", 'w') as f:
            f.write('\n'.join(md_lines))
        
        print(f"\n{'='*80}")
        print("Online Mutation Test Complete!")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"  - ONLINE_MUTATION_REPORT.md")
        print(f"  - results.json")
        print(f"{'='*80}\n")
        
        # Print summary
        print("SUMMARY:")
        print(f"  Max Recall Drop: {max_recall_drop:.4f} ({'✅ PASS' if max_recall_drop <= 0.01 else '❌ FAIL'})")
        print(f"  Max Latency Increase: {max_latency_increase:.1f}% ({'✅ PASS' if max_latency_increase <= 20 else '❌ FAIL'})")
        print(f"  Errors: {self.error_count} ({'✅ PASS' if self.error_count == 0 else '❌ FAIL'})")
        print(f"  Total Queries: {self.total_queries:,}")
        print()


def main():
    """Run online mutation test."""
    # Use MS MARCO (large dataset) or adjust for smaller datasets
    test = OnlineMutationTest(
        dataset_name="msmarco",  # Large dataset with >100k docs
        initial_size=100000,
        query_count=200,
        wave_size=10000,
        num_waves=5,
        wave_interval=5.0,
        query_qps=20.0,
        seed=42
    )
    
    test.run()


if __name__ == "__main__":
    main()

