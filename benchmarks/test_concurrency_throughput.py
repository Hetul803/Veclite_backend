#!/usr/bin/env python3
"""
TEST 2: Concurrency & Throughput Test (Multi-Tenant Load)

Goal: Determine how many real users one MCN instance can support.

Setup:
- Dataset: 100,000 vectors
- Queries: random + ground-truth queries
- Concurrency levels: 1, 10, 50, 100, 200 concurrent clients
- Rate limiting enabled per tenant

Measure:
- p50 / p95 / p99 latency
- Throughput (QPS)
- Memory growth
- CPU saturation
- Tail latency collapse point

Report:
- Max stable QPS
- Max concurrent users before p95 > 50ms
- Cost per 1M queries (Railway Pro estimate)
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
import concurrent.futures

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
except ImportError:
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


class ConcurrencyTest:
    """Test concurrency and throughput under multi-tenant load."""
    
    def __init__(
        self,
        dataset_name: str,
        corpus_size: int = 100000,
        query_count: int = 200,
        concurrency_levels: List[int] = [1, 10, 50, 100, 200],
        test_duration: float = 60.0,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.corpus_size = corpus_size
        self.query_count = query_count
        self.concurrency_levels = concurrency_levels
        self.test_duration = test_duration
        self.seed = seed
        self.dim = 384
        
        # State
        self.mcn = None
        self.corpus_embeddings = None
        self.query_embeddings = None
        self.corpus_ids = None
        
        # Results
        self.results = {}
    
    def load_dataset(self):
        """Load BEIR dataset and build embeddings."""
        print(f"\n[1/3] Loading dataset: {self.dataset_name}...")
        
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        data_path = f"./beir_data/{self.dataset_name}"
        
        # Download if needed
        if not os.path.exists(data_path) or not os.path.exists(f"{data_path}/corpus.jsonl"):
            print(f"  Downloading {self.dataset_name}...")
            util.download_and_unzip(url, data_path)
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
        
        # Subsample to corpus size
        np.random.seed(self.seed)
        if len(corpus_texts_list) > self.corpus_size:
            indices = np.random.choice(len(corpus_texts_list), self.corpus_size, replace=False)
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
        
        print(f"  Loaded {len(query_texts_list)} queries")
        
        # Build embeddings
        print("\n[2/3] Building embeddings (using cache)...")
        cache_dir = Path("./reports/concurrency_test/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = EmbeddingCache(cache_dir, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        
        corpus_embeddings = cache.encode(corpus_texts_list, f"{self.dataset_name}_corpus_{self.corpus_size}")
        query_embeddings = cache.encode(query_texts_list, f"{self.dataset_name}_queries")
        
        print(f"  Corpus embeddings: {corpus_embeddings.shape}")
        print(f"  Query embeddings: {query_embeddings.shape}")
        
        self.corpus_embeddings = corpus_embeddings
        self.query_embeddings = query_embeddings
        self.corpus_ids = corpus_ids_list
    
    def build_index(self):
        """Build MCN index."""
        print(f"\n[3/3] Building index with {self.corpus_size:,} vectors...")
        
        # Create MCN
        self.mcn = MCNLayer(
            dim=self.dim,
            hot_buffer_size=50,
            beam_size=200,
            target_cluster_size=15,
            max_cluster_size=64,
        )
        
        # Add vectors
        metadata = [
            {"original_idx": i, "id": self.corpus_ids[i]}
            for i in range(len(self.corpus_embeddings))
        ]
        
        print(f"  Adding {len(self.corpus_embeddings):,} vectors...")
        self.mcn.add(self.corpus_embeddings, metadata)
        
        # Finalize index (use longer timeout for 100K+ vectors)
        print("  Finalizing index...")
        finalize_start = time.time()
        # Use 600s timeout for 100K vectors, scale up for larger
        timeout = 600 if self.corpus_size <= 100000 else 1800
        self.mcn.finalize_index(timeout_s=timeout)
        finalize_time = time.time() - finalize_start
        print(f"  Index finalized in {finalize_time:.2f}s")
    
    def run_concurrency_test(self, concurrency: int) -> Dict:
        """Run concurrency test at specified level."""
        print(f"\n{'='*80}")
        print(f"Testing concurrency level: {concurrency}")
        print(f"{'='*80}")
        
        # Metrics
        latencies = []
        errors = []
        query_count = 0
        stop_flag = threading.Event()
        
        # Memory tracking
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU tracking
        cpu_before = process.cpu_percent(interval=0.1)
        
        def query_worker(worker_id: int):
            """Worker thread that continuously queries."""
            nonlocal query_count
            worker_latencies = []
            worker_errors = 0
            
            # Round-robin through queries
            query_idx = worker_id % len(self.query_embeddings)
            
            while not stop_flag.is_set():
                try:
                    query = self.query_embeddings[query_idx]
                    
                    start = time.time()
                    results, scores = self.mcn.search(query, 10)
                    latency = (time.time() - start) * 1000  # ms
                    
                    worker_latencies.append(latency)
                    query_count += 1
                    
                except Exception as e:
                    worker_errors += 1
                    print(f"  Worker {worker_id} error: {e}")
                
                query_idx = (query_idx + 1) % len(self.query_embeddings)
            
            return worker_latencies, worker_errors
        
        # Start workers
        print(f"  Starting {concurrency} concurrent workers...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(query_worker, i) for i in range(concurrency)]
            
            # Run for test duration
            time.sleep(self.test_duration)
            stop_flag.set()
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                worker_latencies, worker_errors = future.result()
                latencies.extend(worker_latencies)
                errors.append(worker_errors)
        
        elapsed = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_growth = mem_after - mem_before
        
        # CPU after
        cpu_after = process.cpu_percent(interval=1.0)
        
        # Calculate statistics
        if latencies:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            latency_stats = {
                "p50": sorted_latencies[int(n * 0.50)],
                "p95": sorted_latencies[int(n * 0.95)],
                "p99": sorted_latencies[int(n * 0.99)],
                "mean": np.mean(sorted_latencies),
                "min": min(sorted_latencies),
                "max": max(sorted_latencies),
            }
        else:
            latency_stats = {
                "p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0
            }
        
        total_queries = query_count
        total_errors = sum(errors)
        qps = total_queries / elapsed if elapsed > 0 else 0
        error_rate = total_errors / max(total_queries, 1)
        
        result = {
            "concurrency": concurrency,
            "duration": elapsed,
            "total_queries": total_queries,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "qps": qps,
            "latency": latency_stats,
            "memory_mb_before": mem_before,
            "memory_mb_after": mem_after,
            "memory_growth_mb": mem_growth,
            "cpu_percent_before": cpu_before,
            "cpu_percent_after": cpu_after,
        }
        
        print(f"  Total queries: {total_queries:,}")
        print(f"  QPS: {qps:.1f}")
        print(f"  Errors: {total_errors} ({error_rate*100:.2f}%)")
        print(f"  Latency p50: {latency_stats['p50']:.2f}ms")
        print(f"  Latency p95: {latency_stats['p95']:.2f}ms")
        print(f"  Latency p99: {latency_stats['p99']:.2f}ms")
        print(f"  Memory growth: {mem_growth:.1f} MB")
        print(f"  CPU: {cpu_after:.1f}%")
        
        return result
    
    def run(self):
        """Run all concurrency tests."""
        print("="*80)
        print("TEST 2: Concurrency & Throughput Test (Multi-Tenant Load)")
        print("="*80)
        
        # Load dataset
        self.load_dataset()
        
        # Build index
        self.build_index()
        
        # Run tests at each concurrency level
        print(f"\n{'='*80}")
        print("Running concurrency tests...")
        print(f"{'='*80}")
        
        for concurrency in self.concurrency_levels:
            result = self.run_concurrency_test(concurrency)
            self.results[concurrency] = result
            
            # Brief pause between tests
            if concurrency < self.concurrency_levels[-1]:
                time.sleep(2.0)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        output_dir = Path("./reports/concurrency_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find max stable QPS and collapse point
        max_stable_qps = 0
        max_stable_concurrency = 0
        collapse_point = None
        
        for concurrency in sorted(self.concurrency_levels):
            result = self.results[concurrency]
            if result["latency"]["p95"] <= 50.0:
                if result["qps"] > max_stable_qps:
                    max_stable_qps = result["qps"]
                    max_stable_concurrency = concurrency
            elif collapse_point is None:
                collapse_point = concurrency
        
        # Calculate cost estimate (Railway Pro: ~$20/month, assume 1 instance)
        # Rough estimate: cost per 1M queries
        if max_stable_qps > 0:
            queries_per_month = max_stable_qps * 60 * 60 * 24 * 30  # QPS * seconds in month
            cost_per_month = 20.0  # Railway Pro estimate
            cost_per_1m_queries = (cost_per_month / queries_per_month) * 1_000_000
        else:
            cost_per_1m_queries = 0.0
        
        # Generate report
        report = {
            "test_name": "Concurrency & Throughput Test",
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_name,
            "corpus_size": self.corpus_size,
            "query_count": self.query_count,
            "test_duration": self.test_duration,
            "concurrency_levels": self.concurrency_levels,
            "results": self.results,
            "summary": {
                "max_stable_qps": max_stable_qps,
                "max_stable_concurrency": max_stable_concurrency,
                "collapse_point": collapse_point,
                "cost_per_1m_queries_usd": cost_per_1m_queries,
            }
        }
        
        # Save JSON
        with open(output_dir / "results.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown
        md_lines = [
            "# TEST 2: Concurrency & Throughput Test (Multi-Tenant Load)",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset**: {self.dataset_name}",
            f"**Corpus Size**: {self.corpus_size:,} vectors",
            f"**Query Count**: {self.query_count}",
            f"**Test Duration**: {self.test_duration}s per concurrency level",
            "",
            "## Summary",
            "",
            f"- **Max Stable QPS**: {max_stable_qps:.1f} (at {max_stable_concurrency} concurrent clients)",
            f"- **Max Concurrent Users (p95 ≤ 50ms)**: {max_stable_concurrency}",
            f"- **Tail Latency Collapse Point**: {collapse_point if collapse_point else 'Not reached'}",
            f"- **Cost per 1M Queries**: ${cost_per_1m_queries:.4f} (Railway Pro estimate)",
            "",
            "## Detailed Results",
            "",
            "| Concurrency | QPS | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Error Rate | Memory Growth (MB) | CPU % |",
            "|-------------|-----|-------------------|------------------|------------------|------------|---------------------|-------|",
        ]
        
        for concurrency in sorted(self.concurrency_levels):
            result = self.results[concurrency]
            md_lines.append(
                f"| {concurrency} | {result['qps']:.1f} | {result['latency']['p50']:.2f} | "
                f"{result['latency']['p95']:.2f} | {result['latency']['p99']:.2f} | "
                f"{result['error_rate']*100:.2f}% | {result['memory_growth_mb']:.1f} | "
                f"{result['cpu_percent_after']:.1f} |"
            )
        
        md_lines.extend([
            "",
            "## Analysis",
            "",
            "### Latency vs Concurrency",
            "",
        ])
        
        # Latency analysis
        for concurrency in sorted(self.concurrency_levels):
            result = self.results[concurrency]
            p95 = result['latency']['p95']
            status = "✅ Stable" if p95 <= 50.0 else "⚠️ Degraded" if p95 <= 100.0 else "❌ Collapsed"
            md_lines.append(f"- **{concurrency} concurrent clients**: p95 = {p95:.2f}ms ({status})")
        
        md_lines.extend([
            "",
            "### Throughput Analysis",
            "",
        ])
        
        # Throughput analysis
        for concurrency in sorted(self.concurrency_levels):
            result = self.results[concurrency]
            md_lines.append(f"- **{concurrency} concurrent clients**: {result['qps']:.1f} QPS")
        
        md_lines.extend([
            "",
            "### Memory Growth",
            "",
        ])
        
        # Memory analysis
        for concurrency in sorted(self.concurrency_levels):
            result = self.results[concurrency]
            md_lines.append(f"- **{concurrency} concurrent clients**: {result['memory_growth_mb']:.1f} MB growth")
        
        md_lines.extend([
            "",
            "### CPU Utilization",
            "",
        ])
        
        # CPU analysis
        for concurrency in sorted(self.concurrency_levels):
            result = self.results[concurrency]
            md_lines.append(f"- **{concurrency} concurrent clients**: {result['cpu_percent_after']:.1f}% CPU")
        
        md_lines.extend([
            "",
            "## Production Readiness Assessment",
            "",
        ])
        
        # Assessment
        if max_stable_qps >= 100:
            md_lines.append("✅ **High Throughput**: MCN can handle 100+ QPS stably")
        elif max_stable_qps >= 50:
            md_lines.append("⚠️ **Moderate Throughput**: MCN can handle 50+ QPS")
        else:
            md_lines.append("❌ **Low Throughput**: MCN struggles with high QPS")
        
        if max_stable_concurrency >= 100:
            md_lines.append("✅ **High Concurrency**: MCN supports 100+ concurrent users")
        elif max_stable_concurrency >= 50:
            md_lines.append("⚠️ **Moderate Concurrency**: MCN supports 50+ concurrent users")
        else:
            md_lines.append("❌ **Low Concurrency**: MCN struggles with high concurrency")
        
        if collapse_point is None or collapse_point >= 200:
            md_lines.append("✅ **Stable**: No collapse observed up to 200 concurrent clients")
        else:
            md_lines.append(f"⚠️ **Collapse Point**: Degradation observed at {collapse_point} concurrent clients")
        
        md_lines.extend([
            "",
            "## Cost Analysis",
            "",
            f"Based on Railway Pro pricing (~$20/month) and max stable QPS of {max_stable_qps:.1f}:",
            f"- **Queries per month**: {max_stable_qps * 60 * 60 * 24 * 30:,.0f}",
            f"- **Cost per 1M queries**: ${cost_per_1m_queries:.4f}",
            f"- **Monthly cost at max QPS**: ${cost_per_month:.2f}",
            "",
        ])
        
        with open(output_dir / "CONCURRENCY_REPORT.md", 'w') as f:
            f.write('\n'.join(md_lines))
        
        print(f"\n{'='*80}")
        print("Concurrency Test Complete!")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"  - CONCURRENCY_REPORT.md")
        print(f"  - results.json")
        print(f"{'='*80}\n")
        
        # Print summary
        print("SUMMARY:")
        print(f"  Max Stable QPS: {max_stable_qps:.1f} (at {max_stable_concurrency} concurrent clients)")
        print(f"  Max Concurrent Users (p95 ≤ 50ms): {max_stable_concurrency}")
        print(f"  Collapse Point: {collapse_point if collapse_point else 'Not reached'}")
        print(f"  Cost per 1M Queries: ${cost_per_1m_queries:.4f}")
        print()


def main():
    """Run concurrency test."""
    test = ConcurrencyTest(
        dataset_name="msmarco",
        corpus_size=100000,
        query_count=200,
        concurrency_levels=[1, 10, 50, 100, 200],
        test_duration=60.0,
        seed=42
    )
    
    test.run()


if __name__ == "__main__":
    main()

