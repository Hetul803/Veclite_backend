#!/usr/bin/env python3
"""
Load test for MCN v1: Concurrent query performance.

Tests MCNLayer directly (no HTTP) with N threads x M queries each.
Measures p95 latency and error rate.

Usage:
    python scripts/load_test.py --threads 20 --queries 200
"""
import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcn import MCNLayer


class LoadTester:
    """Load tester for MCN."""
    
    def __init__(
        self,
        dim: int = 384,
        n_vectors: int = 10000,
        n_threads: int = 20,
        queries_per_thread: int = 200
    ):
        self.dim = dim
        self.n_vectors = n_vectors
        self.n_threads = n_threads
        self.queries_per_thread = queries_per_thread
        
        # Initialize MCN with test data
        print(f"Initializing MCN with {n_vectors} vectors...")
        self.mcn = MCNLayer(dim=dim, hot_buffer_size=50, use_background_thread=False)
        
        # Generate test vectors
        vectors = np.random.randn(n_vectors, dim).astype("float32")
        metadata = [{"original_idx": i, "id": f"vec_{i}"} for i in range(n_vectors)]
        
        # Ingest
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch_vecs = vectors[i:i+batch_size]
            batch_meta = metadata[i:i+batch_size]
            self.mcn.add(batch_vecs, batch_meta)
        
        # Finalize
        print("Finalizing index...")
        self.mcn.finalize_index(expected_count=n_vectors, timeout_s=120.0)
        print(f"Index built: {self.mcn.get_cold_index_size()} clusters")
        
        # Generate query vectors
        self.query_vectors = np.random.randn(n_threads * queries_per_thread, dim).astype("float32")
    
    def worker(self, thread_id: int, query_queue: Queue, result_queue: Queue):
        """Worker thread that runs queries."""
        latencies = []
        errors = 0
        
        for i in range(self.queries_per_thread):
            query_idx = thread_id * self.queries_per_thread + i
            query_vec = self.query_vectors[query_idx]
            
            try:
                # Search
                search_start = time.time()
                results, scores = self.mcn.search(query_vec, k=10)
                search_time = (time.time() - search_start) * 1000  # ms
                latencies.append(search_time)
                
                # Validate results
                if len(results) == 0:
                    errors += 1
                elif len(scores) != len(results):
                    errors += 1
                    
            except Exception as e:
                errors += 1
                print(f"Thread {thread_id} query {i} error: {e}")
        
        result_queue.put({
            "thread_id": thread_id,
            "latencies": latencies,
            "errors": errors
        })
    
    def run_load_test(self) -> Dict:
        """Run load test and return results."""
        print(f"\n{'='*80}")
        print(f"Load Test: {self.n_threads} threads × {self.queries_per_thread} queries")
        print(f"{'='*80}\n")
        
        # Create queues
        query_queue = Queue()
        result_queue = Queue()
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for thread_id in range(self.n_threads):
            thread = Thread(target=self.worker, args=(thread_id, query_queue, result_queue))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        all_latencies = []
        total_errors = 0
        
        for _ in range(self.n_threads):
            result = result_queue.get()
            all_latencies.extend(result["latencies"])
            total_errors += result["errors"]
        
        # Compute statistics
        total_queries = self.n_threads * self.queries_per_thread
        error_rate = total_errors / total_queries if total_queries > 0 else 0.0
        
        if len(all_latencies) > 0:
            p50_latency = np.percentile(all_latencies, 50)
            p95_latency = np.percentile(all_latencies, 95)
            p99_latency = np.percentile(all_latencies, 99)
            avg_latency = np.mean(all_latencies)
            min_latency = np.min(all_latencies)
            max_latency = np.max(all_latencies)
        else:
            p50_latency = p95_latency = p99_latency = avg_latency = min_latency = max_latency = 0.0
        
        qps = total_queries / total_time if total_time > 0 else 0.0
        
        results = {
            "total_queries": total_queries,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "total_time": total_time,
            "qps": qps,
            "latency_p50": p50_latency,
            "latency_p95": p95_latency,
            "latency_p99": p99_latency,
            "latency_avg": avg_latency,
            "latency_min": min_latency,
            "latency_max": max_latency,
            "n_threads": self.n_threads,
            "queries_per_thread": self.queries_per_thread
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print load test results."""
        print(f"\n{'='*80}")
        print("Load Test Results")
        print(f"{'='*80}\n")
        
        print(f"Configuration:")
        print(f"  - Threads: {results['n_threads']}")
        print(f"  - Queries per thread: {results['queries_per_thread']}")
        print(f"  - Total queries: {results['total_queries']}")
        print(f"  - Total time: {results['total_time']:.2f}s")
        print(f"  - QPS: {results['qps']:.2f}")
        
        print(f"\nLatency (ms):")
        print(f"  - p50: {results['latency_p50']:.2f}")
        print(f"  - p95: {results['latency_p95']:.2f}")
        print(f"  - p99: {results['latency_p99']:.2f}")
        print(f"  - avg: {results['latency_avg']:.2f}")
        print(f"  - min: {results['latency_min']:.2f}")
        print(f"  - max: {results['latency_max']:.2f}")
        
        print(f"\nErrors:")
        print(f"  - Total errors: {results['total_errors']}")
        print(f"  - Error rate: {results['error_rate']*100:.2f}%")
        
        print(f"\n{'='*80}\n")
        
        # Validation
        if results['error_rate'] > 0.01:
            print(f"⚠️  Warning: Error rate {results['error_rate']*100:.2f}% exceeds 1% threshold")
        else:
            print(f"✅ Error rate {results['error_rate']*100:.2f}% is acceptable")
        
        if results['latency_p95'] > 100:
            print(f"⚠️  Warning: p95 latency {results['latency_p95']:.2f}ms exceeds 100ms threshold")
        else:
            print(f"✅ p95 latency {results['latency_p95']:.2f}ms is acceptable")


def main():
    parser = argparse.ArgumentParser(description="Load test for MCN v1")
    parser.add_argument("--threads", type=int, default=20, help="Number of threads")
    parser.add_argument("--queries", type=int, default=200, help="Queries per thread")
    parser.add_argument("--vectors", type=int, default=10000, help="Number of vectors in index")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimension")
    
    args = parser.parse_args()
    
    # Create load tester
    tester = LoadTester(
        dim=args.dim,
        n_vectors=args.vectors,
        n_threads=args.threads,
        queries_per_thread=args.queries
    )
    
    # Run load test
    results = tester.run_load_test()
    
    # Print results
    tester.print_results(results)


if __name__ == "__main__":
    main()

