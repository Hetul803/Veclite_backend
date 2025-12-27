"""
Load test: 20 QPS search while finalize runs.
Asserts: error rate = 0, p95 stays under threshold.
"""
import numpy as np
import time
import threading
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

BASE_URL = "http://localhost:8000"
TEST_API_KEY = "load_test_tenant"
DIM = 384
TARGET_QPS = 20
TEST_DURATION = 60  # seconds
P95_THRESHOLD_MS = 100  # 100ms threshold


def generate_test_vectors(n: int, dim: int = DIM) -> np.ndarray:
    """Generate random test vectors."""
    return np.random.randn(n, dim).astype("float32")


def ingest_vectors(api_key: str, vectors: np.ndarray, batch_size: int = 100):
    """Ingest vectors in batches."""
    url = f"{BASE_URL}/add"
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        payload = {
            "api_key": api_key,
            "vectors": [
                {
                    "id": f"vec_{i+j}",
                    "values": vec.tolist(),
                    "metadata": {"original_idx": i+j}
                }
                for j, vec in enumerate(batch)
            ]
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"⚠️  Ingest error: {response.status_code} - {response.text}")
        time.sleep(0.1)  # Small delay between batches


def search_worker(api_key: str, query: np.ndarray, results: List[Tuple[float, bool]]):
    """Worker that performs searches and records latency/errors."""
    url = f"{BASE_URL}/search"
    payload = {
        "api_key": api_key,
        "vector": query.tolist(),
        "k": 10
    }
    
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=5)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        success = response.status_code == 200
        results.append((elapsed, success))
        
        if not success:
            print(f"⚠️  Search error: {response.status_code} - {response.text}")
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        results.append((elapsed, False))
        print(f"⚠️  Search exception: {e}")


def run_load_test():
    """Run load test: 20 QPS search while finalize runs."""
    print("="*60)
    print("Load Test: Concurrent Search During Finalize")
    print("="*60)
    print(f"Target QPS: {TARGET_QPS}")
    print(f"Test Duration: {TEST_DURATION}s")
    print(f"P95 Threshold: {P95_THRESHOLD_MS}ms")
    print()
    
    # Step 1: Ingest initial vectors
    print("Step 1: Ingesting 20K vectors...")
    n_vectors = 20_000
    vectors = generate_test_vectors(n_vectors)
    ingest_vectors(TEST_API_KEY, vectors)
    print(f"✅ Ingested {n_vectors} vectors")
    
    # Step 2: Start finalize (non-blocking)
    print("\nStep 2: Starting finalize (build)...")
    finalize_url = f"{BASE_URL}/finalize"
    finalize_response = requests.post(
        finalize_url,
        json={"api_key": TEST_API_KEY, "timeout_s": 600.0},
        timeout=10
    )
    
    if finalize_response.status_code != 200:
        print(f"❌ Finalize failed: {finalize_response.status_code} - {finalize_response.text}")
        return False
    
    build_id = finalize_response.json().get("build_id")
    print(f"✅ Build started: {build_id}")
    
    # Step 3: Run searches at 20 QPS for 60 seconds
    print(f"\nStep 3: Running searches at {TARGET_QPS} QPS for {TEST_DURATION}s...")
    results: List[Tuple[float, bool]] = []
    query = generate_test_vectors(1)[0]  # Single query vector
    
    start_time = time.time()
    interval = 1.0 / TARGET_QPS  # Time between requests
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        request_count = 0
        
        while time.time() - start_time < TEST_DURATION:
            # Submit search
            future = executor.submit(search_worker, TEST_API_KEY, query, results)
            futures.append(future)
            request_count += 1
            
            # Rate limit to TARGET_QPS
            time.sleep(interval)
        
        # Wait for all searches to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"⚠️  Future error: {e}")
    
    elapsed_time = time.time() - start_time
    actual_qps = len(results) / elapsed_time
    
    # Step 4: Analyze results
    print(f"\nStep 4: Analyzing results...")
    print(f"  Total searches: {len(results)}")
    print(f"  Actual QPS: {actual_qps:.2f}")
    print(f"  Test duration: {elapsed_time:.2f}s")
    
    # Calculate error rate
    successful = [r for _, success in results if success]
    error_count = len(results) - len(successful)
    error_rate = error_count / len(results) if results else 0
    
    print(f"  Errors: {error_count} ({error_rate*100:.2f}%)")
    
    # Calculate latency percentiles
    latencies = [lat for lat, _ in results if lat > 0]
    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
        
        print(f"  Latency p50: {p50:.2f}ms")
        print(f"  Latency p95: {p95:.2f}ms")
        print(f"  Latency p99: {p99:.2f}ms")
    else:
        p95 = float('inf')
        print("  ⚠️  No successful searches to calculate latency")
    
    # Step 5: Check finalize status
    print(f"\nStep 5: Checking finalize status...")
    status_url = f"{BASE_URL}/finalize/status?build_id={build_id}"
    status_response = requests.get(status_url, timeout=10)
    if status_response.status_code == 200:
        status_data = status_response.json()
        print(f"  Build status: {status_data.get('status')}")
        if status_data.get('status') == 'ready':
            print(f"  Vectors: {status_data.get('vectors')}")
            print(f"  Clusters: {status_data.get('clusters')}")
    
    # Step 6: Assertions
    print(f"\nStep 6: Assertions...")
    assert error_rate == 0, f"Error rate should be 0, got {error_rate*100:.2f}%"
    assert p95 < P95_THRESHOLD_MS, f"P95 latency should be < {P95_THRESHOLD_MS}ms, got {p95:.2f}ms"
    
    print("="*60)
    print("✅ Load test PASSED")
    print("="*60)
    print(f"  Error rate: {error_rate*100:.2f}% (target: 0%)")
    print(f"  P95 latency: {p95:.2f}ms (target: < {P95_THRESHOLD_MS}ms)")
    print("="*60)
    
    return True


if __name__ == "__main__":
    import sys
    try:
        success = run_load_test()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

