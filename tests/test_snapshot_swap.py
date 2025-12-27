"""
Tests for snapshot swap correctness and non-blocking searches.
"""
import numpy as np
import time
import threading
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcn import SnapshotManager


def test_snapshot_swap_no_vector_loss():
    """Test that snapshot swap doesn't lose vectors."""
    dim = 384
    mgr = SnapshotManager(dim=dim, hot_buffer_size=50)
    
    # Add vectors to write buffer
    n_vectors = 1000
    vectors = np.random.randn(n_vectors, dim).astype("float32")
    metadata = [{"id": f"vec_{i}", "original_idx": i} for i in range(n_vectors)]
    
    mgr.add(vectors, metadata)
    
    # Verify vectors are in write buffer
    assert mgr.write_buffer.size() == n_vectors
    
    # Start build
    build_id = mgr.start_build(timeout_s=300.0)
    
    # Finalize build
    build_result = mgr.finalize_build(build_id)
    assert build_result["status"] == "ready"
    assert build_result["vectors"] == n_vectors
    
    # Swap
    swap_result = mgr.swap_snapshot(build_id)
    assert swap_result["status"] == "swapped"
    assert swap_result["new_vectors"] == n_vectors
    
    # Verify active snapshot has all vectors
    assert mgr.active_snapshot.size() == n_vectors
    
    # Verify write buffer is cleared
    assert mgr.write_buffer.size() == 0


def test_searches_during_finalize_dont_block():
    """Test that searches during finalize don't block for > X seconds."""
    dim = 384
    mgr = SnapshotManager(dim=dim, hot_buffer_size=50)
    
    # Add initial vectors and create first snapshot
    n_initial = 5000
    vectors = np.random.randn(n_initial, dim).astype("float32")
    metadata = [{"id": f"vec_{i}", "original_idx": i} for i in range(n_initial)]
    mgr.add(vectors, metadata)
    
    # Build and swap first snapshot
    build_id1 = mgr.start_build(timeout_s=300.0)
    mgr.finalize_build(build_id1)
    mgr.swap_snapshot(build_id1)
    
    # Add more vectors
    n_new = 2000
    new_vectors = np.random.randn(n_new, dim).astype("float32")
    new_metadata = [{"id": f"vec_{i+n_initial}", "original_idx": i+n_initial} for i in range(n_new)]
    mgr.add(new_vectors, new_metadata)
    
    # Start new build
    build_id2 = mgr.start_build(timeout_s=300.0)
    
    # Perform searches while build is running (should not block)
    query = np.random.randn(dim).astype("float32")
    search_times = []
    
    def search_worker():
        for _ in range(10):
            start = time.time()
            results, scores = mgr.search(query, k=10)
            elapsed = time.time() - start
            search_times.append(elapsed)
            time.sleep(0.1)
    
    # Start search thread
    search_thread = threading.Thread(target=search_worker)
    search_thread.start()
    
    # Finalize build (this takes time, but shouldn't block searches)
    finalize_start = time.time()
    mgr.finalize_build(build_id2)
    finalize_time = time.time() - finalize_start
    
    search_thread.join()
    
    # Verify searches completed quickly (p95 < 1 second)
    search_times_sorted = sorted(search_times)
    p95_idx = int(len(search_times_sorted) * 0.95)
    p95_latency = search_times_sorted[p95_idx]
    
    assert p95_latency < 1.0, f"Searches blocked: p95 latency = {p95_latency:.3f}s"
    assert finalize_time > 0, "Finalize should take some time"
    
    print(f"✅ Searches during finalize: p95={p95_latency*1000:.1f}ms, finalize={finalize_time:.2f}s")


def test_snapshot_mapping_size_invariant():
    """Test that total vector count is preserved across swaps."""
    dim = 384
    mgr = SnapshotManager(dim=dim, hot_buffer_size=50)
    
    total_added = 0
    
    # Add vectors in waves
    for wave in range(3):
        n = 1000
        vectors = np.random.randn(n, dim).astype("float32")
        metadata = [{"id": f"wave{wave}_vec_{i}", "original_idx": total_added + i} for i in range(n)]
        mgr.add(vectors, metadata)
        total_added += n
        
        # Build and swap
        build_id = mgr.start_build(timeout_s=300.0)
        mgr.finalize_build(build_id)
        swap_result = mgr.swap_snapshot(build_id)
        
        # Verify total count
        active_size = mgr.active_snapshot.size() if mgr.active_snapshot else 0
        write_size = mgr.write_buffer.size()
        total_size = active_size + write_size
        
        assert total_size == total_added, f"Vector count mismatch: expected {total_added}, got {total_size}"
    
    print(f"✅ Mapping size invariant preserved: {total_added} vectors across 3 swaps")


if __name__ == "__main__":
    test_snapshot_swap_no_vector_loss()
    test_searches_during_finalize_dont_block()
    test_snapshot_mapping_size_invariant()
    print("✅ All snapshot swap tests passed!")

