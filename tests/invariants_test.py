# tests/invariants_test.py
"""
Invariant tests for MCN v1.
Tests critical invariants that must always hold.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from mcn import MCNLayer


def test_self_match_invariant():
    """
    Invariant: If a query vector exists in memory, it should appear in results.
    """
    dim = 128
    mcn = MCNLayer(dim=dim, use_background_thread=False)
    
    # Add vectors
    n = 100
    vectors = np.random.randn(n, dim).astype("float32")
    metadata = [{"original_idx": i, "id": f"vec_{i}"} for i in range(n)]
    
    mcn.add(vectors, metadata)
    mcn.finalize_index(expected_count=n)
    
    # Query with a vector that exists
    query_idx = 50
    query = vectors[query_idx]
    
    results, scores = mcn.search(query, k=10)
    
    # Check that query vector appears in results
    result_ids = [r.get("original_idx") for r in results]
    assert query_idx in result_ids, f"Self-match invariant violated: query vector {query_idx} not in results"
    
    # Check that it's in top-1
    assert result_ids[0] == query_idx, f"Self-match invariant violated: query vector not ranked #1"


def test_no_duplicate_storage():
    """
    Invariant: Each original_idx should appear exactly once in child store.
    """
    dim = 128
    mcn = MCNLayer(dim=dim, use_background_thread=False)
    
    # Add vectors
    n = 100
    vectors = np.random.randn(n, dim).astype("float32")
    metadata = [{"original_idx": i, "id": f"vec_{i}"} for i in range(n)]
    
    mcn.add(vectors, metadata)
    mcn.finalize_index(expected_count=n)
    
    # Check child store
    child_meta = mcn.child_store.child_meta
    original_indices = [m.get("original_idx") for m in child_meta]
    
    # No duplicates
    assert len(original_indices) == len(set(original_indices)), "Duplicate storage detected"
    
    # All original indices present
    assert set(original_indices) == set(range(n)), "Missing original indices"


def test_mapping_size_invariant():
    """
    Invariant: Mapping size should equal ingested size.
    """
    dim = 128
    mcn = MCNLayer(dim=dim, use_background_thread=False)
    
    # Add vectors
    n = 100
    vectors = np.random.randn(n, dim).astype("float32")
    metadata = [{"original_idx": i, "id": f"vec_{i}"} for i in range(n)]
    
    mcn.add(vectors, metadata)
    mcn.finalize_index(expected_count=n)
    
    # Check sizes
    child_store_size = mcn.child_store.size()
    mapping_size = len(mcn.child_store.orig_idx_to_pos)
    
    assert child_store_size == n, f"Child store size {child_store_size} != ingested size {n}"
    assert mapping_size == n, f"Mapping size {mapping_size} != ingested size {n}"


def test_stable_snapshot():
    """
    Invariant: Search results should be stable under concurrent queries.
    """
    dim = 128
    mcn = MCNLayer(dim=dim, use_background_thread=False)
    
    # Add vectors
    n = 100
    vectors = np.random.randn(n, dim).astype("float32")
    metadata = [{"original_idx": i, "id": f"vec_{i}"} for i in range(n)]
    
    mcn.add(vectors, metadata)
    mcn.finalize_index(expected_count=n)
    
    # Run same query multiple times
    query = vectors[0]
    results1, scores1 = mcn.search(query, k=10)
    results2, scores2 = mcn.search(query, k=10)
    
    # Results should be identical
    ids1 = [r.get("original_idx") for r in results1]
    ids2 = [r.get("original_idx") for r in results2]
    
    assert ids1 == ids2, "Search results not stable"
    assert np.allclose(scores1, scores2), "Search scores not stable"


def test_cluster_size_cap():
    """
    Invariant: No cluster should exceed max_cluster_size.
    """
    dim = 128
    mcn = MCNLayer(dim=dim, use_background_thread=False)
    
    # Add many vectors (will create large clusters)
    n = 1000
    vectors = np.random.randn(n, dim).astype("float32")
    metadata = [{"original_idx": i, "id": f"vec_{i}"} for i in range(n)]
    
    mcn.add(vectors, metadata)
    mcn.finalize_index(expected_count=n)
    
    # Check cluster sizes
    cluster_offsets = mcn.cluster_offsets
    max_cluster_size = 0
    
    for i in range(len(cluster_offsets) - 1):
        cluster_size = cluster_offsets[i + 1] - cluster_offsets[i]
        max_cluster_size = max(max_cluster_size, cluster_size)
    
    assert max_cluster_size <= mcn.config.max_cluster_size, \
        f"Cluster size {max_cluster_size} exceeds max_cluster_size {mcn.config.max_cluster_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

