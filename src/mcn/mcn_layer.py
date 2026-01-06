# src/mcn/mcn_layer_v1.py
"""
MCN v1: Clean, production-ready Memory Compression Networks.
Simplified architecture: ChildStore + ClusterIndex + Search.
No background threads, no FAISS mutation, deterministic.
"""
import numpy as np
import time
import threading
from typing import List, Tuple, Optional, Dict
from .config import MCNConfig
from .store import ChildStore
from .cluster import build_clusters
from .search import route, expand, rerank


class MCNLayer:
    """
    MCN v1: Clean architecture for production.
    
    Architecture:
    - Hot Buffer: Fast RAM for recent vectors
    - ChildStore: All original vectors (for reranking)
    - ClusterIndex: Super vectors + cluster structure
    - Search: Deterministic routing + exact rerank
    """
    
    def __init__(
        self,
        dim: int,
        hot_buffer_size: int = 50,
        similarity_threshold: float = 0.8,
        batch_size: int = 10,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        storage_backend=None,
        use_background_thread: bool = False,  # DISABLED in v1
        **kwargs
    ):
        """
        Initialize MCNLayer.
        
        Args:
            dim: Vector dimension
            hot_buffer_size: Max vectors in hot buffer
            similarity_threshold: (deprecated, kept for API compatibility)
            batch_size: (deprecated, kept for API compatibility)
            max_generations: (deprecated, kept for API compatibility)
            mutation_rate: (deprecated, kept for API compatibility)
            storage_backend: (optional) Storage backend for persistence
            use_background_thread: (ignored in v1, always False)
            **kwargs: Additional config options
        """
        self.dim = dim
        
        # Create config
        self.config = MCNConfig(
            dim=dim,
            hot_buffer_size=hot_buffer_size,
            **kwargs
        )
        
        # Hot Buffer: Fast RAM for recent vectors
        self.hot_buffer_vectors = np.empty((0, dim), dtype="float32")
        self.hot_buffer_meta: List[Dict] = []
        self._hot_buffer_lock = threading.Lock()
        
        # ChildStore: All original vectors
        self.child_store = ChildStore(dim, dtype=self.config.dtype_child)
        
        # ClusterIndex: Super vectors + structure
        self.super_vectors: np.ndarray = np.empty((0, dim), dtype="float32")
        self.cluster_offsets: np.ndarray = np.array([0], dtype="int32")
        self.cluster_child_ids: np.ndarray = np.array([], dtype="int32")
        self._cluster_lock = threading.Lock()
        
        # Storage backend (optional)
        self.storage_backend = storage_backend
        
        # State
        self._index_finalized = False
        self.events: List[Dict] = []
    
    def add(self, vecs: np.ndarray, meta_batch=None):
        """
        Add vectors to hot buffer. Returns immediately.
        Compression happens in finalize_index().
        
        Args:
            vecs: (N, dim) or (dim,) array of vectors
            meta_batch: List of N metadata dicts (or single dict, or None)
        """
        vecs = vecs.astype("float32")
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        n = vecs.shape[0]
        
        # Handle metadata
        if meta_batch is None:
            meta_batch = [{} for _ in range(n)]
        elif isinstance(meta_batch, dict):
            meta_batch = [meta_batch]
        elif not isinstance(meta_batch, list):
            meta_batch = list(meta_batch)
        
        # Ensure correct length
        if len(meta_batch) != n:
            if len(meta_batch) > n:
                meta_batch = meta_batch[:n]
            else:
                meta_batch = list(meta_batch) + [{} for _ in range(n - len(meta_batch))]
        
        # Ensure all are dicts with original_idx
        for i, meta in enumerate(meta_batch):
            if not isinstance(meta, dict):
                meta_batch[i] = {}
            if "original_idx" not in meta_batch[i]:
                meta_batch[i]["original_idx"] = len(self.child_store.child_meta) + i
        
        # Add to hot buffer (optimized: use concatenate instead of vstack for better performance)
        with self._hot_buffer_lock:
            # Use np.concatenate for better performance with large arrays
            if self.hot_buffer_vectors.shape[0] == 0:
                self.hot_buffer_vectors = vecs
            else:
                self.hot_buffer_vectors = np.concatenate([self.hot_buffer_vectors, vecs], axis=0)
            self.hot_buffer_meta.extend(meta_batch)
            
            # If hot buffer exceeds capacity, it will be compressed in finalize_index()
            hot_size = self.hot_buffer_vectors.shape[0]
            
            # Auto-flush warning: if hot buffer gets very large, log a warning
            if hot_size > self.config.hot_buffer_size * 100:
                import warnings
                warnings.warn(
                    f"Hot buffer size ({hot_size}) is {hot_size // self.config.hot_buffer_size}x larger than configured size ({self.config.hot_buffer_size}). "
                    f"Consider calling finalize_index() to compress vectors.",
                    UserWarning
                )
        
        self._log("add", added=n, hot_buffer_size=hot_size)
    
    def finalize_index(self, expected_count: Optional[int] = None, timeout_s: float = 120.0) -> None:
        """
        Flush hot buffer, build clusters, and finalize index.
        Deterministic, synchronous processing.
        
        Args:
            expected_count: Expected number of vectors (for validation)
            timeout_s: Hard timeout (raises if exceeded)
        """
        finalize_start = time.time()
        print(f"Finalizing index (timeout: {timeout_s}s)...")
        
        # Step 1: Flush hot buffer to child store
        with self._hot_buffer_lock:
            if self.hot_buffer_vectors.shape[0] > 0:
                vectors_to_add = self.hot_buffer_vectors.copy()
                metadata_to_add = [m.copy() for m in self.hot_buffer_meta]
                
                # Clear hot buffer
                self.hot_buffer_vectors = np.empty((0, self.dim), dtype="float32")
                self.hot_buffer_meta = []
            else:
                vectors_to_add = np.empty((0, self.dim), dtype="float32")
                metadata_to_add = []
        
        # Add to child store
        if vectors_to_add.shape[0] > 0:
            self.child_store.add(vectors_to_add, metadata_to_add)
        
        # Step 2: Build clusters from all child vectors
        print(f"Building clusters from {self.child_store.size()} vectors...")
        build_start = time.time()
        
        # Get all child vectors and metadata
        child_vectors = self.child_store.child_vectors.copy()
        child_meta = self.child_store.child_meta.copy()
        
        # Normalize child vectors
        from .cluster import _l2norm
        child_vectors = _l2norm(child_vectors)
        
        # Build clusters
        super_vectors, cluster_offsets, cluster_child_ids = build_clusters(
            child_vectors, child_meta, self.config
        )
        
        build_time = time.time() - build_start
        print(f"Clusters built: {len(super_vectors)} super vectors in {build_time:.2f}s")
        
        # Step 3: Update cluster index
        with self._cluster_lock:
            self.super_vectors = super_vectors
            self.cluster_offsets = cluster_offsets
            self.cluster_child_ids = cluster_child_ids
            self._index_finalized = True
        
        # Step 4: Finalize child store
        self.child_store.finalize()
        
        # Step 5: Validate
        finalize_elapsed = time.time() - finalize_start
        print(f"Index finalization completed in {finalize_elapsed:.2f}s (timeout: {timeout_s}s)")
        if finalize_elapsed > timeout_s:
            raise RuntimeError(
                f"finalize_index exceeded timeout {timeout_s}s (took {finalize_elapsed:.1f}s)"
            )
        
        if expected_count is not None:
            actual_count = self.child_store.size()
            if actual_count < expected_count * 0.99:
                raise RuntimeError(
                    f"Indexing incomplete: expected {expected_count}, got {actual_count}"
                )
        
        print(f"Index finalized: {self.child_store.size()} vectors, {len(super_vectors)} clusters")
        self._log("finalize_index", 
                 vectors=self.child_store.size(),
                 clusters=len(super_vectors),
                 time=finalize_elapsed)
    
    def search(self, qvec: np.ndarray, k: int = 5) -> Tuple[List[Dict], np.ndarray]:
        """
        Search for top-k similar vectors.
        
        Args:
            qvec: (dim,) query vector
            k: Number of results to return
            
        Returns:
            results: List of metadata dicts
            scores: (k,) array of similarity scores
        """
        qvec = qvec.astype("float32").flatten()
        
        # Normalize query
        from .cluster import _l2norm
        q_normalized = _l2norm(qvec.reshape(1, -1)).flatten()
        
        # Initialize candidate lists
        candidate_vectors_list = []
        candidate_metadata_list = []
        candidate_ids_list = []
        
        # Step 1: Search hot buffer
        with self._hot_buffer_lock:
            if self.hot_buffer_vectors.shape[0] > 0:
                hot_vecs = _l2norm(self.hot_buffer_vectors)
                hot_scores = (q_normalized @ hot_vecs.T).flatten()
                top_hot = np.argsort(-hot_scores)[:k * 2]
                
                for idx in top_hot:
                    candidate_vectors_list.append(self.hot_buffer_vectors[idx])
                    candidate_metadata_list.append(self.hot_buffer_meta[idx].copy())
                    candidate_ids_list.append(self.hot_buffer_meta[idx].get("original_idx", idx))
        
        # Step 2: Search cluster index
        with self._cluster_lock:
            if self._index_finalized and self.super_vectors.shape[0] > 0:
                # Route to top clusters
                beam_size = max(64, 20 * k, self.config.beam_size)
                top_super_ids = route(self.super_vectors, q_normalized, beam_size)
                
                # Expand to candidate children
                cand_child_ids = expand(
                    self.cluster_offsets,
                    self.cluster_child_ids,
                    top_super_ids
                )
                
                # Rerank candidates
                if len(cand_child_ids) > 0:
                    # Get child vectors for reranking
                    child_vectors = self.child_store.child_vectors
                    top_child_ids, top_scores = rerank(
                        child_vectors,
                        cand_child_ids,
                        q_normalized,
                        k * 2
                    )
                    
                    # Get metadata for top children
                    for child_id in top_child_ids:
                        if 0 <= child_id < len(self.child_store.child_meta):
                            candidate_vectors_list.append(child_vectors[child_id])
                            candidate_metadata_list.append(self.child_store.child_meta[child_id].copy())
                            candidate_ids_list.append(self.child_store.child_meta[child_id].get("original_idx", child_id))
        
        # Step 3: Final rerank and deduplicate
        if len(candidate_vectors_list) == 0:
            return [], np.array([], dtype="float32")
        
        # Combine all candidates
        candidates_matrix = np.vstack(candidate_vectors_list).astype("float32")
        candidates_matrix = _l2norm(candidates_matrix)
        
        # Exact rerank
        similarity_scores = (q_normalized @ candidates_matrix.T).flatten()
        
        # Remove duplicates by original_idx
        unique_indices = []
        seen_ids = set()
        for i, item_id in enumerate(candidate_ids_list):
            if item_id is not None and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_indices.append(i)
        
        if len(unique_indices) == 0:
            return [], np.array([], dtype="float32")
        
        unique_indices = np.array(unique_indices)
        unique_scores = similarity_scores[unique_indices]
        unique_metadata = [candidate_metadata_list[i] for i in unique_indices]
        
        # Sort and return top-k
        top_k = min(k, len(unique_scores))
        sorted_indices = np.argsort(-unique_scores)[:top_k]
        
        final_results = [unique_metadata[i] for i in sorted_indices]
        final_scores = unique_scores[sorted_indices]
        
        self._log("search", k=k, results=len(final_results))
        
        return final_results, final_scores
    
    def size(self) -> int:
        """Total size: hot buffer + child store."""
        with self._hot_buffer_lock:
            hot_size = self.hot_buffer_vectors.shape[0]
        return hot_size + self.child_store.size()
    
    def get_hot_buffer_size(self) -> int:
        """Number of items in hot buffer."""
        with self._hot_buffer_lock:
            return self.hot_buffer_vectors.shape[0]
    
    def get_cold_index_size(self) -> int:
        """Number of super vectors in cluster index."""
        with self._cluster_lock:
            return self.super_vectors.shape[0]
    
    def flush_optimizer(self, timeout: float = 5.0):
        """No-op in v1 (no background optimizer)."""
        pass
    
    def shutdown(self):
        """Shutdown (no-op in v1, but preserve API)."""
        pass
    
    def load(self) -> bool:
        """Load from storage backend (if available)."""
        # TODO: Implement loading
        return False
    
    def save(self, path: str, extra: dict | None = None):
        """Save state (simplified)."""
        import pickle
        state = {
            "hot_buffer_vectors": self.hot_buffer_vectors,
            "hot_buffer_meta": self.hot_buffer_meta,
            "dim": self.dim,
            "config": self.config,
            "events": self.events,
            "extra": extra or {}
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        self._log("save", path=path)
    
    @classmethod
    def load(cls, path: str):
        """Load state."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self = cls(
            dim=state["dim"],
            hot_buffer_size=state.get("hot_buffer_size", 50)
        )
        self.hot_buffer_vectors = state["hot_buffer_vectors"]
        self.hot_buffer_meta = state["hot_buffer_meta"]
        self.events = state.get("events", [])
        self._log("load", path=path)
        return self
    
    def _log(self, event: str, **kw):
        """Log event."""
        rec = {"t": time.time(), "event": event}
        rec.update(kw)
        self.events.append(rec)

