"""
Snapshot Manager: Implements snapshot swap for non-blocking searches during finalize.

Architecture:
- active_snapshot: Read-only MCNLayer for searches (never mutated)
- write_buffer: MCNLayer for ingestion (hot log)
- new_snapshot: MCNLayer being built (background)
- Atomic swap: active_snapshot -> new_snapshot under short lock
"""
import numpy as np
import threading
import time
import uuid
import os
import pickle
from typing import Optional, Tuple, List, Dict, Any
from .mcn_layer import MCNLayer


class SnapshotManager:
    """
    Manages snapshots for non-blocking searches during finalize.
    
    Thread-safe: searches use active_snapshot (read-only), writes go to write_buffer.
    """
    
    def __init__(self, dim: int, hot_buffer_size: int = 50, storage_path: Optional[str] = None, **kwargs):
        self.dim = dim
        self.hot_buffer_size = hot_buffer_size
        self.kwargs = kwargs
        self.storage_path = storage_path  # Path to persistent storage (Railway volume)
        
        # Active snapshot: read-only for searches
        self.active_snapshot: Optional[MCNLayer] = MCNLayer(
            dim=dim,
            hot_buffer_size=hot_buffer_size,
            **kwargs
        )
        
        # Write buffer: receives all new vectors
        self.write_buffer: MCNLayer = MCNLayer(
            dim=dim,
            hot_buffer_size=hot_buffer_size,
            **kwargs
        )
        
        # New snapshot being built (None if not building)
        self.new_snapshot: Optional[MCNLayer] = None
        
        # Build tracking
        self.active_builds: Dict[str, Dict] = {}  # build_id -> {status, start_time, snapshot}
        self._build_lock = threading.Lock()
        self._swap_lock = threading.Lock()  # Short lock for atomic swap
        
        # Stats
        self.swap_count = 0
        self.last_swap_time: Optional[float] = None
    
    def add(self, vecs: np.ndarray, meta_batch: Optional[List[Dict]] = None):
        """
        Add vectors to write buffer (hot log).
        Non-blocking, always succeeds immediately.
        """
        self.write_buffer.add(vecs, meta_batch)
    
    def search(self, qvec: np.ndarray, k: int = 5) -> Tuple[List[Dict], np.ndarray]:
        """
        Search using active snapshot (read-only, never blocks).
        """
        with self._swap_lock:  # Short read lock to get snapshot reference
            snapshot = self.active_snapshot
        
        if snapshot is None:
            # Fallback: search write buffer if no snapshot yet
            return self.write_buffer.search(qvec, k)
        
        return snapshot.search(qvec, k)
    
    def start_build(self, timeout_s: float = 120.0) -> str:
        """
        Start building a new snapshot from write_buffer.
        Returns build_id for status tracking.
        
        This is non-blocking - returns immediately with build_id.
        Actual build happens synchronously (caller should run in executor).
        """
        build_id = str(uuid.uuid4())
        
        with self._build_lock:
            if self.new_snapshot is not None:
                raise RuntimeError("Build already in progress")
            
            # Create new snapshot from write_buffer state
            self.new_snapshot = MCNLayer(
                dim=self.dim,
                hot_buffer_size=self.hot_buffer_size,
                **self.kwargs
            )
            
            # Copy all vectors from write_buffer to new_snapshot
            # (This is the "snapshot" - we'll finalize it separately)
            self._copy_write_buffer_to_snapshot(self.new_snapshot)
            
            self.active_builds[build_id] = {
                "status": "building",
                "start_time": time.time(),
                "snapshot": self.new_snapshot,
                "timeout_s": timeout_s
            }
        
        return build_id
    
    def _copy_write_buffer_to_snapshot(self, snapshot: MCNLayer):
        """
        Copy all vectors to snapshot: active_snapshot (if finalized) + write_buffer.
        This creates a complete snapshot without mutating source.
        """
        # Step 1: Copy from active_snapshot's child_store (if it has finalized vectors)
        if self.active_snapshot and self.active_snapshot._index_finalized:
            with self.active_snapshot.child_store._lock:
                if self.active_snapshot.child_store.size() > 0:
                    active_vecs = self.active_snapshot.child_store.child_vectors.copy()
                    active_meta = [m.copy() for m in self.active_snapshot.child_store.child_meta]
                    snapshot.child_store.add(active_vecs, active_meta)
        
        # Step 2: Copy from write_buffer's hot buffer
        with self.write_buffer._hot_buffer_lock:
            if self.write_buffer.hot_buffer_vectors.shape[0] > 0:
                vectors_to_copy = self.write_buffer.hot_buffer_vectors.copy()
                metadata_to_copy = [m.copy() for m in self.write_buffer.hot_buffer_meta]
                snapshot.child_store.add(vectors_to_copy, metadata_to_copy)
        
        # Step 3: Copy from write_buffer's child_store (if any exist)
        with self.write_buffer.child_store._lock:
            if self.write_buffer.child_store.size() > 0:
                child_vecs = self.write_buffer.child_store.child_vectors.copy()
                child_meta = [m.copy() for m in self.write_buffer.child_store.child_meta]
                snapshot.child_store.add(child_vecs, child_meta)
    
    def finalize_build(self, build_id: str) -> Dict[str, Any]:
        """
        Finalize the build (run finalize_index on new_snapshot).
        This should be called in an executor thread.
        
        Returns build status dict.
        """
        with self._build_lock:
            if build_id not in self.active_builds:
                raise ValueError(f"Build {build_id} not found")
            
            build_info = self.active_builds[build_id]
            snapshot = build_info["snapshot"]
            timeout_s = build_info["timeout_s"]
        
        # Finalize the snapshot (this takes time, but doesn't block searches)
        try:
            finalize_start = time.time()
            snapshot.finalize_index(timeout_s=timeout_s)
            finalize_time = time.time() - finalize_start
            
            # Update build status
            with self._build_lock:
                build_info["status"] = "ready"
                build_info["finalize_time"] = finalize_time
                build_info["vectors"] = snapshot.size()
                build_info["clusters"] = snapshot.get_cold_index_size()
            
            return {
                "build_id": build_id,
                "status": "ready",
                "vectors": snapshot.size(),
                "clusters": snapshot.get_cold_index_size(),
                "finalize_time": finalize_time
            }
        except Exception as e:
            with self._build_lock:
                build_info["status"] = "error"
                build_info["error"] = str(e)
            raise
    
    def swap_snapshot(self, build_id: str) -> Dict[str, Any]:
        """
        Atomically swap active_snapshot -> new_snapshot.
        This is a short critical section (just pointer swap).
        
        Returns swap result dict.
        """
        with self._build_lock:
            if build_id not in self.active_builds:
                raise ValueError(f"Build {build_id} not found")
            
            build_info = self.active_builds[build_id]
            if build_info["status"] != "ready":
                raise RuntimeError(f"Build {build_id} not ready (status: {build_info['status']})")
            
            new_snapshot = build_info["snapshot"]
        
        # Atomic swap under short lock
        with self._swap_lock:
            old_snapshot = self.active_snapshot
            self.active_snapshot = new_snapshot
            self.swap_count += 1
            self.last_swap_time = time.time()
        
        # Clear write_buffer (vectors are now in active_snapshot)
        # But keep it for new vectors
        self.write_buffer = MCNLayer(
            dim=self.dim,
            hot_buffer_size=self.hot_buffer_size,
            **self.kwargs
        )
        
        # Clean up build tracking
        with self._build_lock:
            self.new_snapshot = None
            # Keep build_info for history, but mark as swapped
            build_info["status"] = "swapped"
            build_info["swap_time"] = time.time()
        
        # Save new snapshot to persistent storage (if enabled)
        if self.storage_path:
            try:
                self.save_snapshot(new_snapshot, snapshot_id="latest")
            except Exception as e:
                print(f"Warning: Failed to save snapshot to disk: {e}")
        
        return {
            "build_id": build_id,
            "status": "swapped",
            "old_vectors": old_snapshot.size() if old_snapshot else 0,
            "new_vectors": new_snapshot.size(),
            "swap_count": self.swap_count
        }
    
    def get_build_status(self, build_id: str) -> Dict[str, Any]:
        """Get build status by build_id."""
        with self._build_lock:
            if build_id not in self.active_builds:
                return {"status": "not_found", "build_id": build_id}
            
            build_info = self.active_builds[build_id].copy()
            build_info["build_id"] = build_id
            
            # Calculate elapsed time
            if "start_time" in build_info:
                elapsed = time.time() - build_info["start_time"]
                build_info["elapsed_time"] = elapsed
            
            return build_info
    
    def size(self) -> int:
        """Get total size (active snapshot + write buffer)."""
        active_size = self.active_snapshot.size() if self.active_snapshot else 0
        write_size = self.write_buffer.size()
        return active_size + write_size
    
    def get_hot_buffer_size(self) -> int:
        """Get hot buffer size from write_buffer."""
        return self.write_buffer.get_hot_buffer_size()
    
    def get_cold_index_size(self) -> int:
        """Get cold index size from active snapshot."""
        if self.active_snapshot:
            return self.active_snapshot.get_cold_index_size()
        return 0
    
    def get_active_builds_count(self) -> int:
        """Get count of active builds (status == 'building')."""
        with self._build_lock:
            return len([b for b in self.active_builds.values() if b.get("status") == "building"])

