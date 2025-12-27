# src/mcn/store.py
"""
ChildStore: Stores all original vectors in compact form for reranking.
"""
import numpy as np
from typing import List, Dict, Optional
import threading


class ChildStore:
    """
    Stores child vectors and metadata.
    Single source of truth for all original vectors.
    """
    
    def __init__(self, dim: int, dtype: str = "float32"):
        self.dim = dim
        self.dtype = np.dtype(dtype)
        self._lock = threading.Lock()
        
        # Storage: contiguous arrays for fast access
        self.child_vectors: np.ndarray = np.empty((0, dim), dtype=self.dtype)
        self.child_meta: List[Dict] = []
        
        # Mapping: original_idx -> position in arrays
        self.orig_idx_to_pos: Dict[int, int] = {}
        self._next_original_idx = 0
    
    def add(self, vectors: np.ndarray, meta_batch: Optional[List[Dict]] = None) -> List[int]:
        """
        Add vectors to store.
        
        Args:
            vectors: (N, dim) array of vectors
            meta_batch: List of N metadata dicts (must contain original_idx)
            
        Returns:
            List of positions where vectors were stored
        """
        vectors = vectors.astype(self.dtype)
        n = vectors.shape[0]
        
        if meta_batch is None:
            meta_batch = [{} for _ in range(n)]
        
        # Ensure all metadata have original_idx
        for i, meta in enumerate(meta_batch):
            if not isinstance(meta, dict):
                meta_batch[i] = {}
            if "original_idx" not in meta_batch[i]:
                meta_batch[i]["original_idx"] = self._next_original_idx
                self._next_original_idx += 1
        
        with self._lock:
            start_pos = len(self.child_vectors)
            positions = list(range(start_pos, start_pos + n))
            
            # Append vectors (ensure contiguous)
            self.child_vectors = np.vstack([self.child_vectors, vectors])
            self.child_meta.extend(meta_batch)
            
            # Update mapping
            for pos, meta in zip(positions, meta_batch):
                orig_idx = int(meta["original_idx"])
                self.orig_idx_to_pos[orig_idx] = pos
        
        return positions
    
    def get_vectors(self, ids: List[int]) -> np.ndarray:
        """
        Get vectors by positions or original_idx.
        
        Args:
            ids: List of positions or original_idx values
            
        Returns:
            (len(ids), dim) array of vectors
        """
        with self._lock:
            # Convert original_idx to positions if needed
            positions = []
            for id_val in ids:
                if id_val in self.orig_idx_to_pos:
                    positions.append(self.orig_idx_to_pos[id_val])
                elif 0 <= id_val < len(self.child_vectors):
                    positions.append(id_val)
                else:
                    continue
            
            if len(positions) == 0:
                return np.empty((0, self.dim), dtype=self.dtype)
            
            return self.child_vectors[positions].copy()
    
    def get_meta(self, ids: List[int]) -> List[Dict]:
        """
        Get metadata by positions or original_idx.
        
        Args:
            ids: List of positions or original_idx values
            
        Returns:
            List of metadata dicts
        """
        with self._lock:
            # Convert original_idx to positions if needed
            positions = []
            for id_val in ids:
                if id_val in self.orig_idx_to_pos:
                    positions.append(self.orig_idx_to_pos[id_val])
                elif 0 <= id_val < len(self.child_meta):
                    positions.append(id_val)
                else:
                    continue
            
            return [self.child_meta[pos].copy() for pos in positions if pos < len(self.child_meta)]
    
    def finalize(self):
        """
        Finalize store (ensure contiguous, optimize).
        Called after all vectors are added.
        """
        with self._lock:
            # Ensure contiguous
            if not self.child_vectors.flags["C_CONTIGUOUS"]:
                self.child_vectors = np.ascontiguousarray(self.child_vectors)
    
    def size(self) -> int:
        """Get total number of vectors."""
        with self._lock:
            return len(self.child_vectors)
    
    def clear(self):
        """Clear all data."""
        with self._lock:
            self.child_vectors = np.empty((0, self.dim), dtype=self.dtype)
            self.child_meta = []
            self.orig_idx_to_pos = {}
            self._next_original_idx = 0

