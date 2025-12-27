# src/mcn/config.py
"""
MCN v1 Configuration
Clean, production-ready configuration for Memory Compression Networks.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class MCNConfig:
    """Configuration for MCN v1."""
    
    # Vector dimensions
    dim: int
    
    # Child storage
    dtype_child: Literal["float16", "float32"] = "float32"  # Child vector dtype
    
    # Clustering
    target_cluster_size: int = 15  # Target average cluster size
    max_cluster_size: int = 64  # Hard cap: split clusters exceeding this
    
    # Search
    beam_size: int = 200  # Number of top clusters to consider (default: max(64, 20*k))
    
    # Compression quality
    use_refine: bool = False  # Use GA refinement (slow, only if needed)
    refine_threshold: float = 0.80  # Only refine if min_similarity < this
    max_refine_gen: int = 15  # Max GA generations (hard cap)
    population_size: int = 12  # GA population size
    
    # Benchmark mode
    benchmark_mode: bool = False  # Enable strict validation and timeouts
    
    # Hot buffer
    hot_buffer_size: int = 50  # Max vectors in hot buffer before compression
    
    # Compression mode
    compression_mode: Literal["fast", "balanced", "max_quality"] = "balanced"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.dim > 0, "dim must be positive"
        assert self.target_cluster_size > 0, "target_cluster_size must be positive"
        assert self.max_cluster_size >= self.target_cluster_size, "max_cluster_size must be >= target_cluster_size"
        assert self.beam_size > 0, "beam_size must be positive"
        assert 0.0 <= self.refine_threshold <= 1.0, "refine_threshold must be in [0, 1]"
        assert self.max_refine_gen > 0, "max_refine_gen must be positive"
        assert self.population_size > 0, "population_size must be positive"
        assert self.hot_buffer_size > 0, "hot_buffer_size must be positive"
        
        # Adjust compression mode settings
        if self.compression_mode == "fast":
            self.use_refine = False
            self.max_refine_gen = 5
        elif self.compression_mode == "max_quality":
            self.use_refine = True
            self.max_refine_gen = 20
