"""
Optional Qdrant baseline runner.

If Docker is available, runs Qdrant in a container.
Otherwise, provides instructions for manual setup.
"""
import os
import subprocess
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def start_qdrant_container() -> Optional[str]:
    """Start Qdrant container. Returns container ID or None."""
    if not check_docker():
        return None
    
    # Check if container already exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=mcn-qdrant", "--format", "{{.ID}}"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        container_id = result.stdout.strip().split('\n')[0]
        # Start if stopped
        subprocess.run(["docker", "start", container_id], capture_output=True)
        return container_id
    
    # Create new container
    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", "mcn-qdrant",
            "-p", "6333:6333",
            "-p", "6334:6334",
            "qdrant/qdrant:latest"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        container_id = result.stdout.strip()
        # Wait for Qdrant to be ready
        time.sleep(5)
        return container_id
    
    return None


def stop_qdrant_container():
    """Stop Qdrant container."""
    if not check_docker():
        return
    
    subprocess.run(
        ["docker", "stop", "mcn-qdrant"],
        capture_output=True
    )


def evaluate_qdrant(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truths: List[List[str]],
    metadata: List[Dict],
    collection_name: str = "mcn_test"
) -> Dict:
    """
    Evaluate Qdrant.
    
    Returns results dict or None if Qdrant not available.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        return {
            "available": False,
            "error": "qdrant-client not installed. Install with: pip install qdrant-client",
            "instructions": "See DETAILS.md for manual setup instructions"
        }
    
    if not check_docker():
        return {
            "available": False,
            "error": "Docker not available",
            "instructions": "See DETAILS.md for manual setup instructions"
        }
    
    container_id = start_qdrant_container()
    if not container_id:
        return {
            "available": False,
            "error": "Failed to start Qdrant container",
            "instructions": "See DETAILS.md for manual setup instructions"
        }
    
    try:
        client = QdrantClient(url="http://localhost:6333")
        
        # Create collection
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        dim = vectors.shape[1]
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        
        # Insert vectors
        print("  Inserting vectors into Qdrant...")
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload=metadata[i]
            )
            for i in range(len(vectors))
        ]
        
        build_start = time.time()
        client.upsert(collection_name=collection_name, points=points)
        build_time = time.time() - build_start
        
        # Search
        print("  Running queries...")
        latencies = []
        predictions = []
        
        for query in queries:
            start = time.time()
            results = client.search(
                collection_name=collection_name,
                query_vector=query.tolist(),
                limit=100
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            pred_ids = [str(r.id) for r in results]
            predictions.append(pred_ids)
        
        # Calculate metrics
        from utils_metrics import calculate_metrics, calculate_latency_stats
        
        metrics = calculate_metrics(predictions, ground_truths, None, k_values=[10, 100])
        latency_stats = calculate_latency_stats(latencies)
        
        return {
            "available": True,
            "metrics": metrics,
            "latency_stats": latency_stats,
            "build_time": build_time,
            "compression_ratio": 1.0,  # No compression
        }
    
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "instructions": "See DETAILS.md for manual setup instructions"
        }


def get_qdrant_instructions() -> str:
    """Get instructions for manual Qdrant setup."""
    return """
## Qdrant Baseline Setup

### Option 1: Docker (Recommended)

```bash
# Start Qdrant container
docker run -d --name mcn-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Install client
pip install qdrant-client

# Run evaluation (will automatically use Qdrant)
python benchmarks/run_all.py --qdrant
```

### Option 2: Local Installation

1. Download Qdrant from https://qdrant.tech/documentation/guides/installation/
2. Start Qdrant server: `qdrant`
3. Install client: `pip install qdrant-client`
4. Run evaluation: `python benchmarks/run_all.py --qdrant`

### Option 3: Cloud Qdrant

1. Sign up at https://cloud.qdrant.io
2. Create a cluster
3. Set environment variable: `export QDRANT_URL=your-cluster-url`
4. Set API key: `export QDRANT_API_KEY=your-api-key`
5. Run evaluation: `python benchmarks/run_all.py --qdrant`
"""

