"""
MCN v1 Production Server (FastAPI)
Clean architecture with rate limiting and tenant isolation.
"""
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from threading import Lock
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from typing import List, Any
import numpy as np
import psutil
import asyncio
from contextlib import asynccontextmanager

from mcn import MCNLayer

# Configuration
DIM = 384  # all-MiniLM-L6-v2 dimension
HOT_BUFFER_SIZE = 50
RAM_SAFETY_THRESHOLD = 85.0

# Rate Limiting Configuration
RATE_LIMIT_CONFIG = {
    "default": {
        "max_vectors": 100_000,  # Max vectors per tenant
        "ingest_rate": 1000,  # Vectors per minute
        "qps": 100,  # Queries per second
        "concurrent_searches": 10,  # Max concurrent searches
        "daily_queries": 100_000,  # Max queries per day
    }
}

# Global MCN instance
mcn: Optional[MCNLayer] = None

# Tenant tracking
tenant_stats: Dict[str, Dict] = defaultdict(lambda: {
    "vectors_stored": 0,
    "ingest_count": 0,
    "ingest_window_start": time.time(),
    "query_count": 0,
    "query_window_start": time.time(),
    "daily_queries": 0,
    "daily_window_start": time.time(),
    "concurrent_searches": 0,
})
tenant_lock = Lock()


# Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter for per-tenant limits."""
    
    async def dispatch(self, request, call_next):
        # Extract API key (tenant ID)
        api_key = None
        if request.method == "POST":
            try:
                import json
                body = await request.body()
                data = json.loads(body)
                api_key = data.get("api_key", "default")
            except:
                api_key = "default"
        
        if api_key:
            # Check rate limits
            with tenant_lock:
                stats = tenant_stats[api_key]
                config = RATE_LIMIT_CONFIG.get("default", RATE_LIMIT_CONFIG["default"])
                
                # Check ingest rate
                if "/add" in str(request.url):
                    now = time.time()
                    if now - stats["ingest_window_start"] > 60:
                        stats["ingest_count"] = 0
                        stats["ingest_window_start"] = now
                    
                    if stats["ingest_count"] >= config["ingest_rate"]:
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Rate limit exceeded",
                                "message": f"Max {config['ingest_rate']} vectors per minute",
                                "retry_after": 60
                            }
                        )
                
                # Check QPS
                if "/search" in str(request.url):
                    now = time.time()
                    if now - stats["query_window_start"] > 1:
                        stats["query_count"] = 0
                        stats["query_window_start"] = now
                    
                    if stats["query_count"] >= config["qps"]:
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Rate limit exceeded",
                                "message": f"Max {config['qps']} queries per second",
                                "retry_after": 1
                            }
                        )
                    
                    # Check daily limit
                    if now - stats["daily_window_start"] > 86400:
                        stats["daily_queries"] = 0
                        stats["daily_window_start"] = now
                    
                    if stats["daily_queries"] >= config["daily_queries"]:
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Daily limit exceeded",
                                "message": f"Max {config['daily_queries']} queries per day",
                                "retry_after": 86400
                            }
                        )
                    
                    # Check concurrent searches
                    if stats["concurrent_searches"] >= config["concurrent_searches"]:
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Concurrent limit exceeded",
                                "message": f"Max {config['concurrent_searches']} concurrent searches",
                                "retry_after": 1
                            }
                        )
        
        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    global mcn
    
    # Startup: Initialize MCN v1
    print("Initializing MCN v1...")
    mcn = MCNLayer(
        dim=DIM,
        hot_buffer_size=HOT_BUFFER_SIZE,
        use_background_thread=False  # v1: no background threads
    )
    
    print(f"MCN v1 initialized: dim={DIM}, hot_buffer_size={HOT_BUFFER_SIZE}")
    
    yield
    
    # Shutdown: Persist data
    print("Shutting down MCN...")
    if mcn is not None:
        try:
            mcn.save("mcn_state.pkl")
        except Exception as e:
            print(f"Warning: Save during shutdown failed: {e}")
    print("MCN shutdown complete")


app = FastAPI(
    title="MCN v1 Vector Database API",
    description="Production server for Memory Compression Networks v1",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)


# Request/Response Models
class VectorItem(BaseModel):
    id: str
    values: List[float] = Field(..., min_length=1)
    metadata: Dict[str, Any] = {}


class AddRequest(BaseModel):
    api_key: str
    vectors: List[VectorItem]


class SearchRequest(BaseModel):
    api_key: str
    vector: List[float] = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=100)


class HealthResponse(BaseModel):
    status: str
    ram_usage: str
    hot_buffer_size: Optional[int] = None
    cold_index_size: Optional[int] = None
    total_vectors: Optional[int] = None


# Helper Functions
def get_ram_usage_percent() -> float:
    """Get current RAM usage percentage."""
    return psutil.virtual_memory().percent


def inject_user_id(metadata: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Inject user_id into metadata for security filtering."""
    metadata = dict(metadata)
    metadata['user_id'] = user_id
    return metadata


def filter_by_user_id(results: List[Dict], scores: np.ndarray, user_id: str) -> tuple:
    """Security filter: Only return items belonging to the user."""
    filtered_results = []
    filtered_scores = []
    
    for result, score in zip(results, scores):
        result_user_id = None
        if isinstance(result, dict):
            result_user_id = result.get('metadata', {}).get('user_id')
            if not result_user_id:
                result_user_id = result.get('user_id')
        
        if result_user_id == user_id:
            filtered_results.append(result)
            filtered_scores.append(float(score))
    
    return filtered_results, np.array(filtered_scores, dtype="float32")


# API Endpoints
@app.post("/add")
async def add_vectors(request: AddRequest):
    """
    Ingest vectors with rate limiting and tenant limits.
    """
    global mcn
    
    if mcn is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    try:
        # Check RAM usage
        ram_percent = get_ram_usage_percent()
        if ram_percent > RAM_SAFETY_THRESHOLD:
            raise HTTPException(
                status_code=503,
                detail=f"RAM usage {ram_percent:.1f}% exceeds threshold {RAM_SAFETY_THRESHOLD}%"
            )
        
        # Validate vectors
        if not request.vectors:
            raise HTTPException(status_code=400, detail="Empty vectors list")
        
        # Extract user_id from api_key
        user_id = request.api_key
        
        # Check tenant limits
        with tenant_lock:
            stats = tenant_stats[user_id]
            config = RATE_LIMIT_CONFIG.get("default", RATE_LIMIT_CONFIG["default"])
            
            if stats["vectors_stored"] + len(request.vectors) > config["max_vectors"]:
                raise HTTPException(
                    status_code=403,
                    detail=f"Max {config['max_vectors']} vectors per tenant exceeded"
                )
        
        # Convert to numpy arrays
        vectors_list = []
        metadata_list = []
        
        for item in request.vectors:
            if len(item.values) != DIM:
                raise HTTPException(
                    status_code=400,
                    detail=f"Vector dimension mismatch: expected {DIM}, got {len(item.values)}"
                )
            
            vectors_list.append(item.values)
            metadata = inject_user_id(item.metadata, user_id)
            metadata['id'] = item.id
            metadata['original_idx'] = len(metadata_list)  # Assign sequential original_idx
            metadata_list.append(metadata)
        
        vectors_array = np.array(vectors_list, dtype="float32")
        
        # Add to MCN (run in thread pool to avoid blocking)
        def add_vectors():
            mcn.add(vectors_array, metadata_list)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, add_vectors)
        
        # Update tenant stats
        with tenant_lock:
            tenant_stats[user_id]["vectors_stored"] += len(request.vectors)
            tenant_stats[user_id]["ingest_count"] += len(request.vectors)
        
        return {
            "status": "success",
            "added": len(request.vectors),
            "ram_usage_percent": get_ram_usage_percent(),
            "total_vectors": tenant_stats[user_id]["vectors_stored"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /add: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/search")
async def search_vectors(request: SearchRequest):
    """
    Search vectors with rate limiting and security filtering.
    """
    global mcn
    
    if mcn is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    try:
        # Validate vector dimension
        if len(request.vector) != DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension mismatch: expected {DIM}, got {len(request.vector)}"
            )
        
        # Extract user_id from api_key
        user_id = request.api_key
        
        # Update tenant stats (concurrent search tracking)
        with tenant_lock:
            tenant_stats[user_id]["concurrent_searches"] += 1
            tenant_stats[user_id]["query_count"] += 1
            tenant_stats[user_id]["daily_queries"] += 1
        
        try:
            # Convert to numpy array
            query_vector = np.array(request.vector, dtype="float32")
            
            # Search (run in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            results, scores = await loop.run_in_executor(
                None, mcn.search, query_vector, request.k
            )
            
            # Security Filter: Only return items belonging to this user
            filtered_results, filtered_scores = filter_by_user_id(results, scores, user_id)
            
            # Format response
            response_items = []
            for result, score in zip(filtered_results, filtered_scores):
                item = {
                    "id": result.get("id", ""),
                    "metadata": result.get("metadata", result),
                    "score": float(score)
                }
                response_items.append(item)
            
            return {
                "status": "success",
                "results": response_items,
                "count": len(response_items)
            }
        finally:
            # Decrement concurrent searches
            with tenant_lock:
                tenant_stats[user_id]["concurrent_searches"] = max(0, tenant_stats[user_id]["concurrent_searches"] - 1)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/finalize")
async def finalize_index():
    """
    Finalize index (build clusters).
    Should be called after all vectors are ingested.
    """
    global mcn
    
    if mcn is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    try:
        def finalize():
            mcn.finalize_index()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, finalize)
        
        return {
            "status": "success",
            "total_vectors": mcn.size(),
            "clusters": mcn.get_cold_index_size()
        }
    except Exception as e:
        print(f"Error in /finalize: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global mcn
    
    ram_percent = get_ram_usage_percent()
    ram_usage_str = f"{ram_percent:.1f}%"
    
    if mcn is None:
        return HealthResponse(
            status="initializing",
            ram_usage=ram_usage_str
        )
    
    try:
        hot_size = mcn.get_hot_buffer_size()
        cold_size = mcn.get_cold_index_size()
        total = mcn.size()
        
        return HealthResponse(
            status="ok",
            ram_usage=ram_usage_str,
            hot_buffer_size=hot_size,
            cold_index_size=cold_size,
            total_vectors=total
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {str(e)}",
            ram_usage=ram_usage_str
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "MCN v1 Vector Database API",
        "version": "1.0.0",
        "endpoints": {
            "POST /add": "Ingest vectors",
            "POST /search": "Search vectors",
            "POST /finalize": "Finalize index (build clusters)",
            "GET /health": "Health check",
            "GET /": "This endpoint"
        },
        "configuration": {
            "dimension": DIM,
            "hot_buffer_size": HOT_BUFFER_SIZE
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
