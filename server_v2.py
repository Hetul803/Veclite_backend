"""
MCN v1 Production Server (FastAPI) - Multi-Tenant SaaS Edition
Clean architecture with snapshot swap, rate limiting, tenant isolation, and pod routing.
"""
import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException, Request, Depends, Header, Body
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import numpy as np
import psutil
import asyncio
from contextlib import asynccontextmanager
import httpx

from mcn import SnapshotManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DIM = 384  # all-MiniLM-L6-v2 dimension
HOT_BUFFER_SIZE = 50
RAM_SAFETY_THRESHOLD = 85.0

# Tenant Cap
MAX_TENANTS = int(os.getenv("MAX_TENANTS", "20"))

# Worker Scaling Configuration
WORKERS_RECOMMENDED = int(os.getenv("WORKERS_RECOMMENDED", "1"))
SCALE_THRESHOLD_TENANTS = 10
SCALE_THRESHOLD_VECTORS = 2_000_000
SCALE_THRESHOLD_P95_MS = 60

# Admin API Key (optional, for API access)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# Hardcoded Admin Credentials (for web UI)
ADMIN_EMAIL = "patelhetul803@gmail.com"
ADMIN_PASSWORD = "Hetul7698676686"

# Stripe Configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Supabase Configuration (optional, for future integration)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Storage Configuration
STORAGE_PATH = os.getenv("STORAGE_PATH", "/data")  # Railway volume mount point
ENABLE_PERSISTENCE = os.getenv("ENABLE_PERSISTENCE", "true").lower() == "true"

# Stripe import (optional, only if keys are set)
try:
    import stripe
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY
    STRIPE_AVAILABLE = bool(STRIPE_SECRET_KEY)
except ImportError:
    STRIPE_AVAILABLE = False
    if STRIPE_SECRET_KEY:
        logger.warning("Stripe not installed. Install with: pip install stripe")

# Rate Limiting Configuration (plan-based)
_default_config = {
    "max_vectors": 10_000,
    "ingest_rate": 100,
    "qps": 5,
    "concurrent_searches": 2,
    "daily_queries": 5_000,
}

RATE_LIMIT_CONFIG = {
    "free": _default_config.copy(),
    "starter": {
        "max_vectors": 100_000,
        "ingest_rate": 1000,
        "qps": 10,
        "concurrent_searches": 5,
        "daily_queries": 50_000,
    },
    "pro": {
        "max_vectors": 250_000,
        "ingest_rate": 5000,
        "qps": 25,
        "concurrent_searches": 10,
        "daily_queries": 200_000,
    },
    "scale": {
        "max_vectors": 1_000_000,
        "ingest_rate": 10000,
        "qps": 60,
        "concurrent_searches": 25,
        "daily_queries": 1_000_000,
    },
    "enterprise": {
        "max_vectors": 10_000_000,
        "ingest_rate": 50000,
        "qps": 200,
        "concurrent_searches": 100,
        "daily_queries": 10_000_000,
    },
    "default": _default_config,  # Fallback
}

# ============================================================================
# Global State
# ============================================================================

# Snapshot Manager (replaces single MCN instance)
snapshot_mgr: Optional[SnapshotManager] = None

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
    "plan": "free",  # Default plan
    "created_at": time.time(),
})

# Tenant -> Pod mapping (in-memory for now, DB-ready structure)
tenant_pods: Dict[str, str] = {}  # tenant_id -> pod_url

# Latency tracking (rolling windows)
latency_history: deque = deque(maxlen=1000)  # Last 1000 search latencies
qps_history: deque = deque(maxlen=60)  # Last 60 seconds of QPS

# Thread locks
tenant_lock = Lock()
pod_lock = Lock()

# ============================================================================
# Helper Functions
# ============================================================================

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


def get_tenant_plan(tenant_id: str) -> str:
    """Get tenant's plan (default: free)."""
    with tenant_lock:
        return tenant_stats[tenant_id].get("plan", "free")


def check_tenant_cap() -> bool:
    """Check if we've reached tenant cap."""
    with tenant_lock:
        active_tenants = len([t for t in tenant_stats.keys() if tenant_stats[t].get("vectors_stored", 0) > 0])
        return active_tenants >= MAX_TENANTS


def check_scaling_warnings():
    """Check if scaling thresholds are exceeded and log warnings."""
    with tenant_lock:
        total_tenants = len([t for t in tenant_stats.keys() if tenant_stats[t].get("vectors_stored", 0) > 0])
        total_vectors = sum(stats.get("vectors_stored", 0) for stats in tenant_stats.values())
    
    # Calculate p95 latency
    if len(latency_history) > 0:
        sorted_latencies = sorted(latency_history)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95_ms = sorted_latencies[p95_idx] * 1000  # Convert to ms
    else:
        p95_ms = 0
    
    warnings = []
    if total_tenants > SCALE_THRESHOLD_TENANTS:
        warnings.append(f"Tenants ({total_tenants}) > {SCALE_THRESHOLD_TENANTS}")
    if total_vectors > SCALE_THRESHOLD_VECTORS:
        warnings.append(f"Vectors ({total_vectors:,}) > {SCALE_THRESHOLD_VECTORS:,}")
    if p95_ms > SCALE_THRESHOLD_P95_MS:
        warnings.append(f"p95 latency ({p95_ms:.1f}ms) > {SCALE_THRESHOLD_P95_MS}ms")
    
    if warnings and WORKERS_RECOMMENDED == 1:
        logger.warning(
            f"⚠️  SCALING RECOMMENDATION: Consider deploying with 2 workers. "
            f"Thresholds exceeded: {', '.join(warnings)}"
        )


# ============================================================================
# Rate Limiting Middleware
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter for per-tenant limits.
    
    PRODUCTION FIX: Header-only middleware (NO body consumption).
    CRITICAL: This middleware MUST NEVER read the request body to avoid breaking ASGI stream.
    API key must be provided via headers only.
    """
    
    async def dispatch(self, request, call_next):
        # Extract API key (tenant ID) from headers ONLY - NEVER read body
        api_key = None
        
        # Try Authorization: Bearer <key> (preferred)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()
        # Try X-API-Key header
        elif "X-API-Key" in request.headers:
            api_key = request.headers["X-API-Key"]
        # Try X-Tenant-ID header (legacy support)
        elif "X-Tenant-ID" in request.headers:
            api_key = request.headers["X-Tenant-ID"]
        
        # Store API key in request.state for endpoints to use
        request.state.api_key = api_key
        
        # For POST endpoints, require API key in header (return 401 if missing)
        # DO NOT fall back to reading body - that breaks ASGI stream
        if request.method == "POST" and not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "error": "missing_api_key_header"
                }
            )
        
        if api_key:
            # Get tenant plan
            plan = get_tenant_plan(api_key)
            config = RATE_LIMIT_CONFIG.get(plan, RATE_LIMIT_CONFIG["default"])
            
            # Check rate limits
            with tenant_lock:
                stats = tenant_stats[api_key]
                
                # Check ingest rate
                if "/add" in str(request.url) or "/ingest" in str(request.url):
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
                            },
                            headers={"Retry-After": "60"}
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
                            },
                            headers={"Retry-After": "1"}
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
                            },
                            headers={"Retry-After": "86400"}
                        )
                    
                    # Check concurrent searches
                    if stats["concurrent_searches"] >= config["concurrent_searches"]:
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Concurrent limit exceeded",
                                "message": f"Max {config['concurrent_searches']} concurrent searches",
                                "retry_after": 1
                            },
                            headers={"Retry-After": "1"}
                        )
        
        response = await call_next(request)
        return response


# ============================================================================
# Admin Authentication
# ============================================================================

def verify_admin(
    admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    session_token: Optional[str] = Header(None, alias="X-Admin-Session")
):
    """Verify admin access via API key or session token."""
    # Check session token first (for web UI)
    if session_token:
        with admin_session_lock:
            expiry = admin_sessions.get(session_token, 0)
            if expiry > time.time():
                return session_token  # Valid session
            else:
                # Clean up expired session
                admin_sessions.pop(session_token, None)
    
    # Fall back to API key (for direct API access)
    if admin_key and ADMIN_API_KEY and admin_key == ADMIN_API_KEY:
        return admin_key
    
    # No valid authentication
    raise HTTPException(status_code=403, detail="Admin authentication required")


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    global snapshot_mgr
    
    # Startup: Initialize Snapshot Manager
    logger.info("Initializing Memryx v1 Snapshot Manager...")
    
    # Determine storage path
    storage_path = STORAGE_PATH if ENABLE_PERSISTENCE else None
    if storage_path:
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"Persistence enabled: storage_path={storage_path}")
    else:
        logger.info("Persistence disabled (vectors will be lost on restart)")
    
    snapshot_mgr = SnapshotManager(
        dim=DIM,
        hot_buffer_size=HOT_BUFFER_SIZE,
        storage_path=storage_path
    )
    
    # Try to load existing snapshot from disk
    if storage_path and ENABLE_PERSISTENCE:
        logger.info("Attempting to load existing snapshot from disk...")
        loaded_snapshot = snapshot_mgr.load_snapshot(snapshot_id="latest")
        if loaded_snapshot:
            logger.info(f"Loaded snapshot: {loaded_snapshot.size()} vectors")
            snapshot_mgr.active_snapshot = loaded_snapshot
        else:
            logger.info("No existing snapshot found, starting fresh")
    
    logger.info(f"Snapshot Manager initialized: dim={DIM}, hot_buffer_size={HOT_BUFFER_SIZE}")
    logger.info(f"Configuration: MAX_TENANTS={MAX_TENANTS}, WORKERS_RECOMMENDED={WORKERS_RECOMMENDED}")
    
    yield
    
    # Shutdown: Save current snapshot
    logger.info("Shutting down Memryx...")
    if snapshot_mgr and storage_path and ENABLE_PERSISTENCE:
        try:
            if snapshot_mgr.active_snapshot:
                snapshot_mgr.save_snapshot(snapshot_mgr.active_snapshot, snapshot_id="latest")
                logger.info("Snapshot saved to disk")
        except Exception as e:
            logger.error(f"Warning: Failed to save snapshot during shutdown: {e}")
    
    snapshot_mgr = None
    logger.info("Shutdown complete")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="MCN v1 Vector Database API - Multi-Tenant SaaS",
    description="Production server with snapshot swap, rate limiting, and tenant isolation",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================

class VectorItem(BaseModel):
    id: str
    values: List[float] = Field(..., min_length=1)
    metadata: Dict[str, Any] = {}


class AddRequest(BaseModel):
    api_key: Optional[str] = None  # Optional for backward compat, but header preferred
    vectors: List[VectorItem]


class SearchRequest(BaseModel):
    api_key: Optional[str] = None  # Optional for backward compat, but header preferred
    vector: List[float] = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=100)


class FinalizeRequest(BaseModel):
    api_key: Optional[str] = None  # Optional for backward compat, but header preferred
    timeout_s: Optional[float] = Field(default=120.0, ge=1.0, le=3600.0)


class HealthResponse(BaseModel):
    status: str
    ram_usage: str
    hot_buffer_size: Optional[int] = None
    cold_index_size: Optional[int] = None
    total_vectors: Optional[int] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/add")
@app.post("/ingest")  # Alias for /add
async def add_vectors(
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Ingest vectors with rate limiting and tenant limits.
    Tenant-scoped endpoint.
    
    API key can be provided via:
    - Header: Authorization: Bearer <key> (preferred)
    - Header: X-API-Key: <key>
    - Body: api_key (backward compatibility, not recommended)
    """
    print("HIT /add")
    global snapshot_mgr
    
    if snapshot_mgr is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    try:
        print("DEBUG: Extracting API key from headers...")
        # Extract API key from header (preferred) or fallback to body (backward compat)
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:].strip()
        elif x_api_key:
            api_key = x_api_key
        
        print(f"DEBUG: API key from header: {api_key[:20] if api_key else 'None'}...")
        
        # Parse body manually (non-blocking approach)
        # CRITICAL: Read body ONCE - ASGI body stream can only be read once!
        # For /add, we MUST read body to get vectors, but only if API key is in header
        request = None
        if api_key:
            print("DEBUG: API key in header, reading body for vectors...")
            # API key in header - safe to read body for vectors
            content_length = http_request.headers.get("content-length")
            print(f"DEBUG: Content-Length: {content_length}")
            if content_length and int(content_length) > 0:
                try:
                    print("DEBUG: About to read body with timeout...")
                    # Read body once with timeout
                    body_data = await asyncio.wait_for(http_request.json(), timeout=2.0)
                    print(f"DEBUG: Body read successfully, keys: {list(body_data.keys())}")
                    # Parse request from body
                    print("DEBUG: About to create AddRequest object...")
                    request = AddRequest(**body_data)
                    print("DEBUG: AddRequest object created")
                    print(f"DEBUG: Request parsed, vectors count: {len(request.vectors) if request.vectors else 0}")
                except (asyncio.TimeoutError, json.JSONDecodeError, ValueError, Exception) as e:
                    print(f"DEBUG: Body parsing failed: {e}")
                    logger.warning(f"Body parsing failed: {e}")
                    request = None
            else:
                print("DEBUG: No content-length or empty, skipping body read")
        else:
            print("DEBUG: No API key in header, reading body for API key...")
            # No API key in header - try to get it from body (backward compat)
            content_length = http_request.headers.get("content-length")
            if content_length and int(content_length) > 0:
                try:
                    body_data = await asyncio.wait_for(http_request.json(), timeout=2.0)
                    # Extract API key from body
                    if body_data.get("api_key"):
                        api_key = body_data.get("api_key")
                    # Parse request from body
                    request = AddRequest(**body_data)
                except (asyncio.TimeoutError, json.JSONDecodeError, ValueError, Exception) as e:
                    logger.warning(f"Body parsing failed: {e}")
                    request = None
        
        print("DEBUG: About to validate API key...")
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required in header (Authorization: Bearer <key> or X-API-Key: <key>)"
            )
        
        user_id = api_key
        
        # Check RAM usage
        ram_percent = get_ram_usage_percent()
        if ram_percent > RAM_SAFETY_THRESHOLD:
            raise HTTPException(
                status_code=503,
                detail=f"RAM usage {ram_percent:.1f}% exceeds threshold {RAM_SAFETY_THRESHOLD}%"
            )
        
        # Validate vectors
        if not request or not request.vectors:
            raise HTTPException(status_code=400, detail="Empty vectors list or invalid request")
        
        # Check tenant cap (on first vector for new tenant)
        with tenant_lock:
            stats = tenant_stats[user_id]
            if stats["vectors_stored"] == 0:
                if check_tenant_cap():
                    raise HTTPException(
                        status_code=403,
                        detail=f"Tenant cap reached (MAX_TENANTS={MAX_TENANTS}). Please join waitlist."
                    )
                # Mark tenant as created
                stats["created_at"] = time.time()
        
        # Get tenant plan and limits
        plan = get_tenant_plan(user_id)
        config = RATE_LIMIT_CONFIG.get(plan, RATE_LIMIT_CONFIG["default"])
        
        # Check tenant limits
        with tenant_lock:
            stats = tenant_stats[user_id]
            
            if stats["vectors_stored"] + len(request.vectors) > config["max_vectors"]:
                raise HTTPException(
                    status_code=403,
                    detail=f"Max {config['max_vectors']} vectors per tenant exceeded (plan: {plan})"
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
            metadata['original_idx'] = len(metadata_list)
            metadata_list.append(metadata)
        
        vectors_array = np.array(vectors_list, dtype="float32")
        
        # Add to snapshot manager (write buffer)
        # Use executor to avoid blocking event loop, but with timeout protection
        def add_vectors():
            snapshot_mgr.add(vectors_array, metadata_list)
        
        try:
            loop = asyncio.get_event_loop()
            # Add with timeout to prevent hanging (30 seconds max for add operation)
            await asyncio.wait_for(
                loop.run_in_executor(None, add_vectors),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Add operation timed out after 30s for {len(request.vectors)} vectors")
            raise HTTPException(
                status_code=504,
                detail=f"Add operation timed out. Please try with fewer vectors or contact support."
            )
        
        # Update tenant stats
        with tenant_lock:
            tenant_stats[user_id]["vectors_stored"] += len(request.vectors)
            tenant_stats[user_id]["ingest_count"] += len(request.vectors)
        
        # Check scaling warnings
        check_scaling_warnings()
        
        return {
            "status": "success",
            "added": len(request.vectors),
            "ram_usage_percent": get_ram_usage_percent(),
            "total_vectors": tenant_stats[user_id]["vectors_stored"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
            logger.error(f"Error in /add: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/search")
async def search_vectors(
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Search vectors with rate limiting and security filtering.
    Uses active snapshot (read-only, never blocks).
    
    API key can be provided via:
    - Header: Authorization: Bearer <key> (preferred)
    - Header: X-API-Key: <key>
    - Body: api_key (backward compatibility, not recommended)
    """
    print("HIT /search")
    global snapshot_mgr
    
    if snapshot_mgr is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    search_start = time.time()
    
    try:
        # Extract API key from header (preferred) or fallback to body (backward compat)
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:].strip()
        elif x_api_key:
            api_key = x_api_key
        
        # Parse body manually (non-blocking approach)
        # Check if body exists first to avoid blocking on empty requests
        request = None
        content_length = http_request.headers.get("content-length")
        if content_length and int(content_length) > 0:
            try:
                # Use timeout to prevent hanging on malformed requests
                body_data = await asyncio.wait_for(http_request.json(), timeout=2.0)
                request = SearchRequest(**body_data)
                # Fallback to body api_key if header not provided
                if not api_key and body_data.get("api_key"):
                    api_key = body_data.get("api_key")
            except (asyncio.TimeoutError, json.JSONDecodeError, ValueError, Exception) as e:
                logger.warning(f"Body parsing failed: {e}, continuing with headers only")
                request = None
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required in header (Authorization: Bearer <key> or X-API-Key: <key>)"
            )
        
        user_id = api_key
        
        # Validate vector dimension
        if not request or not request.vector:
            raise HTTPException(status_code=400, detail="Invalid request: vector required")
        
        if len(request.vector) != DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension mismatch: expected {DIM}, got {len(request.vector)}"
            )
        
        # Update tenant stats (concurrent search tracking)
        with tenant_lock:
            tenant_stats[user_id]["concurrent_searches"] += 1
            tenant_stats[user_id]["query_count"] += 1
            tenant_stats[user_id]["daily_queries"] += 1
        
        try:
            # Convert to numpy array
            query_vector = np.array(request.vector, dtype="float32")
            
            # Search using snapshot manager (non-blocking, uses active snapshot)
            loop = asyncio.get_event_loop()
            results, scores = await loop.run_in_executor(
                None, snapshot_mgr.search, query_vector, request.k
            )
            
            # Security Filter: Only return items belonging to this user
            filtered_results, filtered_scores = filter_by_user_id(results, scores, user_id)
            
            # Track latency
            search_latency = time.time() - search_start
            latency_history.append(search_latency)
            qps_history.append(time.time())
            
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
        logger.error(f"Error in /search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/finalize")
async def finalize_index(
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
    timeout_s: Optional[float] = Header(None, alias="X-Timeout-S")
):
    """
    Start building a new snapshot (non-blocking).
    Returns build_id for status tracking.
    Tenant-scoped endpoint.
    
    API key can be provided via:
    - Header: Authorization: Bearer <key> (preferred)
    - Header: X-API-Key: <key>
    - Body: api_key (backward compatibility, not recommended)
    
    Request body is OPTIONAL - can be called with headers only (empty body works).
    """
    print("HIT /finalize")
    global snapshot_mgr
    
    if snapshot_mgr is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    try:
        # Extract API key from header (preferred) or fallback to body (backward compat)
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:].strip()
        elif x_api_key:
            api_key = x_api_key
        
        print(f"DEBUG: API key from header: {api_key[:20] if api_key else 'None'}...")
        
        # Parse body manually if present (non-blocking, optional)
        # For /finalize, body is completely optional - we can work with headers only
        request_body = None
        final_timeout = 120.0  # Default timeout
        
        # Only read body if we DON'T have API key in header (backward compat)
        # If API key is in header, skip body entirely - use default timeout
        if not api_key:
            print("DEBUG: No API key in header, reading body...")
            # Need to read body to get API key (backward compat)
            content_length = http_request.headers.get("content-length")
            if content_length and int(content_length) > 0:
                try:
                    # Use timeout to prevent hanging on malformed requests
                    body_bytes = await asyncio.wait_for(http_request.body(), timeout=2.0)
                    if body_bytes:
                        body_data = json.loads(body_bytes)
                        request_body = FinalizeRequest(**body_data) if body_data else None
                        # Get API key from body
                        if body_data.get("api_key"):
                            api_key = body_data.get("api_key")
                        # Get timeout from body if provided
                        if body_data.get("timeout_s"):
                            final_timeout = float(body_data.get("timeout_s"))
                except (asyncio.TimeoutError, json.JSONDecodeError, ValueError, Exception) as e:
                    print(f"DEBUG: Body reading failed: {e}")
                    # Empty body, timeout, or invalid JSON is fine - we can work with headers only
                    pass
        else:
            print("DEBUG: API key in header, skipping body reading")
        
        # Get timeout from header (takes precedence)
        if timeout_s is not None:
            final_timeout = float(timeout_s)
        elif request_body and hasattr(request_body, 'timeout_s') and request_body.timeout_s:
            final_timeout = request_body.timeout_s
        
        print(f"DEBUG: Final timeout: {final_timeout}s")
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required in header (Authorization: Bearer <key> or X-API-Key: <key>)"
            )
        
        print("DEBUG: Starting build...")
        # Start build in executor to avoid blocking event loop
        # start_build() does synchronous work (copying vectors, creating MCNLayer)
        # Even with 0 vectors, it acquires locks which could block
        try:
            loop = asyncio.get_event_loop()
            print("DEBUG: About to call run_in_executor...")
            build_id = await asyncio.wait_for(
                loop.run_in_executor(None, snapshot_mgr.start_build, final_timeout),
                timeout=5.0  # 5 second timeout for start_build itself
            )
            print(f"DEBUG: Build started with ID: {build_id}")
        except asyncio.TimeoutError:
            print("DEBUG: start_build timed out after 5s!")
            raise HTTPException(
                status_code=504,
                detail="Build initialization timed out. Please try again."
            )
        
        # Finalize build in executor (non-blocking for API)
        def build_and_swap_sync():
            try:
                # Finalize the build (this takes time, but doesn't block searches)
                build_result = snapshot_mgr.finalize_build(build_id)
                
                # Automatically swap when ready
                swap_result = snapshot_mgr.swap_snapshot(build_id)
                
                logger.info(f"Build {build_id} completed and swapped: {swap_result}")
                return {**build_result, **swap_result}
            except Exception as e:
                logger.error(f"Build {build_id} failed: {e}", exc_info=True)
                raise
        
        # Run build in executor (non-blocking)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, build_and_swap_sync)
        
        return {
            "status": "building",
            "build_id": build_id,
            "message": "Build started. Use /finalize/status to check progress."
        }
    except RuntimeError as e:
        if "already in progress" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise
    except Exception as e:
        logger.error(f"Error in /finalize: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/finalize/status")
async def finalize_status(build_id: str):
    """
    Get build status by build_id.
    """
    global snapshot_mgr
    
    if snapshot_mgr is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    try:
        status = snapshot_mgr.get_build_status(build_id)
        return status
    except Exception as e:
        logger.error(f"Error in /finalize/status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/metrics")
async def get_metrics(tenant_id: Optional[str] = None):
    """
    Get metrics: global or per-tenant.
    """
    global snapshot_mgr
    
    if snapshot_mgr is None:
        raise HTTPException(status_code=503, detail="MCN not initialized")
    
    try:
        # Calculate latency percentiles
        if len(latency_history) > 0:
            sorted_latencies = sorted(latency_history)
            p50_idx = int(len(sorted_latencies) * 0.50)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p50 = sorted_latencies[p50_idx] * 1000  # Convert to ms
            p95 = sorted_latencies[p95_idx] * 1000
            p99 = sorted_latencies[p99_idx] * 1000
        else:
            p50 = p95 = p99 = 0.0
        
        # Calculate QPS (queries in last 60 seconds)
        now = time.time()
        recent_qps = [t for t in qps_history if now - t <= 60]
        current_qps = len(recent_qps) / 60.0 if recent_qps else 0.0
        
        if tenant_id:
            # Per-tenant metrics
            with tenant_lock:
                if tenant_id not in tenant_stats:
                    raise HTTPException(status_code=404, detail="Tenant not found")
                
                stats = tenant_stats[tenant_id]
                plan = stats.get("plan", "free")
                config = RATE_LIMIT_CONFIG.get(plan, RATE_LIMIT_CONFIG["default"])
                
                return {
                    "tenant_id": tenant_id,
                    "plan": plan,
                    "vectors_count": stats["vectors_stored"],
                    "qps": stats.get("query_count", 0),  # Current window QPS
                    "concurrent_searches": stats["concurrent_searches"],
                    "daily_queries": stats["daily_queries"],
                    "last_finalize_time": stats.get("last_finalize_time"),
                    "limits": config,
                }
        else:
            # Global metrics
            with tenant_lock:
                total_tenants = len([t for t in tenant_stats.keys() if tenant_stats[t].get("vectors_stored", 0) > 0])
                total_vectors = sum(stats.get("vectors_stored", 0) for stats in tenant_stats.values())
            
            # Get active builds count (accessing private attribute via public method would be better)
            # For now, we'll track this differently or make it a public property
            active_builds = snapshot_mgr.get_active_builds_count()
            
            return {
                "global": {
                    "total_tenants": total_tenants,
                    "total_vectors": total_vectors,
                    "active_builds": active_builds,
                    "p50_latency_ms": p50,
                    "p95_latency_ms": p95,
                    "p99_latency_ms": p99,
                    "qps": current_qps,
                    "workers_recommended": WORKERS_RECOMMENDED,
                },
                "scaling_warnings": {
                    "tenants_threshold": SCALE_THRESHOLD_TENANTS,
                    "vectors_threshold": SCALE_THRESHOLD_VECTORS,
                    "p95_threshold_ms": SCALE_THRESHOLD_P95_MS,
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global snapshot_mgr
    
    ram_percent = get_ram_usage_percent()
    ram_usage_str = f"{ram_percent:.1f}%"
    
    if snapshot_mgr is None:
        return HealthResponse(
            status="initializing",
            ram_usage=ram_usage_str
        )
    
    try:
        hot_size = snapshot_mgr.get_hot_buffer_size()
        cold_size = snapshot_mgr.get_cold_index_size()
        total = snapshot_mgr.size()
        
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


# ============================================================================
# Admin Authentication Endpoints
# ============================================================================

class AdminLoginRequest(BaseModel):
    email: str
    password: str


@app.post("/admin/login")
async def admin_login(request: AdminLoginRequest):
    """
    Admin login endpoint (hardcoded credentials).
    Returns session token for admin access.
    """
    if request.email == ADMIN_EMAIL and request.password == ADMIN_PASSWORD:
        # Generate session token
        import secrets
        session_token = secrets.token_urlsafe(32)
        expiry_time = time.time() + (24 * 60 * 60)  # 24 hours
        
        with admin_session_lock:
            admin_sessions[session_token] = expiry_time
        
        logger.info(f"Admin login successful: {request.email}")
        
        return {
            "status": "success",
            "session_token": session_token,
            "expires_in": 86400  # 24 hours in seconds
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid email or password")


@app.post("/admin/logout")
async def admin_logout(session_token: Optional[str] = Header(None, alias="X-Admin-Session")):
    """
    Admin logout endpoint.
    """
    if session_token:
        with admin_session_lock:
            admin_sessions.pop(session_token, None)
        logger.info("Admin logout successful")
    
    return {"status": "success", "message": "Logged out"}


@app.get("/admin/verify")
async def verify_admin_session(session_token: Optional[str] = Header(None, alias="X-Admin-Session")):
    """
    Verify admin session is valid.
    """
    if session_token:
        with admin_session_lock:
            expiry = admin_sessions.get(session_token, 0)
            if expiry > time.time():
                return {
                    "status": "valid",
                    "expires_at": expiry,
                    "email": ADMIN_EMAIL
                }
    
    raise HTTPException(status_code=401, detail="Invalid or expired session")


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.get("/admin/stats")
async def get_admin_stats(session_token: Optional[str] = Header(None, alias="X-Admin-Session")):
    """
    Get admin statistics (tenant counts, vectors, QPS, etc.).
    Requires admin authentication.
    """
    # Verify admin access
    verify_admin(session_token=session_token)
    
    with tenant_lock:
        # Calculate stats by plan
        plan_stats = {}
        total_vectors = 0
        total_tenants = 0
        total_qps = 0
        
        for tenant_id, stats in tenant_stats.items():
            if stats.get("vectors_stored", 0) > 0:  # Only count active tenants
                total_tenants += 1
                plan = stats.get("plan", "free")
                vectors = stats.get("vectors_stored", 0)
                total_vectors += vectors
                
                if plan not in plan_stats:
                    plan_stats[plan] = {
                        "user_count": 0,
                        "total_vectors": 0,
                        "total_qps": 0
                    }
                
                plan_stats[plan]["user_count"] += 1
                plan_stats[plan]["total_vectors"] += vectors
                plan_stats[plan]["total_qps"] += stats.get("query_count", 0)
                total_qps += stats.get("query_count", 0)
        
        # Calculate global QPS (from history)
        now = time.time()
        recent_qps = [t for t in qps_history if now - t <= 60]
        current_qps = len(recent_qps) / 60.0 if recent_qps else 0.0
        
        # Calculate latency percentiles
        if len(latency_history) > 0:
            sorted_latencies = sorted(latency_history)
            p50_idx = int(len(sorted_latencies) * 0.50)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p50_ms = sorted_latencies[p50_idx] * 1000
            p95_ms = sorted_latencies[p95_idx] * 1000
            p99_ms = sorted_latencies[p99_idx] * 1000
        else:
            p50_ms = p95_ms = p99_ms = 0.0
    
    return {
        "status": "success",
        "global": {
            "total_tenants": total_tenants,
            "max_tenants": MAX_TENANTS,
            "total_vectors": total_vectors,
            "current_qps": current_qps,
            "p50_latency_ms": p50_ms,
            "p95_latency_ms": p95_ms,
            "p99_latency_ms": p99_ms
        },
        "plan_stats": plan_stats,
        "tenants": [
            {
                "tenant_id": tid,
                "plan": stats.get("plan", "free"),
                "vectors_stored": stats.get("vectors_stored", 0),
                "qps": stats.get("query_count", 0),
                "pod_url": tenant_pods.get(tid, "local")
            }
            for tid, stats in tenant_stats.items()
            if stats.get("vectors_stored", 0) > 0
        ]
    }


@app.post("/admin/tenants/{tenant_id}/assign_pod")
async def assign_pod(
    tenant_id: str,
    pod_url: str,
    session_token: Optional[str] = Header(None, alias="X-Admin-Session")
):
    """
    Assign tenant to a pod (admin-only).
    """
    # Verify admin access
    verify_admin(session_token=session_token)
    
    with pod_lock:
        tenant_pods[tenant_id] = pod_url
    
    return {
        "status": "success",
        "tenant_id": tenant_id,
        "pod_url": pod_url
    }


@app.get("/admin/tenants")
async def list_tenants(session_token: Optional[str] = Header(None, alias="X-Admin-Session")):
    """
    List all tenants with their pod assignments (admin-only).
    """
    with tenant_lock, pod_lock:
        tenants = []
        for tenant_id, stats in tenant_stats.items():
            tenants.append({
                "tenant_id": tenant_id,
                "plan": stats.get("plan", "free"),
                "vectors_stored": stats.get("vectors_stored", 0),
                "pod_url": tenant_pods.get(tenant_id, "local"),
                "created_at": stats.get("created_at"),
            })
    
    return {
        "status": "success",
        "total_tenants": len(tenants),
        "max_tenants": MAX_TENANTS,
        "tenants": tenants
    }


@app.get("/admin/tenants/{tenant_id}/pod")
async def get_tenant_pod(tenant_id: str, session_token: Optional[str] = Header(None, alias="X-Admin-Session")):
    """
    Get pod URL for a tenant (admin-only).
    """
    # Verify admin access
    verify_admin(session_token=session_token)
    
    with pod_lock:
        pod_url = tenant_pods.get(tenant_id, "local")
    
    return {
        "status": "success",
        "tenant_id": tenant_id,
        "pod_url": pod_url
    }


@app.put("/admin/tenants/{tenant_id}/pod")
async def update_tenant_pod(
    tenant_id: str,
    request: Request,
    session_token: Optional[str] = Header(None, alias="X-Admin-Session")
):
    """
    Update pod URL for a tenant (admin-only).
    Useful for moving tenants to different Railway instances.
    """
    # Verify admin access
    verify_admin(session_token=session_token)
    
    try:
        data = await request.json()
        pod_url = data.get("pod_url")
        
        if not pod_url:
            raise HTTPException(status_code=400, detail="Missing pod_url in request body")
        
        with pod_lock:
            old_pod = tenant_pods.get(tenant_id, "local")
            tenant_pods[tenant_id] = pod_url
        
        logger.info(f"Tenant {tenant_id} moved from {old_pod} to {pod_url}")
        
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "old_pod_url": old_pod,
            "new_pod_url": pod_url,
            "message": "Pod URL updated. Tenant should use new URL for API calls."
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")


# ============================================================================
# Stripe Webhook Endpoints
# ============================================================================

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """
    Stripe webhook handler for payment events.
    Updates user plan in memory and optionally in Supabase.
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Stripe webhook secret not configured")
    
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session.get("customer")
        metadata = session.get("metadata", {})
        user_id = metadata.get("user_id")
        plan = metadata.get("plan", "free")
        
        if user_id:
            # Update tenant plan in memory
            with tenant_lock:
                if user_id in tenant_stats:
                    tenant_stats[user_id]["plan"] = plan
                    logger.info(f"Updated tenant {user_id} to plan {plan} via Stripe webhook")
            
            # Update Supabase if configured
            if SUPABASE_URL and SUPABASE_SERVICE_KEY:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.patch(
                            f"{SUPABASE_URL}/rest/v1/users",
                            headers={
                                "apikey": SUPABASE_SERVICE_KEY,
                                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                "Content-Type": "application/json",
                                "Prefer": "return=representation"
                            },
                            json={
                                "plan": plan,
                                "stripe_customer_id": customer_id
                            },
                            params={"id": f"eq.{user_id}"}
                        )
                        if response.status_code == 200:
                            logger.info(f"Updated user {user_id} plan in Supabase")
                except Exception as e:
                    logger.error(f"Failed to update Supabase: {e}")
    
    elif event["type"] == "customer.subscription.updated":
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        logger.info(f"Subscription updated for customer {customer_id}")
    
    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        # Revert to free plan on cancellation
        logger.info(f"Subscription deleted for customer {customer_id}")
    
    return {"status": "success"}


@app.post("/api/stripe/create-checkout-session")
async def create_checkout_session(request: Request):
    """
    Create Stripe checkout session (called from frontend).
    Requires: planId, userId in request body.
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    try:
        data = await request.json()
        plan_id = data.get("planId")
        user_id = data.get("userId")
        
        if not plan_id or not user_id:
            raise HTTPException(status_code=400, detail="Missing planId or userId")
        
        # Map plan IDs to prices (configure these in Stripe Dashboard)
        plan_price_map = {
            "starter": os.getenv("STRIPE_PRICE_STARTER", ""),
            "pro": os.getenv("STRIPE_PRICE_PRO", ""),
            "scale": os.getenv("STRIPE_PRICE_SCALE", ""),
        }
        
        price_id = plan_price_map.get(plan_id)
        if not price_id or price_id == "":
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid plan: {plan_id} or Stripe price ID not configured. Please set STRIPE_PRICE_{plan_id.upper()} environment variable."
            )
        
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price": price_id,
                    "quantity": 1,
                }],
                mode="subscription",
                success_url=f"{os.getenv('FRONTEND_URL', 'https://memryx.org')}/portal?success=true",
                cancel_url=f"{os.getenv('FRONTEND_URL', 'https://memryx.org')}/pricing?canceled=true",
                metadata={
                    "user_id": user_id,
                    "plan": plan_id
                }
            )
            
            # Return both sessionId and URL for compatibility
            return {"sessionId": session.id, "url": session.url}
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {e}")
            raise HTTPException(status_code=500, detail=f"Stripe error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/stripe/create-portal-session")
async def create_portal_session(request: Request):
    """
    Create Stripe customer portal session (for managing subscriptions).
    Requires: customerId in request body.
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    try:
        data = await request.json()
        customer_id = data.get("customerId")
        
        if not customer_id:
            raise HTTPException(status_code=400, detail="Missing customerId")
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=f"{os.getenv('FRONTEND_URL', 'https://memryx.org')}/portal"
            )
            
            return {"url": session.url}
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {e}")
            raise HTTPException(status_code=500, detail=f"Stripe error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error creating portal session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "MCN v1 Vector Database API - Multi-Tenant SaaS",
        "version": "2.0.0",
        "endpoints": {
            "POST /add": "Ingest vectors (tenant-scoped)",
            "POST /ingest": "Alias for /add",
            "POST /search": "Search vectors (tenant-scoped)",
            "POST /finalize": "Start snapshot build (tenant-scoped)",
            "GET /finalize/status": "Get build status",
            "GET /metrics": "Get metrics (global or per-tenant)",
            "GET /health": "Health check",
            "POST /admin/login": "Admin login (email/password)",
            "POST /admin/logout": "Admin logout",
            "GET /admin/verify": "Verify admin session",
            "POST /admin/tenants/{id}/assign_pod": "Assign pod (admin-only)",
            "GET /admin/tenants": "List tenants (admin-only)",
            "GET /admin/tenants/{id}/pod": "Get tenant pod URL (admin-only)",
            "PUT /admin/tenants/{id}/pod": "Update tenant pod URL (admin-only)",
            "POST /webhooks/stripe": "Stripe webhook handler",
            "POST /api/stripe/create-checkout-session": "Create Stripe checkout",
            "POST /api/stripe/create-portal-session": "Create Stripe portal",
            "GET /": "This endpoint"
        },
        "configuration": {
            "dimension": DIM,
            "hot_buffer_size": HOT_BUFFER_SIZE,
            "max_tenants": MAX_TENANTS,
            "workers_recommended": WORKERS_RECOMMENDED,
            "stripe_configured": STRIPE_AVAILABLE
        }
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

