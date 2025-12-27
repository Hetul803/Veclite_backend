# Railway Storage Setup Guide

## Overview

Memryx stores vectors in **memory** for fast access, but needs **persistent disk storage** to survive server restarts.

**Without persistence**: All vectors are lost when Railway restarts the service.

**With persistence**: Vectors are saved to disk after each `finalize` operation and loaded on startup.

---

## Railway Volume Setup

### Step 1: Add Volume to Railway

1. Go to your Railway project dashboard
2. Click on your service
3. Go to **"Volumes"** tab
4. Click **"Add Volume"**
5. Set:
   - **Name**: `memryx-data` (or any name)
   - **Mount Path**: `/data` (this is the default)
   - **Size**: Start with **10GB** (you can increase later)

### Step 2: Set Environment Variables

Add these to your Railway service:

```bash
# Enable persistence (set to "false" to disable)
ENABLE_PERSISTENCE=true

# Storage path (should match Railway volume mount path)
STORAGE_PATH=/data
```

**Note**: If you use a different mount path in Railway, update `STORAGE_PATH` to match.

---

## How It Works

### Automatic Saving
- **When**: After each successful `finalize` operation
- **What**: Saves the complete snapshot (vectors, clusters, metadata)
- **Where**: `/data/snapshot_latest.pkl` (on Railway volume)

### Automatic Loading
- **When**: On server startup
- **What**: Loads the latest snapshot from disk
- **Fallback**: If no snapshot exists, starts fresh (empty index)

### Storage Format
- Uses Python `pickle` format
- Contains: vectors, metadata, cluster index, configuration
- File size: ~150KB per 1K vectors (compressed)

---

## Storage Size Estimates

| Vectors | Estimated Size | Recommended Volume |
|---------|---------------|-------------------|
| 10K     | ~1.5 MB       | 1 GB              |
| 100K    | ~15 MB        | 10 GB             |
| 1M      | ~150 MB       | 50 GB             |
| 10M     | ~1.5 GB       | 100 GB            |

**Note**: Actual size depends on metadata size and compression.

---

## Testing Persistence

### 1. Add vectors and finalize:
```bash
curl -X POST https://your-railway-url.railway.app/add \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your-api-key",
    "vectors": [...]
  }'

curl -X POST https://your-railway-url.railway.app/finalize \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key"}'
```

### 2. Check logs:
You should see:
```
Snapshot saved: /data/snapshot_latest.pkl (1000 vectors)
```

### 3. Restart service:
- In Railway, click "Restart" on your service

### 4. Check logs again:
You should see:
```
Attempting to load existing snapshot from disk...
Snapshot loaded: /data/snapshot_latest.pkl (1000 vectors)
```

### 5. Verify:
- Search should still work with the same vectors
- No need to re-ingest vectors

---

## Troubleshooting

### "Snapshot file not found"
- **Cause**: No previous finalize completed, or volume not mounted
- **Fix**: This is normal on first run. Add vectors and finalize.

### "Error saving snapshot"
- **Cause**: Volume not mounted, or insufficient permissions
- **Fix**: 
  1. Check Railway volume is attached
  2. Verify `STORAGE_PATH` matches mount path
  3. Check Railway logs for permission errors

### "Error loading snapshot"
- **Cause**: Corrupted file, or incompatible format
- **Fix**: 
  1. Check file exists: `ls -la /data/`
  2. If corrupted, delete and re-finalize
  3. Check Railway logs for details

### Vectors still lost after restart
- **Cause**: `ENABLE_PERSISTENCE=false` or `STORAGE_PATH` incorrect
- **Fix**: 
  1. Check environment variables in Railway
  2. Verify volume is mounted
  3. Check logs for "Persistence enabled" message

---

## Disabling Persistence

If you want to disable persistence (for testing, or if you don't need it):

```bash
ENABLE_PERSISTENCE=false
```

**Warning**: All vectors will be lost on restart if persistence is disabled.

---

## Manual Backup

To backup snapshots manually:

1. **Via Railway CLI**:
```bash
railway run ls -la /data/
railway run cp /data/snapshot_latest.pkl /data/backup_$(date +%Y%m%d).pkl
```

2. **Via Railway Dashboard**:
- Use Railway's volume export feature (if available)

---

## Multi-Instance Considerations

**Important**: Each Railway instance has its own volume. If you run multiple instances:

- Each instance saves to its own volume
- Load balancing will route requests to different instances
- **Solution**: Use pod routing (assign tenants to specific instances) OR use shared storage (S3, etc.)

For now, with 1-2 instances, this is fine. For production scale, consider:
- Shared storage (S3, GCS)
- Database-backed persistence
- Distributed file system

---

## Summary

✅ **What you need to do**:
1. Add Railway volume (mount at `/data`)
2. Set `ENABLE_PERSISTENCE=true`
3. Set `STORAGE_PATH=/data` (or your mount path)
4. Deploy

✅ **What happens automatically**:
- Saves after each `finalize`
- Loads on startup
- No code changes needed

✅ **Storage size**:
- Start with 10GB volume
- Increase as needed (Railway allows resizing)

---

## Code Changes Made

The following code was added to support persistence:

1. **`SnapshotManager.save_snapshot()`**: Saves snapshot to disk
2. **`SnapshotManager.load_snapshot()`**: Loads snapshot from disk
3. **Auto-save**: After successful finalize/swap
4. **Auto-load**: On server startup

No changes needed to your application code - it's all automatic!

