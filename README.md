# Memryx Backend

Production-ready multi-tenant vector database backend with 12.71x compression and exact recall parity.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn server_v2:app --host 0.0.0.0 --port 8000 --workers 1
```

### Environment Variables

```bash
MAX_TENANTS=20
WORKERS_RECOMMENDED=1
ADMIN_API_KEY=your-secret-admin-key

# Optional: Stripe
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Optional: Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
```

## API Endpoints

- `POST /add` - Ingest vectors
- `POST /search` - Search vectors
- `POST /finalize` - Build index
- `GET /finalize/status` - Check build status
- `GET /metrics` - Get metrics
- `GET /health` - Health check

See `server_v2.py` for complete API documentation.

## Deployment

Deploy to Railway with:

```bash
uvicorn server_v2:app --host 0.0.0.0 --port $PORT --workers 1
```

## License

Proprietary

