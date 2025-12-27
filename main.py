"""
Main entry point for Railway deployment.
Imports the FastAPI app from server_v2.py.
"""
from server_v2 import app

# Export app for uvicorn
__all__ = ["app"]

