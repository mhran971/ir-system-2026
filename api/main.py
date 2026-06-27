"""
IR System 2026 — REST API Gateway
Run from the project root so relative data/ paths resolve correctly:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Ensure the project root is on sys.path (mirrors ui/app.py line 12).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import services
from api.routes import (
    documents,
    evaluation,
    refine,
    search,
    stats,
    term_details,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all IR services once before the first request."""
    services.init_all()
    yield


app = FastAPI(
    title="IR System 2026 API",
    description=(
        "REST gateway for the IR System 2026 retrieval engine. "
        "Exposes the same operations as the Streamlit GUI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(search.router)
app.include_router(refine.router)
app.include_router(stats.router)
app.include_router(evaluation.router)
app.include_router(term_details.router)
app.include_router(documents.router)


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "IR System 2026 API is running. Visit /docs for the full API reference."}
