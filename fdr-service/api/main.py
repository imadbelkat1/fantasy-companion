from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from .endpoints.teams import router as teams_router
from .endpoints.fixtures import router as fixtures_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FDR Service API",
    description="Football Data Ratings and Fixture Difficulty Service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(teams_router, prefix="/api/v1", tags=["teams"])
app.include_router(fixtures_router, prefix="/api/v1", tags=["fixtures"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FDR Service API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}