import logging
import traceback
import os
import sys
import asyncio
import httpx
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from database import connect_to_mongo, close_mongo_connection
from api_routes import router as api_router

# USE ABSOLUTE PATHS
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BACKEND_DIR, "backend_errors.log")
TRACEBACK_FILE = os.path.join(BACKEND_DIR, "TRACEBACK.txt")

# Ensure handlers are attached to the uvicorn logger specifically
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

uvicorn_logger = logging.getLogger("uvicorn.error")
uvicorn_logger.addHandler(file_handler)
uvicorn_logger.setLevel(logging.INFO)

# Also configure root logger just in case
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), file_handler])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    uvicorn_logger.info("--- SERVER STARTING UP (v1.3.1) ---")
    await connect_to_mongo()
    
    # Start Stay-Awake Pulse (Every 10 minutes)
    stop_event = asyncio.Event()
    pulse_task = asyncio.create_task(stay_awake_pulse(stop_event))
    
    yield
    
    # Shutdown Pulse
    stop_event.set()
    await pulse_task
    await close_mongo_connection()

async def stay_awake_pulse(stop_event: asyncio.Event):
    """Internal task to keep the server warm by self-pinging."""
    # Wait for the server to actually be ready
    await asyncio.sleep(5)
    
    url = "http://localhost:8000/api/health"
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            try:
                # We use localhost here because internal pings don't count towards Render sleep,
                # but they keep the internal worker process active and logs moving.
                res = await client.get(url)
                uvicorn_logger.info(f"--- STAY-AWAKE PULSE: {res.status_code} ---")
            except Exception as e:
                uvicorn_logger.warning(f"Stay-awake pulse failed: {e}")
            
            # Sleep for 10 minutes (600 seconds)
            # Use shorter intervals during testing if needed, but 10m is standard for 'keep-alive'
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=600)
            except asyncio.TimeoutError:
                continue

app = FastAPI(
    title="AI Dataset Diagnosis & Auto-Fix System API",
    version="1.3.1",
    lifespan=lifespan,
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = traceback.format_exc()
    uvicorn_logger.error(f"CRITICAL ERROR: {exc}\n{error_msg}")
    
    # Attempt to write to file
    try:
        with open(TRACEBACK_FILE, "w", encoding="utf-8") as f:
            f.write(error_msg)
    except Exception:
        pass
        
    # RETURN THE ERROR IN THE RESPONSE SO THE USER CAN SEE IT IN STREAMLIT
    return JSONResponse(
        status_code=500,
        content={
            "detail": "CRITICAL_BACKEND_ERROR",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": error_msg  # <--- THIS IS THE KEY
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "backend_dir": BACKEND_DIR,
        "python_version": sys.version
    }

@app.get("/")
def read_root():
    return {"message": "API v1.3.0 Ready"}
