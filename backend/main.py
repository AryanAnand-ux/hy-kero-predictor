"""
main.py — FastAPI Backend for HY Kero Flash Point Predictor

Features:
  - Pre-loads model artifacts at startup
  - CORS configured via environment variable
  - Request logging middleware
  - Health check endpoint
"""
import os
import time
import logging
import asyncio
import hmac
from uuid import uuid4
from contextlib import asynccontextmanager
from collections import defaultdict

from dotenv import load_dotenv

try:
    from .constants import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS, RATE_LIMIT_CLEANUP_INTERVAL
except ImportError:
    from constants import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS, RATE_LIMIT_CLEANUP_INTERVAL

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request, Security, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .routes import predict, history, models, upload, chat
except ImportError:
    from routes import predict, history, models, upload, chat

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hykero")


# ── Rate Limiter (thread-safe with automatic cleanup) ─────────────────────────
class RateLimiter:
    def __init__(self, requests_limit: int, window_seconds: int, cleanup_interval: int = 300):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval
        self.history: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()

    async def check(self, ip: str) -> bool:
        async with self._lock:
            now = time.time()
            # Periodic cleanup of stale IPs to prevent memory leak
            if now - self._last_cleanup > self.cleanup_interval:
                stale_ips = [
                    k for k, v in self.history.items()
                    if not v or (now - v[-1]) > self.window_seconds
                ]
                for k in stale_ips:
                    del self.history[k]
                self._last_cleanup = now

            self.history[ip] = [t for t in self.history[ip] if now - t < self.window_seconds]
            if len(self.history[ip]) >= self.requests_limit:
                return False
            self.history[ip].append(now)
            return True

limiter = RateLimiter(
    requests_limit=RATE_LIMIT_MAX_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    cleanup_interval=RATE_LIMIT_CLEANUP_INTERVAL,
)


# ── Security (API Key Authentication) ─────────────────────────────────────────
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
APP_ENV = os.getenv("APP_ENV", "development").lower()
IS_PRODUCTION = APP_ENV == "production"


def _get_expected_api_key() -> str | None:
    configured = os.getenv("API_KEY", "").strip()
    if configured:
        return configured
    if IS_PRODUCTION:
        logger.warning(
            "API_KEY is not set in production environment. "
            "All API requests will be rejected until API_KEY is configured."
        )
        return None
    return "hykero-secret-key"


EXPECTED_API_KEY = _get_expected_api_key()


def get_api_key(api_key: str = Security(api_key_header)) -> str | None:
    if not EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Server API_KEY not configured. Contact your administrator.",
        )
    if not api_key or not hmac.compare_digest(api_key, EXPECTED_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
    return api_key


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy resources on startup, clean up on shutdown."""
    logger.info("Initializing database...")
    try:
        from database import init_db
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)

    logger.info("Loading model artifacts...")
    try:
        from routes.predict import preload_model
        preload_model()
        logger.info("Model artifacts loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e} (will retry on first request)")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="HY Kero Flash Point Predictor API",
    description="Real-time Flash Point prediction for Heavy Kerosene using CDU sensor data",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").strip()
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]


CORS_ORIGINS = _parse_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


# ── Request logging & Rate limiting middleware ────────────────────────────────
@app.middleware("http")
async def process_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    health_paths = {"/api/health", "/api/health/live", "/api/health/ready"}

    if request.url.path.startswith("/api") and request.url.path not in health_paths:
        # Use X-Forwarded-For header when behind a reverse proxy (e.g. Nginx),
        # falling back to client.host for direct connections.
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )
        if not await limiter.check(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too Many Requests. Rate limit exceeded."}
            )

    # Body size guard for non-upload JSON endpoints
    content_length = request.headers.get("content-length")
    if content_length and "/upload/" not in request.url.path:
        try:
            if int(content_length) > 1_048_576:  # 1 MB
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large. Maximum allowed size is 1MB."}
                )
        except ValueError:
            pass

    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.0f}ms)")
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/api", tags=["Prediction"], dependencies=[Depends(get_api_key)])
app.include_router(history.router, prefix="/api", tags=["History"], dependencies=[Depends(get_api_key)])
app.include_router(models.router,  prefix="/api", tags=["Models"], dependencies=[Depends(get_api_key)])
app.include_router(upload.router,  prefix="/api", tags=["Upload"], dependencies=[Depends(get_api_key)])
app.include_router(chat.router,    prefix="/api", tags=["Chat"], dependencies=[Depends(get_api_key)])


@app.get("/")
def root():
    return {
        "message": "HY Kero Flash Point Predictor API",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/api/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/api/health/ready")
def readiness():
    """Lightweight readiness probe — no migrations, no heavy model loading."""
    try:
        from database import get_db_connection

        # Simple database ping — don't run init_db() on every probe
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()

        from routes.predict import _get_cached_model
        _, _, _, model_name = _get_cached_model()
        return {
            "status": "ready",
            "model_loaded": True,
            "model_name": model_name,
            "database_ready": True,
        }
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "model_loaded": False,
                "database_ready": False,
            }
        )


@app.get("/api/health")
def health():
    """Health check endpoint for monitoring / Docker."""
    try:
        from routes.predict import _get_cached_model
        _, _, _, model_name = _get_cached_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": model_name,
            "ready": True,
        }
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "model_loaded": False,
                "ready": False,
            }
        )
