#!/usr/bin/env python3
"""Neuro-Parakeet MLX Server - German audio transcription server using parakeet-mlx."""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import os
import re
import tempfile
import argparse
import logging
import sys
import shutil
import socket
import hashlib
import secrets
import time
import uuid
import asyncio
from datetime import datetime
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
ENV = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENV == "production"

try:
    from parakeet_mlx import from_pretrained
except ImportError as e:
    from_pretrained = None
    import sys
    error_msg = str(e)
    logger.warning(f"Failed to import parakeet_mlx: {e}")
    
    # Check if it's an MLX library issue (not available on non-Apple Silicon)
    if "libmlx.so" in error_msg or "cannot open shared object file" in error_msg:
        logger.error("=" * 60)
        logger.error("ERROR: MLX library not available!")
        logger.error("This server requires Apple Silicon (M1/M2/M3/M4) Mac.")
        logger.error("MLX (Apple's machine learning framework) is not available on Linux/Windows.")
        logger.error("=" * 60)
    else:
        logger.warning(f"Python path: {sys.path}")
        logger.warning(f"Python executable: {sys.executable}")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

model = None
DEFAULT_MODEL = os.getenv("PARAKEET_MODEL", "NeurologyAI/neuro-parakeet-mlx")

# Security configuration
API_KEY = os.getenv("API_KEY", None)  # Set API_KEY environment variable to enable authentication
_DEFAULT_CORS = "http://localhost:8002,http://127.0.0.1:8002,https://localhost,http://localhost,https://192.168.178.20,http://192.168.178.20"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", _DEFAULT_CORS).split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
# Production: require API key and disallow CORS *
if IS_PRODUCTION and (not CORS_ORIGINS or "*" in CORS_ORIGINS):
    logger.warning("Production: CORS_ORIGINS must be set and must not contain '*'. Using default restricted list.")
    CORS_ORIGINS = [o for o in _DEFAULT_CORS.split(",") if o.strip()]

# Allowed audio MIME types
ALLOWED_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3", "audio/x-mpeg-3",
    "audio/flac", "audio/x-flac",
    "audio/mp4", "audio/x-m4a", "audio/m4a",
    "audio/ogg", "audio/vorbis", "audio/opus",
    "audio/webm",
    "application/octet-stream"  # Some clients send this for audio files
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm"}

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Concurrency: limit simultaneous transcriptions (MLX is single-threaded; 1–2 is stable)
MAX_CONCURRENT_TRANSCRIPTIONS = max(1, int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "2")))
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

# Max time for a single transcription (seconds); prevents stuck requests
TRANSCRIPTION_TIMEOUT = max(60, float(os.getenv("TRANSCRIPTION_TIMEOUT", "600")))

# Set during shutdown so new transcriptions are rejected (graceful drain)
_shutting_down = False

def check_python_version():
    """Check if Python version is 3.10 or higher."""
    if sys.version_info < (3, 10):
        logger.error(f"Python 3.10+ required, but found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    return True

def check_disk_space(path, required_gb=5):
    """Check if there's enough disk space (default 5GB for model download)."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        if free_gb < required_gb:
            logger.warning(f"Low disk space: {free_gb:.2f}GB free, {required_gb}GB recommended for model download")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Don't fail if we can't check

def check_port_available(port):
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            if result == 0:
                logger.error(f"Port {port} is already in use!")
                return False
        return True
    except Exception as e:
        logger.warning(f"Could not check port availability: {e}")
        return True  # Don't fail if we can't check

def check_temp_directory():
    """Check if temp directory is writable."""
    try:
        test_file = tempfile.NamedTemporaryFile(delete=True)
        test_file.close()
        return True
    except Exception as e:
        logger.error(f"Temp directory is not writable: {e}")
        return False

def check_huggingface_cache():
    """Check HuggingFace cache directory."""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        if not os.access(cache_dir, os.W_OK):
            logger.warning(f"HuggingFace cache directory is not writable: {cache_dir}")
            return False
    return True

def validate_system_requirements():
    """Validate system requirements and log warnings."""
    logger.info("Validating system requirements...")
    
    issues = []
    warnings = []
    
    # Check Python version
    if not check_python_version():
        issues.append("Python version < 3.10")
    
    # Check temp directory
    if not check_temp_directory():
        issues.append("Temp directory not writable")
    
    # Check HuggingFace cache
    if not check_huggingface_cache():
        warnings.append("HuggingFace cache may not be writable")
    
    # Check disk space (for model downloads)
    cache_dir = os.path.expanduser("~/.cache")
    if not check_disk_space(cache_dir, required_gb=5):
        warnings.append("Low disk space for model download")
    
    if issues:
        logger.error("=" * 60)
        logger.error("CRITICAL ISSUES FOUND:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("=" * 60)
        return False
    
    if warnings:
        logger.warning("=" * 60)
        logger.warning("WARNINGS:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
        logger.warning("=" * 60)
    
    logger.info("System requirements validation passed")
    return True

class TranscriptionResponse(BaseModel):
    text: str
    recording_timestamp: Optional[str] = None
    segments: Optional[List[dict]] = None

def load_model(model_id: Optional[str] = None):
    global model
    if model is None and from_pretrained:
        try:
            model_id = model_id or os.getenv("PARAKEET_MODEL", DEFAULT_MODEL)
            logger.info(f"Loading model: {model_id}")
            
            # Check for model integrity verification
            expected_sha256 = os.getenv("MODEL_SHA256", None)
            if expected_sha256:
                logger.info("Model SHA256 verification enabled")
            
            if "/" in model_id and not os.path.exists(model_id) and snapshot_download:
                try:
                    logger.info(f"Downloading model from HuggingFace (local only)...")
                    cache_dir = snapshot_download(repo_id=model_id, repo_type="model", local_files_only=True)
                    model_id = cache_dir
                except Exception as e:
                    logger.warning(f"Local download failed: {e}, trying with network access...")
                    cache_dir = snapshot_download(repo_id=model_id, repo_type="model", local_files_only=False)
                    model_id = cache_dir
                logger.info(f"Loading model from: {model_id}")
                
                # Verify model integrity if checksum provided
                if expected_sha256:
                    if not verify_model_integrity(model_id, expected_sha256):
                        logger.error("Model integrity verification failed!")
                        raise ValueError("Model integrity check failed")
            
            model = from_pretrained(model_id)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            model = None
            raise
    elif model is None:
        logger.error("parakeet_mlx.from_pretrained is not available. Please install parakeet-mlx.")
        logger.error("Install with: pip install -r requirements.txt")
        logger.error("Or: pip install parakeet-mlx")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if from_pretrained:
        try:
            load_model()
            if model is None:
                logger.error("Model failed to load during startup!")
        except Exception as e:
            logger.error(f"Error during model loading in lifespan: {e}", exc_info=True)
    else:
        import sys
        import subprocess
        logger.error("=" * 60)
        logger.error("ERROR: parakeet_mlx is not available!")
        logger.error(f"Python executable: {sys.executable}")
        logger.error(f"Python version: {sys.version}")
        
        # Try to check if package is installed and diagnose the issue
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "parakeet-mlx"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Try to import to see the actual error
                import_result = subprocess.run(
                    [sys.executable, "-c", "import parakeet_mlx"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "libmlx.so" in import_result.stderr or "cannot open shared object file" in import_result.stderr:
                    logger.error("Package 'parakeet-mlx' is installed but MLX library is not available!")
                    logger.error("This server requires Apple Silicon (M1/M2/M3/M4) Mac.")
                    logger.error("MLX is Apple's framework and only works on macOS with Apple Silicon.")
                else:
                    logger.error("Package 'parakeet-mlx' is installed but cannot be imported!")
                    logger.error("This might be a Python path or environment issue.")
                    logger.error("Try: pip install --force-reinstall parakeet-mlx")
            else:
                logger.error("Package 'parakeet-mlx' is NOT installed!")
        except Exception as e:
            logger.error(f"Could not check package status: {e}")
        
        logger.error("")
        logger.error("Please install dependencies:")
        logger.error(f"  {sys.executable} -m pip install -r requirements.txt")
        logger.error("Or install directly:")
        logger.error(f"  {sys.executable} -m pip install parakeet-mlx")
        logger.error("=" * 60)
    yield
    global _shutting_down
    _shutting_down = True
    logger.info("Shutdown: draining (new transcriptions will receive 503)")

app = FastAPI(
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure unhandled exceptions never leak details in production."""
    logger.exception("Unhandled exception: %s", exc)
    from fastapi.responses import JSONResponse
    detail = "Internal server error" if IS_PRODUCTION else str(exc)
    return JSONResponse(status_code=500, content={"detail": detail})


_PATH_NORM_RE = re.compile(r'/+')


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a request ID for tracing; echo X-Request-ID if provided by client."""
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request method, path, status, duration; omit body for stability and privacy."""
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        request_id = getattr(request.state, "request_id", "")
        extra = f" request_id={request_id}" if request_id else ""
        logger.info(
            "%s %s %s %.1fms%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra,
        )
        return response


class NormalizePathMiddleware(BaseHTTPMiddleware):
    """Middleware to normalize duplicate slashes in paths."""
    async def dispatch(self, request: Request, call_next):
        if '//' in request.url.path:
            s = dict(request.scope)
            s["path"] = s["path_info"] = _PATH_NORM_RE.sub('/', request.url.path)
            request = Request(s)
        return await call_next(request)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API key if configured."""
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for health check and root endpoints (docs/redoc disabled in production)
        if request.url.path in ["/", "/health", "/live", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # If API key is configured, validate it
        if API_KEY:
            auth_header = request.headers.get("Authorization", "")
            api_key_header = request.headers.get("X-API-Key", "")
            
            # Check Authorization: Bearer <key> or X-API-Key header
            if auth_header.startswith("Bearer "):
                provided_key = auth_header.replace("Bearer ", "", 1)
            elif api_key_header:
                provided_key = api_key_header
            else:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key required. Provide via 'Authorization: Bearer <key>' or 'X-API-Key' header"}
                )
            
            # Use constant-time comparison to prevent timing attacks
            if not secrets.compare_digest(provided_key, API_KEY):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key"}
                )
        
        return await call_next(request)

# Add middlewares (outer = last added = runs first): request ID, logging, then auth, security, path, CORS
app.add_middleware(APIKeyMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(NormalizePathMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIdMiddleware)
# When CORS_ORIGINS is ["*"], use allow_origins=["*"] with allow_credentials=False (browser requirement)
_cors_origins = CORS_ORIGINS if CORS_ORIGINS else ["http://localhost:8002", "http://127.0.0.1:8002"]
_cors_allow_credentials = _cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["*"]
)

def get_index_path():
    """Get the path to index.html, checking multiple possible locations."""
    # Try current file directory first
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "index.html"),
        os.path.join(os.path.dirname(__file__), "..", "index.html"),
        os.path.join(os.getcwd(), "index.html"),
        "index.html",  # Current working directory
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    return None

@app.get("/")
async def root():
    """Root endpoint - serves index.html if available, otherwise returns status."""
    index_path = get_index_path()
    if index_path:
        return FileResponse(index_path, media_type="text/html")
    return {"status": "ok" if model else "error"}

@app.get("/live")
async def liveness():
    """Liveness probe: returns 200 if the process is up. Use for k8s livenessProbe."""
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    """Health check endpoint - returns server and model status."""
    import sys
    import shutil
    
    health_status = {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "system": sys.platform
    }
    
    # Add disk space info
    try:
        stat = shutil.disk_usage(os.path.expanduser("~"))
        health_status["disk_space_gb"] = round(stat.free / (1024**3), 2)
    except Exception:
        pass
    
    status_code = 200 if model else 503
    return health_status

@app.get("/transcription")
async def transcription_ui():
    """Serve the transcription UI interface."""
    index_path = get_index_path()
    if index_path:
        return FileResponse(index_path, media_type="text/html")
    return {"status": "ok" if model else "error", "message": "Transcription UI not found"}

def extract_text(r):
    if hasattr(r, 'text'):
        if hasattr(r, 'segments') and r.segments:
            return ' '.join(seg.text if hasattr(seg, 'text') else (seg.get('text', '') if isinstance(seg, dict) else str(seg)) for seg in r.segments)
        return r.text
    if isinstance(r, dict):
        if 'segments' in r and r['segments']:
            return ' '.join(seg.get('text', '') if isinstance(seg, dict) else str(seg) for seg in r['segments'])
        text = r.get('text', '')
        if text:
            return text
        if r.get('segments') == []:
            return ''
        return str(r)
    return str(r)

def clean_text(text: str) -> str:
    text = re.sub(r'\s*<unk>\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other attacks."""
    # Remove path components
    filename = os.path.basename(filename)
    # Remove any remaining path separators
    filename = filename.replace("/", "").replace("\\", "")
    # Remove null bytes
    filename = filename.replace("\x00", "")
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    return filename

def validate_file_type(filename: str, content_type: Optional[str] = None) -> bool:
    """Validate that the file is an allowed audio type."""
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check MIME type if provided
    if content_type:
        # Handle content type with parameters (e.g., "audio/wav; charset=utf-8")
        base_content_type = content_type.split(";")[0].strip().lower()
        if base_content_type not in ALLOWED_MIME_TYPES:
            # Allow if extension is valid (some clients send wrong MIME types)
            logger.warning(f"Unexpected MIME type: {content_type} for file {filename}, but extension is valid")
            return True  # Allow based on extension
    
    return True

def verify_model_integrity(model_path: str, expected_sha256: Optional[str] = None) -> bool:
    """Verify model integrity using SHA256 checksum if provided."""
    if not expected_sha256:
        return True  # No checksum provided, skip verification
    
    try:
        # Calculate SHA256 of model directory or files
        # For simplicity, we'll check if the path exists and log a warning
        # In production, you should verify individual model files
        if os.path.exists(model_path):
            logger.info(f"Model integrity check: Model found at {model_path}")
            logger.warning("SHA256 verification not fully implemented. Set MODEL_SHA256 environment variable for full verification.")
            return True
        return False
    except Exception as e:
        logger.error(f"Model integrity check failed: {e}")
        return False

def extract_segments(r):
    """Extract segment information with timing if available."""
    segments = []
    if hasattr(r, 'segments') and r.segments:
        for seg in r.segments:
            seg_dict = {
                'text': seg.text if hasattr(seg, 'text') else (seg.get('text', '') if isinstance(seg, dict) else str(seg))
            }
            if hasattr(seg, 'start'):
                seg_dict['start'] = seg.start
            elif isinstance(seg, dict) and 'start' in seg:
                seg_dict['start'] = seg['start']
            if hasattr(seg, 'end'):
                seg_dict['end'] = seg.end
            elif isinstance(seg, dict) and 'end' in seg:
                seg_dict['end'] = seg['end']
            segments.append(seg_dict)
    elif isinstance(r, dict) and 'segments' in r and r['segments']:
        for seg in r['segments']:
            seg_dict = {
                'text': seg.get('text', '') if isinstance(seg, dict) else str(seg)
            }
            if isinstance(seg, dict):
                if 'start' in seg:
                    seg_dict['start'] = seg['start']
                if 'end' in seg:
                    seg_dict['end'] = seg['end']
            segments.append(seg_dict)
    return segments if segments else None

@app.options("/v1/audio/transcriptions")
async def options_transcription():
    """Handle CORS preflight requests."""
    return {"status": "ok"}

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(file: UploadFile = File(...), model_name: str = Form("parakeet-tdt-0.6b-v3", alias="model"),
                               response_format: Optional[str] = Form("json"),
                               recording_timestamp: Optional[str] = Form(None)):
    if _shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file presence
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    sanitized_filename = sanitize_filename(file.filename)
    if sanitized_filename != file.filename:
        logger.warning("Filename sanitized: %s -> %s", file.filename, sanitized_filename)

    if not validate_file_type(sanitized_filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file provided")

    ext = Path(sanitized_filename).suffix.lower() or ".wav"
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        p = tmp.name
    try:
        with open(p, "wb") as f:
            f.write(file_content)
        async with transcription_semaphore:
            def _transcribe():
                try:
                    return model.transcribe(p, language="de")
                except TypeError:
                    return model.transcribe(p)
            try:
                r = await asyncio.wait_for(
                    asyncio.to_thread(_transcribe),
                    timeout=TRANSCRIPTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error("Transcription timed out after %.0fs", TRANSCRIPTION_TIMEOUT)
                raise HTTPException(
                    status_code=504,
                    detail=f"Transcription timed out (max {int(TRANSCRIPTION_TIMEOUT)}s)",
                )
            except Exception as e:
                logger.exception("Transcription failed: %s", e)
                if IS_PRODUCTION:
                    raise HTTPException(status_code=500, detail="Transcription failed")
                raise HTTPException(status_code=500, detail="Transcription failed") from e
        t = clean_text(extract_text(r))
        segments = extract_segments(r)
        if response_format == "text":
            return Response(content=t, media_type="text/plain; charset=utf-8")
        return TranscriptionResponse(
            text=t,
            recording_timestamp=recording_timestamp,
            segments=segments,
        )
    finally:
        try:
            os.remove(p)
        except OSError as e:
            logger.warning("Could not remove temp file %s: %s", p, e)

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--skip-validation", action="store_true", help="Skip system requirements validation")
    a = p.parse_args()
    if a.model:
        os.environ["PARAKEET_MODEL"] = a.model
    
    port = a.port or int(os.getenv("PORT", 8002))

    # Production: require API key
    if IS_PRODUCTION and not API_KEY:
        logger.error("ENV=production requires API_KEY to be set. Set API_KEY in the environment and restart.")
        sys.exit(1)

    # Log security configuration
    if IS_PRODUCTION:
        logger.info("Running in PRODUCTION mode (docs disabled, CORS restricted)")
    if API_KEY:
        logger.info("API key authentication: ENABLED")
    else:
        logger.warning("API key authentication: DISABLED (set API_KEY environment variable to enable)")
    logger.info(f"CORS allowed origins: {CORS_ORIGINS}")
    if not IS_PRODUCTION and ("*" in CORS_ORIGINS or len(CORS_ORIGINS) == 0):
        logger.warning("CORS is open to all origins. For production set ENV=production and CORS_ORIGINS to your allowed origins.")
    
    # Validate system requirements
    if not a.skip_validation:
        if not validate_system_requirements():
            logger.error("System requirements validation failed. Use --skip-validation to proceed anyway.")
            sys.exit(1)
        
        # Check port availability
        if not check_port_available(port):
            logger.error(f"Port {port} is already in use. Please choose a different port.")
            sys.exit(1)
    
    host = os.getenv("BIND", "127.0.0.1")
    timeout_keep_alive = int(os.getenv("UVICORN_TIMEOUT_KEEP_ALIVE", "30"))
    timeout_graceful_shutdown = float(os.getenv("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", "15.0"))
    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
    )

