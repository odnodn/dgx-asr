#!/usr/bin/env python3
"""Neuro-Parakeet MLX Server with Speaker Diarization.

This server extends the parakeet-mlx-server with speaker diarization capabilities.
It reuses parakeet_server.py for base transcription and adds diarization endpoints
using pyannote.audio or NeMo SortedFormer strategies.

Usage:
    python parakeet_server_diarization.py --port 8003
    python parakeet_server_diarization.py --diarization-strategy pyannote
    python parakeet_server_diarization.py --diarization-strategy sortedformer
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import os
import re
import tempfile
import argparse
import logging
import sys
import shutil
import secrets
import time
import uuid
import asyncio
from pathlib import Path

# Import base server utilities
from parakeet_server import (
    clean_text,
    extract_text,
    extract_segments,
    sanitize_filename,
    validate_file_type,
    load_model,
    check_python_version,
    validate_system_requirements,
    check_port_available,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

ENV = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENV == "production"

# Diarization configuration
DIARIZATION_STRATEGY = os.getenv("DIARIZATION_STRATEGY", "pyannote")  # pyannote or sortedformer
DEFAULT_NUM_SPEAKERS = int(os.getenv("DEFAULT_NUM_SPEAKERS", "2"))
PYANNOTE_AUTH_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN", None)

# Default speaker names by number of speakers
DEFAULT_SPEAKER_NAMES: Dict[int, List[str]] = {
    2: ["Arzt", "Patient"],
    3: ["Arzt", "Patient", "Angehöriger"],
    4: ["Arzt", "Patient", "Angehöriger", "Begleitung"],
}

# Medical vocabulary for enhanced recognition
MEDICAL_VOCABULARY_PATH = os.getenv("MEDICAL_VOCABULARY_PATH", None)

# Security
API_KEY = os.getenv("API_KEY", None)
_DEFAULT_CORS = "http://localhost:8003,http://127.0.0.1:8003,https://localhost,http://localhost"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", _DEFAULT_CORS).split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# Concurrency
MAX_CONCURRENT_TRANSCRIPTIONS = max(1, int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "2")))
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
TRANSCRIPTION_TIMEOUT = max(60, float(os.getenv("TRANSCRIPTION_TIMEOUT", "900")))

_shutting_down = False

# ─── Diarization backends ───────────────────────────────────────────────────────

diarization_pipeline = None


def load_pyannote_pipeline():
    """Load pyannote.audio diarization pipeline."""
    global diarization_pipeline
    try:
        from pyannote.audio import Pipeline

        model_name = os.getenv(
            "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"
        )
        logger.info(f"Loading pyannote pipeline: {model_name}")

        kwargs = {}
        if PYANNOTE_AUTH_TOKEN:
            kwargs["use_auth_token"] = PYANNOTE_AUTH_TOKEN

        diarization_pipeline = Pipeline.from_pretrained(model_name, **kwargs)

        # Move to GPU if available
        try:
            import torch

            if torch.cuda.is_available():
                diarization_pipeline.to(torch.device("cuda"))
                logger.info("Pyannote pipeline moved to CUDA")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                diarization_pipeline.to(torch.device("mps"))
                logger.info("Pyannote pipeline moved to MPS (Apple Silicon)")
        except Exception as e:
            logger.warning(f"Could not move pipeline to GPU: {e}")

        logger.info("Pyannote pipeline loaded successfully")
    except ImportError:
        logger.error("pyannote.audio not installed. Install with: pip install pyannote.audio")
        diarization_pipeline = None
    except Exception as e:
        logger.error(f"Failed to load pyannote pipeline: {e}", exc_info=True)
        diarization_pipeline = None


def load_sortedformer_pipeline():
    """Load NeMo SortedFormer diarization pipeline (placeholder for NeMo-based approach)."""
    global diarization_pipeline
    try:
        # SortedFormer requires NeMo toolkit
        from nemo.collections.asr.models import SortedFormerModel

        model_name = os.getenv(
            "SORTEDFORMER_MODEL", "nvidia/sortedformer_diar_base"
        )
        logger.info(f"Loading SortedFormer model: {model_name}")
        diarization_pipeline = SortedFormerModel.from_pretrained(model_name)
        logger.info("SortedFormer model loaded successfully")
    except ImportError:
        logger.warning(
            "NeMo toolkit not available for SortedFormer. "
            "Install with: pip install nemo_toolkit[asr]"
        )
        diarization_pipeline = None
    except Exception as e:
        logger.error(f"Failed to load SortedFormer: {e}", exc_info=True)
        diarization_pipeline = None


def run_pyannote_diarization(audio_path: str, num_speakers: int) -> List[Dict]:
    """Run pyannote diarization on an audio file.

    Returns a list of segments with speaker labels and timestamps.
    """
    if diarization_pipeline is None:
        raise RuntimeError("Pyannote pipeline not loaded")

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    diarization_result = diarization_pipeline(audio_path, **kwargs)

    segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
        )

    return segments


def run_sortedformer_diarization(audio_path: str, num_speakers: int) -> List[Dict]:
    """Run SortedFormer diarization on an audio file.

    Returns a list of segments with speaker labels and timestamps.
    """
    if diarization_pipeline is None:
        raise RuntimeError("SortedFormer model not loaded")

    # NeMo SortedFormer API
    result = diarization_pipeline.diarize(
        audio_path, num_speakers=num_speakers
    )

    segments = []
    if hasattr(result, "segments"):
        for seg in result.segments:
            segments.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "speaker": seg.speaker,
                }
            )
    elif isinstance(result, list):
        for seg in result:
            if isinstance(seg, dict):
                segments.append(
                    {
                        "start": round(seg.get("start", 0), 3),
                        "end": round(seg.get("end", 0), 3),
                        "speaker": seg.get("speaker", "UNKNOWN"),
                    }
                )

    return segments


def assign_speaker_names(
    diarization_segments: List[Dict], speaker_names: Optional[List[str]] = None
) -> List[Dict]:
    """Map generic speaker labels (SPEAKER_00, etc.) to provided names."""
    # Collect unique speakers in order of appearance
    seen = {}
    for seg in diarization_segments:
        spk = seg["speaker"]
        if spk not in seen:
            seen[spk] = len(seen)

    # Build mapping
    name_map = {}
    for spk, idx in seen.items():
        if speaker_names and idx < len(speaker_names):
            name_map[spk] = speaker_names[idx]
        else:
            name_map[spk] = f"Speaker {idx + 1}"

    # Apply mapping
    for seg in diarization_segments:
        seg["speaker"] = name_map[seg["speaker"]]

    return diarization_segments


def merge_transcription_with_diarization(
    transcription_segments: Optional[List[Dict]],
    diarization_segments: List[Dict],
    full_text: str,
) -> List[Dict]:
    """Merge ASR transcription segments with diarization speaker labels.

    Uses temporal overlap to assign speaker labels to transcription segments.
    """
    if not transcription_segments:
        # If no segment timing from ASR, return diarization segments with full text split
        if not diarization_segments:
            return [{"speaker": "Speaker 1", "text": full_text, "start": 0.0, "end": 0.0}]
        # Assign full text proportionally
        total_duration = max(seg["end"] for seg in diarization_segments) if diarization_segments else 1.0
        words = full_text.split()
        total_words = len(words)
        result = []
        word_idx = 0
        for seg in diarization_segments:
            seg_duration = seg["end"] - seg["start"]
            n_words = max(1, int(round(total_words * seg_duration / total_duration)))
            seg_words = words[word_idx : word_idx + n_words]
            word_idx += n_words
            result.append(
                {
                    "speaker": seg["speaker"],
                    "text": " ".join(seg_words),
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )
        # Assign remaining words to last segment
        if word_idx < total_words and result:
            result[-1]["text"] += " " + " ".join(words[word_idx:])
        return result

    # Match transcription segments to diarization segments by temporal overlap
    result = []
    for t_seg in transcription_segments:
        t_start = t_seg.get("start", 0.0)
        t_end = t_seg.get("end", 0.0)
        t_text = t_seg.get("text", "")

        # Find the diarization segment with maximum overlap
        best_speaker = "Unknown"
        best_overlap = 0.0
        for d_seg in diarization_segments:
            overlap_start = max(t_start, d_seg["start"])
            overlap_end = min(t_end, d_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        result.append(
            {
                "speaker": best_speaker,
                "text": t_text,
                "start": t_start,
                "end": t_end,
            }
        )

    # Merge consecutive segments from the same speaker
    merged = []
    for seg in result:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    return merged


# ─── Medical Vocabulary ─────────────────────────────────────────────────────────

medical_vocabulary: Optional[List[str]] = None


def load_medical_vocabulary(path: Optional[str] = None) -> Optional[List[str]]:
    """Load medical vocabulary from a file (one term per line).

    The vocabulary can be used for post-processing corrections and boosting
    recognition of domain-specific terms.
    """
    global medical_vocabulary
    vocab_path = path or MEDICAL_VOCABULARY_PATH
    if not vocab_path:
        return None

    if not os.path.exists(vocab_path):
        logger.warning(f"Medical vocabulary file not found: {vocab_path}")
        return None

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            terms = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        medical_vocabulary = terms
        logger.info(f"Loaded {len(terms)} medical vocabulary terms from {vocab_path}")
        return terms
    except Exception as e:
        logger.error(f"Failed to load medical vocabulary: {e}")
        return None


def apply_vocabulary_corrections(text: str, vocabulary: Optional[List[str]] = None) -> str:
    """Apply vocabulary-based post-processing corrections to transcribed text.

    This performs case-insensitive matching and replaces recognized terms
    with their canonical form from the vocabulary.
    """
    vocab = vocabulary or medical_vocabulary
    if not vocab:
        return text

    # Build a lookup for case-insensitive matching
    for term in vocab:
        # Replace case-insensitive occurrences with the canonical form
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(term, text)

    return text


# ─── Pydantic Models ────────────────────────────────────────────────────────────


class DiarizationSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float


class DiarizationResponse(BaseModel):
    text: str
    segments: List[DiarizationSegment]
    speakers: List[str]
    num_speakers: int
    recording_timestamp: Optional[str] = None
    diarization_strategy: str


class DiarizationConfig(BaseModel):
    """Configuration for diarization requests."""

    num_speakers: int = Field(default=2, ge=1, le=20, description="Number of speakers")
    speaker_names: Optional[List[str]] = Field(
        default=None, description="Custom speaker names (e.g., ['Arzt', 'Patient'])"
    )
    diarization_strategy: Optional[str] = Field(
        default=None, description="Diarization strategy: 'pyannote' or 'sortedformer'"
    )


# ─── Application Setup ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    # Load ASR model
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}", exc_info=True)

    # Load diarization pipeline
    strategy = DIARIZATION_STRATEGY.lower()
    if strategy == "pyannote":
        load_pyannote_pipeline()
    elif strategy == "sortedformer":
        load_sortedformer_pipeline()
    else:
        logger.error(f"Unknown diarization strategy: {strategy}. Use 'pyannote' or 'sortedformer'.")

    # Load medical vocabulary
    load_medical_vocabulary()

    yield
    global _shutting_down
    _shutting_down = True
    logger.info("Shutdown: draining requests")


app = FastAPI(
    title="Neuro-Parakeet MLX Diarization Server",
    description="German audio transcription with speaker diarization",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)


# ─── Middlewares ────────────────────────────────────────────────────────────────


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key if configured."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ["/", "/health", "/live", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        if API_KEY:
            auth_header = request.headers.get("Authorization", "")
            api_key_header = request.headers.get("X-API-Key", "")

            if auth_header.startswith("Bearer "):
                provided_key = auth_header.replace("Bearer ", "", 1)
            elif api_key_header:
                provided_key = api_key_header
            else:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key required"},
                )

            if not secrets.compare_digest(provided_key, API_KEY):
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        return await call_next(request)


app.add_middleware(APIKeyMiddleware)

_cors_origins = CORS_ORIGINS if CORS_ORIGINS else ["http://localhost:8003"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["*"],
)


# ─── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Root endpoint."""
    from parakeet_server import model

    return {
        "service": "neuro-parakeet-diarization",
        "status": "ok" if model else "no_model",
        "diarization_strategy": DIARIZATION_STRATEGY,
        "diarization_loaded": diarization_pipeline is not None,
    }


@app.get("/health")
async def health_check():
    """Health check with diarization status."""
    from parakeet_server import model

    return {
        "status": "healthy" if model and diarization_pipeline else "degraded",
        "model_loaded": model is not None,
        "diarization_loaded": diarization_pipeline is not None,
        "diarization_strategy": DIARIZATION_STRATEGY,
        "medical_vocabulary_loaded": medical_vocabulary is not None,
        "medical_vocabulary_terms": len(medical_vocabulary) if medical_vocabulary else 0,
    }


@app.get("/live")
async def liveness():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions", response_model=DiarizationResponse)
async def create_transcription_with_diarization(
    file: UploadFile = File(...),
    model_name: str = Form("parakeet-tdt-0.6b-v3", alias="model"),
    response_format: Optional[str] = Form("json"),
    recording_timestamp: Optional[str] = Form(None),
    num_speakers: int = Form(DEFAULT_NUM_SPEAKERS),
    speaker_names: Optional[str] = Form(None),
    diarization_strategy: Optional[str] = Form(None),
):
    """OpenAI-compatible transcription endpoint with speaker diarization.

    Extends the standard /v1/audio/transcriptions endpoint with diarization.

    Additional form fields:
    - num_speakers: Number of speakers (default: 2)
    - speaker_names: Comma-separated speaker names (default: 'Arzt,Patient')
    - diarization_strategy: Override default strategy ('pyannote' or 'sortedformer')
    """
    from parakeet_server import model

    if _shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    if not model:
        raise HTTPException(status_code=503, detail="ASR model not loaded")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    sanitized_filename = sanitize_filename(file.filename)
    if not validate_file_type(sanitized_filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB",
        )
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file provided")

    # Parse speaker names
    names = None
    if speaker_names:
        names = [n.strip() for n in speaker_names.split(",") if n.strip()]
    if not names:
        names = DEFAULT_SPEAKER_NAMES.get(num_speakers, [f"Speaker {i+1}" for i in range(num_speakers)])

    # Determine strategy
    strategy = (diarization_strategy or DIARIZATION_STRATEGY).lower()

    ext = Path(sanitized_filename).suffix.lower() or ".wav"
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        audio_path = tmp.name

    try:
        with open(audio_path, "wb") as f:
            f.write(file_content)

        async with transcription_semaphore:
            # Run ASR transcription
            def _transcribe():
                try:
                    return model.transcribe(audio_path, language="de")
                except TypeError:
                    return model.transcribe(audio_path)

            # Run diarization
            def _diarize():
                if strategy == "pyannote":
                    return run_pyannote_diarization(audio_path, num_speakers)
                elif strategy == "sortedformer":
                    return run_sortedformer_diarization(audio_path, num_speakers)
                else:
                    raise ValueError(f"Unknown diarization strategy: {strategy}")

            try:
                # Run both in parallel
                transcription_future = asyncio.to_thread(_transcribe)
                diarization_future = asyncio.to_thread(_diarize)

                r, diar_segments = await asyncio.wait_for(
                    asyncio.gather(transcription_future, diarization_future),
                    timeout=TRANSCRIPTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error("Transcription/diarization timed out after %.0fs", TRANSCRIPTION_TIMEOUT)
                raise HTTPException(
                    status_code=504,
                    detail=f"Processing timed out (max {int(TRANSCRIPTION_TIMEOUT)}s)",
                )
            except RuntimeError as e:
                if "not loaded" in str(e).lower():
                    raise HTTPException(
                        status_code=503,
                        detail=f"Diarization pipeline not loaded ({strategy})",
                    )
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.exception("Transcription/diarization failed: %s", e)
                raise HTTPException(status_code=500, detail="Processing failed")

        # Process results
        full_text = clean_text(extract_text(r))
        asr_segments = extract_segments(r)

        # Apply medical vocabulary corrections
        if medical_vocabulary:
            full_text = apply_vocabulary_corrections(full_text)
            if asr_segments:
                for seg in asr_segments:
                    seg["text"] = apply_vocabulary_corrections(seg.get("text", ""))

        # Assign speaker names to diarization segments
        diar_segments = assign_speaker_names(diar_segments, names)

        # Merge transcription with diarization
        merged_segments = merge_transcription_with_diarization(asr_segments, diar_segments, full_text)

        # Build response
        speakers = list(dict.fromkeys(seg["speaker"] for seg in merged_segments))

        if response_format == "text":
            # Format as speaker-labeled text
            lines = [f"{seg['speaker']}: {seg['text']}" for seg in merged_segments if seg["text"].strip()]
            return Response(content="\n".join(lines), media_type="text/plain; charset=utf-8")

        return DiarizationResponse(
            text=full_text,
            segments=[DiarizationSegment(**seg) for seg in merged_segments],
            speakers=speakers,
            num_speakers=len(speakers),
            recording_timestamp=recording_timestamp,
            diarization_strategy=strategy,
        )

    finally:
        try:
            os.remove(audio_path)
        except OSError as e:
            logger.warning("Could not remove temp file %s: %s", audio_path, e)


@app.post("/v1/audio/diarize")
async def diarize_only(
    file: UploadFile = File(...),
    num_speakers: int = Form(DEFAULT_NUM_SPEAKERS),
    speaker_names: Optional[str] = Form(None),
    diarization_strategy: Optional[str] = Form(None),
):
    """Diarization-only endpoint (no transcription).

    Returns speaker segments with timestamps but without transcription text.
    """
    if _shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down")
    if diarization_pipeline is None:
        raise HTTPException(status_code=503, detail="Diarization pipeline not loaded")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    sanitized_filename = sanitize_filename(file.filename)
    if not validate_file_type(sanitized_filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB",
        )
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file provided")

    # Parse speaker names
    names = None
    if speaker_names:
        names = [n.strip() for n in speaker_names.split(",") if n.strip()]
    if not names:
        names = DEFAULT_SPEAKER_NAMES.get(num_speakers, [f"Speaker {i+1}" for i in range(num_speakers)])

    strategy = (diarization_strategy or DIARIZATION_STRATEGY).lower()

    ext = Path(sanitized_filename).suffix.lower() or ".wav"
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        audio_path = tmp.name

    try:
        with open(audio_path, "wb") as f:
            f.write(file_content)

        async with transcription_semaphore:
            def _diarize():
                if strategy == "pyannote":
                    return run_pyannote_diarization(audio_path, num_speakers)
                elif strategy == "sortedformer":
                    return run_sortedformer_diarization(audio_path, num_speakers)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

            try:
                diar_segments = await asyncio.wait_for(
                    asyncio.to_thread(_diarize),
                    timeout=TRANSCRIPTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Diarization timed out")
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                logger.exception("Diarization failed: %s", e)
                raise HTTPException(status_code=500, detail="Diarization failed")

        diar_segments = assign_speaker_names(diar_segments, names)
        speakers = list(dict.fromkeys(seg["speaker"] for seg in diar_segments))

        return {
            "segments": diar_segments,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "diarization_strategy": strategy,
        }

    finally:
        try:
            os.remove(audio_path)
        except OSError as e:
            logger.warning("Could not remove temp file %s: %s", audio_path, e)


@app.get("/v1/diarization/config")
async def get_diarization_config():
    """Get current diarization configuration."""
    return {
        "diarization_strategy": DIARIZATION_STRATEGY,
        "default_num_speakers": DEFAULT_NUM_SPEAKERS,
        "default_speaker_names": DEFAULT_SPEAKER_NAMES,
        "diarization_loaded": diarization_pipeline is not None,
        "medical_vocabulary_loaded": medical_vocabulary is not None,
        "medical_vocabulary_terms": len(medical_vocabulary) if medical_vocabulary else 0,
    }


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description="Parakeet MLX Server with Speaker Diarization")
    p.add_argument("--model", type=str, default=None, help="ASR model ID")
    p.add_argument("--port", type=int, default=None, help="Server port (default: 8003)")
    p.add_argument(
        "--diarization-strategy",
        type=str,
        default=None,
        choices=["pyannote", "sortedformer"],
        help="Diarization strategy",
    )
    p.add_argument("--num-speakers", type=int, default=None, help="Default number of speakers")
    p.add_argument("--vocabulary", type=str, default=None, help="Path to medical vocabulary file")
    p.add_argument("--skip-validation", action="store_true", help="Skip system validation")
    a = p.parse_args()

    if a.model:
        os.environ["PARAKEET_MODEL"] = a.model
    if a.diarization_strategy:
        DIARIZATION_STRATEGY = a.diarization_strategy
        os.environ["DIARIZATION_STRATEGY"] = a.diarization_strategy
    if a.num_speakers:
        DEFAULT_NUM_SPEAKERS = a.num_speakers
    if a.vocabulary:
        os.environ["MEDICAL_VOCABULARY_PATH"] = a.vocabulary

    port = a.port or int(os.getenv("PORT", 8003))

    if IS_PRODUCTION and not API_KEY:
        logger.error("ENV=production requires API_KEY. Set API_KEY and restart.")
        sys.exit(1)

    if not a.skip_validation:
        if not validate_system_requirements():
            logger.error("System requirements validation failed. Use --skip-validation to proceed.")
            sys.exit(1)
        if not check_port_available(port):
            logger.error(f"Port {port} is already in use.")
            sys.exit(1)

    logger.info(f"Starting diarization server on port {port}")
    logger.info(f"Diarization strategy: {DIARIZATION_STRATEGY}")
    logger.info(f"Default speakers: {DEFAULT_NUM_SPEAKERS}")

    host = os.getenv("BIND", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)
