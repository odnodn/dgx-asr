# Deployment: Parallel Execution and Hardware Configuration

This document covers running the diarization server with different hardware backends (MLX, GPU, CPU) and configuring parallel vs. sequential execution of ASR and diarization pipelines.

---

## Table of Contents

- [Execution Modes](#execution-modes)
  - [Parallel Execution (Default)](#parallel-execution-default)
  - [Sequential Execution](#sequential-execution)
- [Hardware Backends](#hardware-backends)
  - [Apple Silicon with MLX](#apple-silicon-with-mlx)
  - [NVIDIA GPU with CUDA](#nvidia-gpu-with-cuda)
  - [CPU-Only Mode](#cpu-only-mode)
- [Resource Allocation](#resource-allocation)
  - [Concurrency Control](#concurrency-control)
  - [Memory Considerations](#memory-considerations)
  - [Timeout Configuration](#timeout-configuration)
- [Deployment Scenarios](#deployment-scenarios)
  - [Single Mac Mini (MLX + CPU Diarization)](#single-mac-mini-mlx--cpu-diarization)
  - [NVIDIA DGX / GPU Server](#nvidia-dgx--gpu-server)
  - [CPU-Only Server](#cpu-only-server)
  - [Hybrid: MLX ASR + GPU Diarization (Separate Machines)](#hybrid-mlx-asr--gpu-diarization-separate-machines)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

---

## Execution Modes

### Parallel Execution (Default)

By default, ASR transcription and speaker diarization run **in parallel** using `asyncio.gather()`:

```python
# Internal implementation (parakeet_server_diarization.py)
transcription_future = asyncio.to_thread(_transcribe)
diarization_future = asyncio.to_thread(_diarize)

r, diar_segments = await asyncio.gather(transcription_future, diarization_future)
```

**How it works:**
1. Both tasks are dispatched to separate threads via `asyncio.to_thread()`.
2. `asyncio.gather()` waits for both to complete.
3. Results are merged using temporal overlap matching.

**Advantages:**
- Faster total processing time (wall clock)
- ASR and diarization use different compute resources in many configurations
- Optimal for MLX (ASR) + CPU/GPU (diarization) setups

**When parallel works best:**
- ASR runs on MLX (Apple Neural Engine / GPU)
- Diarization runs on CPU or a separate GPU
- Sufficient memory for both models simultaneously

### Sequential Execution

To run ASR and diarization sequentially (one after the other), use the **diarize-only endpoint** combined with the base transcription server:

```bash
# Step 1: Transcribe with the base server
curl -X POST http://localhost:8002/v1/audio/transcriptions \
  -F file=@recording.wav

# Step 2: Diarize separately
curl -X POST http://localhost:8003/v1/audio/diarize \
  -F file=@recording.wav \
  -F num_speakers=2
```

**Alternatively**, reduce concurrency to force sequential behavior within the diarization server:

```bash
# Limit to 1 concurrent transcription (effectively sequential per request)
export MAX_CONCURRENT_TRANSCRIPTIONS=1
python parakeet_server_diarization.py --port 8003
```

> **Note:** Within a single request, ASR and diarization always run in parallel. Sequential mode applies only at the request level (one request at a time).

**When to use sequential/separate execution:**
- Limited memory (cannot load both models simultaneously)
- Debugging (isolate ASR from diarization issues)
- When diarization is optional and not needed for every request

---

## Hardware Backends

### Apple Silicon with MLX

The primary deployment target. ASR runs on MLX (Apple's machine learning framework), while diarization uses PyTorch on CPU or MPS.

```bash
# Start server on Apple Silicon
python parakeet_server_diarization.py --port 8003 --diarization-strategy pyannote
```

**Resource split:**

| Component | Backend | Hardware |
|-----------|---------|----------|
| ASR (Parakeet TDT) | MLX | Apple Neural Engine + GPU |
| Diarization (pyannote) | PyTorch | MPS (Metal) or CPU |
| Audio preprocessing | librosa/scipy | CPU |

**MLX advantages on Apple Silicon:**
- Unified memory — no CPU↔GPU copy overhead
- Native Neural Engine acceleration for transformer models
- Low power consumption for continuous operation

**Pyannote on MPS:**
The diarization server automatically detects MPS availability:
```python
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    diarization_pipeline.to(torch.device("mps"))
```

**Caveats:**
- MPS support for pyannote may be incomplete for some operations
- If MPS fails, the pipeline falls back to CPU automatically
- MLX and MPS share the unified memory pool — monitor total usage

### NVIDIA GPU with CUDA

For NVIDIA GPU servers (DGX, A100, RTX, etc.), both ASR and diarization can run on GPU:

```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Start server
export PYANNOTE_AUTH_TOKEN="hf_..."
python parakeet_server_diarization.py --port 8003
```

**Resource split:**

| Component | Backend | Hardware |
|-----------|---------|----------|
| ASR (Parakeet TDT) | MLX or NeMo | GPU (if NeMo) or CPU |
| Diarization (pyannote) | PyTorch CUDA | GPU |
| Audio preprocessing | librosa/scipy | CPU |

**CUDA auto-detection:**
```python
if torch.cuda.is_available():
    diarization_pipeline.to(torch.device("cuda"))
```

**Multi-GPU considerations:**
- Pyannote uses a single GPU by default
- For multi-GPU, set `CUDA_VISIBLE_DEVICES` to control allocation:

```bash
# ASR on GPU 0, diarization on GPU 1
CUDA_VISIBLE_DEVICES=1 python parakeet_server_diarization.py --port 8003
```

### CPU-Only Mode

For servers without GPU acceleration:

```bash
# Force CPU execution
CUDA_VISIBLE_DEVICES="" python parakeet_server_diarization.py --port 8003
```

**Resource split:**

| Component | Backend | Hardware |
|-----------|---------|----------|
| ASR (Parakeet TDT) | parakeet-mlx (CPU fallback) | CPU |
| Diarization (pyannote) | PyTorch CPU | CPU |
| Audio preprocessing | librosa/scipy | CPU |

**Performance impact:**
- ASR: 3–10× slower than MLX/GPU
- Diarization: 2–5× slower than GPU
- Acceptable for low-volume or batch processing

**Optimization for CPU:**
```bash
# Increase thread count for PyTorch
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Reduce concurrency to avoid memory pressure
export MAX_CONCURRENT_TRANSCRIPTIONS=1
```

---

## Resource Allocation

### Concurrency Control

The server uses a semaphore to limit concurrent processing:

```bash
# Default: 2 concurrent requests
export MAX_CONCURRENT_TRANSCRIPTIONS=2

# For high-memory systems (64GB+ unified memory or 48GB+ VRAM)
export MAX_CONCURRENT_TRANSCRIPTIONS=4

# For constrained systems (16GB)
export MAX_CONCURRENT_TRANSCRIPTIONS=1
```

**Impact:**
- Each concurrent request loads audio + runs both models simultaneously
- Memory usage scales linearly with concurrency
- Requests beyond the limit are queued (not rejected)

### Memory Considerations

| Component | Approximate Memory |
|-----------|-------------------|
| Parakeet TDT 0.6B (MLX) | ~1.2 GB |
| Pyannote diarization pipeline | ~1.5 GB |
| Per-request audio buffer | ~50–200 MB |
| Medical vocabulary | ~1 MB |
| **Total baseline** | **~3 GB** |
| **Per additional concurrent request** | **+200–500 MB** |

**Recommendations:**
- 16 GB system: `MAX_CONCURRENT_TRANSCRIPTIONS=1`
- 32 GB system: `MAX_CONCURRENT_TRANSCRIPTIONS=2` (default)
- 64+ GB system: `MAX_CONCURRENT_TRANSCRIPTIONS=4`

### Timeout Configuration

```bash
# Processing timeout per request (seconds)
export TRANSCRIPTION_TIMEOUT=900  # 15 minutes (default)

# For short audio only (<1 min)
export TRANSCRIPTION_TIMEOUT=120

# For long recordings (>30 min)
export TRANSCRIPTION_TIMEOUT=1800
```

The timeout covers the combined time for parallel ASR + diarization. If exceeded, the request returns HTTP 504.

---

## Deployment Scenarios

### Single Mac Mini (MLX + CPU Diarization)

The most common deployment for this project — Apple Silicon Mac Mini running both ASR and diarization.

```bash
# Environment
export PYANNOTE_AUTH_TOKEN="hf_..."
export MEDICAL_VOCABULARY_PATH=./medical_vocabulary.txt
export MAX_CONCURRENT_TRANSCRIPTIONS=2
export PORT=8003

# Start
python parakeet_server_diarization.py \
  --port 8003 \
  --diarization-strategy pyannote \
  --vocabulary ./medical_vocabulary.txt
```

**Expected performance (M1/M2/M3 Mac Mini, 16 GB):**
- 1-minute audio: ~5–15 seconds total processing
- ASR: ~3–8 seconds (MLX accelerated)
- Diarization: ~4–12 seconds (CPU/MPS)
- Parallel execution reduces total to max(ASR, diarization)

### NVIDIA DGX / GPU Server

For high-throughput production with NVIDIA hardware:

```bash
export PYANNOTE_AUTH_TOKEN="hf_..."
export MEDICAL_VOCABULARY_PATH=./medical_vocabulary.txt
export MAX_CONCURRENT_TRANSCRIPTIONS=4
export ENV=production
export API_KEY="your-secure-api-key"
export BIND=0.0.0.0
export PORT=8003

python parakeet_server_diarization.py \
  --port 8003 \
  --diarization-strategy pyannote
```

**Expected performance (A100 80GB):**
- 1-minute audio: ~2–5 seconds total processing
- High concurrency (4+ simultaneous requests)
- Pyannote fully accelerated on CUDA

### CPU-Only Server

For development, testing, or low-volume deployments:

```bash
export CUDA_VISIBLE_DEVICES=""
export MAX_CONCURRENT_TRANSCRIPTIONS=1
export TRANSCRIPTION_TIMEOUT=1800
export PYANNOTE_AUTH_TOKEN="hf_..."

python parakeet_server_diarization.py --port 8003
```

**Expected performance (8-core CPU, 32 GB RAM):**
- 1-minute audio: ~30–90 seconds total processing
- Suitable for batch processing or low-traffic APIs

### Hybrid: MLX ASR + GPU Diarization (Separate Machines)

For maximum throughput, run ASR and diarization on separate machines:

**Machine 1 (Mac Mini — ASR only):**
```bash
# Base parakeet server (no diarization)
python parakeet_server.py --port 8002
```

**Machine 2 (GPU Server — Diarization only):**
```bash
# Diarization-only mode
export PYANNOTE_AUTH_TOKEN="hf_..."
python parakeet_server_diarization.py --port 8003
```

**Client orchestration:**
```bash
# Step 1: Transcribe on Mac Mini
TRANSCRIPT=$(curl -s -X POST http://mac-mini:8002/v1/audio/transcriptions \
  -F file=@recording.wav)

# Step 2: Diarize on GPU server
DIARIZATION=$(curl -s -X POST http://gpu-server:8003/v1/audio/diarize \
  -F file=@recording.wav \
  -F num_speakers=2)

# Step 3: Merge results in application
```

---

## Performance Benchmarks

Approximate processing times for a 1-minute audio file (2 speakers):

| Hardware | ASR | Diarization | Total (Parallel) | Total (Sequential) |
|----------|-----|-------------|-------------------|-------------------|
| Mac Mini M2 16GB | ~5s | ~8s | ~8s | ~13s |
| Mac Mini M3 Pro 36GB | ~3s | ~6s | ~6s | ~9s |
| NVIDIA A100 80GB | ~2s | ~3s | ~3s | ~5s |
| NVIDIA RTX 4090 24GB | ~2s | ~4s | ~4s | ~6s |
| CPU (8-core, no GPU) | ~20s | ~45s | ~45s | ~65s |

> **Note:** These are approximate values. Actual performance depends on audio complexity, number of speakers, and system load.

**Key insight:** Parallel execution saves time equal to the duration of the faster task. With balanced workloads, it approaches 2× speedup over sequential.

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `MPS backend error` | PyTorch MPS compatibility issue | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` or force CPU |
| `CUDA out of memory` | Insufficient VRAM | Reduce `MAX_CONCURRENT_TRANSCRIPTIONS` |
| Both models loading slowly | Cold start | Models are loaded once at startup; subsequent requests are fast |
| `Processing timed out` | Audio too long or system overloaded | Increase `TRANSCRIPTION_TIMEOUT` or split audio |
| Diarization inaccurate | Audio quality issues | Ensure 16kHz mono, minimal background noise |

### Forcing CPU for Diarization (when MPS is unstable)

```bash
# Disable MPS fallback
export PYTORCH_MPS_FALLBACK_ENABLED=0

# Or force CPU via PyTorch
python -c "
import torch
torch.set_default_device('cpu')
"
```

### Monitoring Resource Usage

```bash
# Apple Silicon — unified memory
sudo powermetrics --samplers gpu_power -i 1000

# NVIDIA GPU
nvidia-smi --loop=1

# General system
htop
```

### Log Levels

```bash
# Verbose logging for debugging
export LOG_LEVEL=DEBUG
python parakeet_server_diarization.py --port 8003

# Production (errors only)
export LOG_LEVEL=WARNING
```
