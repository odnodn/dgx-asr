# Speaker Diarization

This document covers the speaker diarization capabilities of the parakeet-mlx diarization server, including system architecture, supported backends, HuggingFace authentication for gated models, and API usage.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Diarization Backends](#diarization-backends)
  - [Pyannote (Default)](#pyannote-default)
  - [NeMo SortedFormer](#nemo-sortedformer)
- [HuggingFace Token for Gated Models](#huggingface-token-for-gated-models)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
  - [Transcription with Diarization](#transcription-with-diarization)
  - [Diarization Only](#diarization-only)
  - [Configuration Endpoint](#configuration-endpoint)
- [Speaker Name Assignment](#speaker-name-assignment)
- [Medical Vocabulary Post-Processing](#medical-vocabulary-post-processing)
- [Segment Merging Algorithm](#segment-merging-algorithm)

---

## Architecture Overview

The diarization server (`parakeet_server_diarization.py`) is a **standalone FastAPI application** that extends the base `parakeet_server.py` without modifying it. It reuses core utilities (text cleaning, segment extraction, file validation, model loading) via Python imports.

```
┌─────────────────────────────────────────────────────────────────┐
│                 parakeet_server_diarization.py                   │
│                                                                 │
│  ┌──────────────┐   ┌────────────────────┐   ┌──────────────┐  │
│  │   FastAPI     │   │  Diarization       │   │  Medical     │  │
│  │   Endpoints   │──▶│  Pipeline          │   │  Vocabulary  │  │
│  │              │   │  (pyannote/NeMo)   │   │  Corrections │  │
│  └──────┬───────┘   └────────────────────┘   └──────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              parakeet_server.py (imports)                 │   │
│  │  • load_model()       • clean_text()                     │   │
│  │  • extract_text()     • extract_segments()               │   │
│  │  • sanitize_filename() • validate_file_type()            │   │
│  │  • check_python_version() • validate_system_requirements()│   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

1. **Audio Upload** — File is validated (type, size, extension) and written to a temporary path.
2. **Parallel Execution** — ASR transcription and speaker diarization run concurrently via `asyncio.gather()`.
3. **Speaker Name Mapping** — Generic speaker labels (`SPEAKER_00`, `SPEAKER_01`) are mapped to provided names.
4. **Temporal Merge** — ASR segments are merged with diarization segments using temporal overlap matching.
5. **Post-Processing** — Medical vocabulary corrections are applied to the final text.
6. **Response** — JSON response with full text, speaker-labeled segments, and metadata.

```
Audio File
    │
    ├──▶ ASR Transcription (parakeet-mlx) ──▶ text + word-level segments
    │                                              │
    ├──▶ Speaker Diarization (pyannote/NeMo) ──▶ speaker segments (start, end, speaker)
    │                                              │
    ▼                                              ▼
    ┌──────────────────────────────────────────────────┐
    │         Temporal Overlap Merging                  │
    │   Match ASR segments to speaker by overlap       │
    │   Merge consecutive same-speaker segments        │
    └──────────────────────────────────────────────────┘
                        │
                        ▼
            Medical Vocabulary Corrections
                        │
                        ▼
              Final Speaker-Labeled Output
```

---

## Diarization Backends

### Pyannote (Default)

[pyannote.audio](https://github.com/pyannote/pyannote-audio) is a neural speaker diarization toolkit. The server uses the `pyannote/speaker-diarization-3.1` pipeline by default.

**Features:**
- Pre-trained neural models for speaker segmentation and clustering
- Supports specifying a known number of speakers
- Automatic GPU acceleration (CUDA or Apple MPS)
- Widely used in academic and production settings

**Model:** `pyannote/speaker-diarization-3.1` (configurable via `PYANNOTE_MODEL` env var)

### NeMo SortedFormer

[SortedFormer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html) is NVIDIA's speaker diarization model based on the NeMo framework.

**Features:**
- Designed for multi-speaker scenarios
- Integrates with NeMo ASR pipeline
- End-to-end neural diarization

**Model:** `nvidia/sortedformer_diar_base` (configurable via `SORTEDFORMER_MODEL` env var)

> **Note:** SortedFormer requires the full NeMo toolkit: `pip install nemo_toolkit[asr]`

---

## HuggingFace Token for Gated Models

The pyannote speaker diarization model (`pyannote/speaker-diarization-3.1`) is a **gated model** on HuggingFace. You must accept the model's license and provide an authentication token to download it.

### Step 1: Accept the Model License

1. Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) on HuggingFace.
2. Log in with your HuggingFace account.
3. Read and accept the license agreement.
4. Also accept the license for the segmentation model: [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

> **Important:** You must accept **both** model licenses — the diarization pipeline depends on the segmentation model internally.

### Step 2: Create an Access Token

1. Go to [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Click **"New token"**.
3. Give it a name (e.g., `parakeet-diarization`).
4. Select **"Read"** permission (sufficient for model download).
5. Click **"Generate"** and copy the token.

### Step 3: Configure the Token

Set the token as the `PYANNOTE_AUTH_TOKEN` environment variable:

```bash
# Option A: Environment variable (recommended for production)
export PYANNOTE_AUTH_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Option B: .env file
echo 'PYANNOTE_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' >> .env

# Option C: Command line
python parakeet_server_diarization.py --port 8003
# (with PYANNOTE_AUTH_TOKEN already set in environment)
```

### Step 4: Verify Access

```bash
# Test that the token works
python -c "
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token='hf_YOUR_TOKEN_HERE'
)
print('Pipeline loaded successfully!')
"
```

### Troubleshooting Token Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `401 Unauthorized` | Token is invalid or expired | Regenerate token on HuggingFace |
| `403 Forbidden` | License not accepted | Accept license on model page |
| `Repository not found` | Token lacks read permission | Create token with "Read" scope |
| Model downloads but fails | Segmentation model license not accepted | Accept `pyannote/segmentation-3.0` license |

---

## Installation

```bash
# Install base server dependencies
pip install -r requirements.txt

# Install diarization dependencies
pip install -r requirements-diarization.txt
```

### Dependencies (requirements-diarization.txt)

| Package | Purpose |
|---------|---------|
| `pyannote.audio>=3.3.2` | Speaker diarization pipeline |
| `torch>=2.0.0` | Neural network backend |
| `torchaudio>=2.0.0` | Audio loading for pyannote |
| `scipy>=1.11.0` | Signal processing |
| `soundfile>=0.12.1` | Audio file I/O |
| `librosa>=0.10.0` | Audio preprocessing |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DIARIZATION_STRATEGY` | `pyannote` | Backend: `pyannote` or `sortedformer` |
| `DEFAULT_NUM_SPEAKERS` | `2` | Default number of speakers |
| `PYANNOTE_AUTH_TOKEN` | (none) | HuggingFace token for pyannote models |
| `PYANNOTE_MODEL` | `pyannote/speaker-diarization-3.1` | Pyannote model name |
| `SORTEDFORMER_MODEL` | `nvidia/sortedformer_diar_base` | SortedFormer model name |
| `MEDICAL_VOCABULARY_PATH` | (none) | Path to medical vocabulary file |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `2` | Max concurrent requests |
| `TRANSCRIPTION_TIMEOUT` | `900` | Timeout in seconds |
| `API_KEY` | (none) | API key for authentication |
| `CORS_ORIGINS` | `http://localhost:8003,...` | Allowed CORS origins |
| `ENV` | `development` | Set to `production` to enforce API key |
| `PORT` | `8003` | Server port |
| `BIND` | `127.0.0.1` | Bind address |

### Command-Line Arguments

```bash
python parakeet_server_diarization.py \
  --port 8003 \
  --diarization-strategy pyannote \
  --num-speakers 2 \
  --vocabulary ./medical_vocabulary.txt \
  --model nvidia/parakeet-tdt-0.6b-v3
```

| Argument | Description |
|----------|-------------|
| `--port` | Server port (default: 8003) |
| `--diarization-strategy` | `pyannote` or `sortedformer` |
| `--num-speakers` | Default number of speakers |
| `--vocabulary` | Path to medical vocabulary file |
| `--model` | ASR model ID |
| `--skip-validation` | Skip system requirement checks |

---

## API Endpoints

### Transcription with Diarization

**`POST /v1/audio/transcriptions`** — OpenAI-compatible endpoint with speaker labels.

```bash
curl -X POST http://localhost:8003/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=parakeet-tdt-0.6b-v3 \
  -F num_speakers=2 \
  -F speaker_names="Arzt,Patient" \
  -F response_format=json
```

**Form Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | (required) | Audio file (WAV, MP3, FLAC, OGG, M4A, WEBM) |
| `model` | string | `parakeet-tdt-0.6b-v3` | Model name (OpenAI compatibility) |
| `num_speakers` | int | `2` | Number of speakers in the audio |
| `speaker_names` | string | `Arzt,Patient` | Comma-separated speaker names |
| `diarization_strategy` | string | (server default) | Override diarization backend |
| `response_format` | string | `json` | `json` or `text` |
| `recording_timestamp` | string | (none) | Optional recording timestamp |

**JSON Response:**

```json
{
  "text": "Full transcription text...",
  "segments": [
    {
      "speaker": "Arzt",
      "text": "Guten Tag, wie geht es Ihnen heute?",
      "start": 0.0,
      "end": 2.5
    },
    {
      "speaker": "Patient",
      "text": "Mir geht es nicht so gut, ich habe Kopfschmerzen.",
      "start": 2.8,
      "end": 5.1
    }
  ],
  "speakers": ["Arzt", "Patient"],
  "num_speakers": 2,
  "recording_timestamp": null,
  "diarization_strategy": "pyannote"
}
```

**Text Response** (when `response_format=text`):

```
Arzt: Guten Tag, wie geht es Ihnen heute?
Patient: Mir geht es nicht so gut, ich habe Kopfschmerzen.
```

### Diarization Only

**`POST /v1/audio/diarize`** — Speaker segmentation without transcription.

```bash
curl -X POST http://localhost:8003/v1/audio/diarize \
  -F file=@recording.wav \
  -F num_speakers=2 \
  -F speaker_names="Arzt,Patient"
```

**Response:**

```json
{
  "segments": [
    {"speaker": "Arzt", "start": 0.0, "end": 2.5},
    {"speaker": "Patient", "start": 2.8, "end": 5.1},
    {"speaker": "Arzt", "start": 5.3, "end": 8.0}
  ],
  "speakers": ["Arzt", "Patient"],
  "num_speakers": 2,
  "diarization_strategy": "pyannote"
}
```

### Configuration Endpoint

**`GET /v1/diarization/config`** — Returns current server configuration.

```json
{
  "diarization_strategy": "pyannote",
  "default_num_speakers": 2,
  "default_speaker_names": {
    "2": ["Arzt", "Patient"],
    "3": ["Arzt", "Patient", "Angehöriger"],
    "4": ["Arzt", "Patient", "Angehöriger", "Begleitung"]
  },
  "diarization_loaded": true,
  "medical_vocabulary_loaded": true,
  "medical_vocabulary_terms": 215
}
```

---

## Speaker Name Assignment

Diarization backends produce generic speaker labels (e.g., `SPEAKER_00`, `SPEAKER_01`). The server maps these to meaningful names in order of appearance:

1. **Custom names** — If `speaker_names` is provided (e.g., `"Arzt,Patient"`), speakers are named in the order they first appear in the audio.
2. **Default names** — If no names are provided, built-in defaults are used based on `num_speakers`:
   - 2 speakers: `["Arzt", "Patient"]`
   - 3 speakers: `["Arzt", "Patient", "Angehöriger"]`
   - 4 speakers: `["Arzt", "Patient", "Angehöriger", "Begleitung"]`
3. **Fallback** — If more speakers are detected than names provided, extras are labeled `Speaker N`.

> **Note:** Speaker assignment is based on **order of first appearance** in the audio — the first person to speak is assigned the first name.

---

## Medical Vocabulary Post-Processing

The server supports post-processing corrections using a medical vocabulary file. This corrects case and spelling of domain-specific terms in the transcription output.

### How It Works

1. The vocabulary file (`medical_vocabulary.txt`) contains canonical forms of medical terms, one per line.
2. After transcription, a case-insensitive search-and-replace is performed.
3. Any match in the transcription text is replaced with the canonical form from the vocabulary.

**Example:**
- ASR outputs: `"der patient hat eine migräne mit aura"`
- After correction: `"der Patient hat eine Migräne mit Aura"`

### Enabling Vocabulary Corrections

```bash
# Via environment variable
export MEDICAL_VOCABULARY_PATH=./medical_vocabulary.txt

# Via command line
python parakeet_server_diarization.py --vocabulary ./medical_vocabulary.txt
```

---

## Segment Merging Algorithm

The temporal overlap merging algorithm assigns speaker labels to ASR segments:

1. For each ASR segment (with start/end timestamps):
   - Calculate overlap duration with every diarization segment.
   - Assign the speaker with the maximum overlap.
2. After assignment, consecutive segments from the same speaker are merged:
   - Text is concatenated.
   - Start time is from the first segment, end time from the last.

**Edge Cases:**
- If ASR produces no segment-level timing, text is distributed proportionally across diarization segments by duration.
- If a transcription segment has zero overlap with all diarization segments, it is labeled "Unknown".
