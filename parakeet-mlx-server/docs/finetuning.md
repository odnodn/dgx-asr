# Fine-Tuning Parakeet TDT for German Medical Context

This document covers the fine-tuning workflow for adapting the NVIDIA Parakeet TDT 0.6B v3 model to German medical speech recognition using NVIDIA NeMo.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
  - [Manifest Format](#manifest-format)
  - [Audio Requirements](#audio-requirements)
  - [Creating Sample Data](#creating-sample-data)
  - [Validation](#validation)
- [Training Configuration](#training-configuration)
  - [Hyperparameters](#hyperparameters)
  - [Optimizer and Scheduler](#optimizer-and-scheduler)
  - [Precision and Performance](#precision-and-performance)
- [Running Fine-Tuning](#running-fine-tuning)
  - [Basic Training](#basic-training)
  - [Resuming Training](#resuming-training)
  - [Custom Vocabulary](#custom-vocabulary)
- [Medical Vocabulary File](#medical-vocabulary-file)
- [Output and Checkpoints](#output-and-checkpoints)
- [Deployment After Fine-Tuning](#deployment-after-fine-tuning)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Overview

The fine-tuning script (`scripts/finetune_parakeet.py`) adapts the pre-trained Parakeet TDT 0.6B v3 model for German medical speech recognition. This is useful for:

- Improving recognition of medical terminology (diagnoses, medications, abbreviations)
- Adapting to specific acoustic environments (clinics, consultation rooms)
- Improving speaker-specific patterns in doctor-patient conversations
- Reducing word error rate on domain-specific content

**Base Model:** [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)  
**Framework:** NVIDIA NeMo + PyTorch Lightning  
**Architecture:** Token-and-Duration Transducer (TDT)

---

## Prerequisites

### Software

```bash
# NeMo toolkit with ASR support
pip install nemo_toolkit[asr]

# PyTorch Lightning
pip install pytorch-lightning

# Optional: Weights & Biases for experiment tracking
pip install wandb
```

### Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| Minimum | 16 GB GPU | Batch size 4, gradient accumulation 8 |
| Recommended | 24+ GB GPU | Batch size 8, gradient accumulation 4 |
| Multi-GPU | 2× 16 GB | Automatic with NeMo's distributed training |

The script uses BF16 mixed precision by default, which requires:
- NVIDIA Ampere (A100, A6000) or newer GPU, **or**
- Apple Silicon with MPS backend (limited NeMo support)

---

## Data Preparation

### Manifest Format

Training data uses the **NeMo manifest format** — a JSON Lines file where each line is a JSON object:

```json
{"audio_filepath": "audio/patient_001.wav", "text": "der patient klagt über kopfschmerzen seit drei tagen", "duration": 4.2}
{"audio_filepath": "audio/patient_002.wav", "text": "die diagnose lautet migräne mit aura", "duration": 3.1}
{"audio_filepath": "audio/patient_003.wav", "text": "wir verschreiben ibuprofen sechshundert milligramm bei bedarf", "duration": 4.8}
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | string | Path to audio file (relative to manifest or absolute) |
| `text` | string | Ground-truth transcription (lowercase recommended) |
| `duration` | float | Audio duration in seconds |

### Audio Requirements

- **Format:** WAV (16-bit PCM recommended), also supports FLAC, MP3
- **Sample Rate:** 16 kHz (model's native rate; files are resampled if different)
- **Channels:** Mono (stereo is averaged)
- **Duration:** 0.5–20 seconds per utterance (configurable via `min_duration`/`max_duration`)
- **Quality:** Clean recordings with minimal background noise yield best results

### Directory Structure

```
medical_data/
├── manifest.json          # Training manifest (required)
├── manifest_val.json      # Validation manifest (optional, auto-detected)
└── audio/
    ├── patient_001.wav
    ├── patient_002.wav
    ├── consultation_001.wav
    └── ...
```

### Creating Sample Data

Generate a reference manifest with example entries:

```bash
python scripts/finetune_parakeet.py --create-sample --output-dir ./medical_data
```

This creates `medical_data/manifest_sample.json` with example medical German utterances. Replace with your actual audio data before training.

### Validation

Validate your manifest before training to catch formatting errors:

```bash
python scripts/finetune_parakeet.py --validate --data-dir ./medical_data
```

The validator checks:
- JSON syntax on each line
- Presence of required fields (`audio_filepath`, `text`, `duration`)
- Audio file existence (warns if missing)
- Duration validity (must be positive number)

---

## Training Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 8 | Batch size per step |
| `--learning-rate` | 1e-4 | Peak learning rate |
| Gradient accumulation | 4 | Effective batch = batch_size × 4 = 32 |
| Gradient clipping | 1.0 | Max gradient norm |
| Max duration | 20.0s | Maximum audio duration per sample |
| Min duration | 0.5s | Minimum audio duration per sample |

### Optimizer and Scheduler

- **Optimizer:** AdamW (`β₁=0.9, β₂=0.98, weight_decay=1e-3`)
- **Scheduler:** Cosine Annealing
  - Warmup: 500 steps
  - Minimum LR: 1e-6
- **Validation interval:** Every 25% of an epoch (`val_check_interval=0.25`)

### Precision and Performance

- **Mixed Precision:** BF16 (bfloat16) by default
  - Reduces memory usage by ~50% vs FP32
  - Maintains numerical stability for gradients
- **Gradient Accumulation:** 4 steps
  - Effective batch size = `batch_size × accumulate_grad_batches`
  - Default: 8 × 4 = 32 effective batch size
- **Pin Memory:** Enabled for faster CPU→GPU data transfer
- **Trim Silence:** Removes leading/trailing silence from audio samples

---

## Running Fine-Tuning

### Basic Training

```bash
python scripts/finetune_parakeet.py \
  --data-dir ./medical_data \
  --output-dir ./finetuned_model \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-4
```

### With Separate Train/Validation Manifests

```bash
python scripts/finetune_parakeet.py \
  --train-manifest ./data/train_manifest.json \
  --val-manifest ./data/val_manifest.json \
  --output-dir ./finetuned_model
```

### Resuming Training

If training is interrupted, resume from the latest checkpoint:

```bash
python scripts/finetune_parakeet.py \
  --data-dir ./medical_data \
  --output-dir ./finetuned_model \
  --resume
```

The script automatically finds the latest `.ckpt` file in `<output-dir>/checkpoints/`.

### Custom Vocabulary

Provide a medical vocabulary file to enhance domain recognition:

```bash
python scripts/finetune_parakeet.py \
  --data-dir ./medical_data \
  --output-dir ./finetuned_model \
  --vocabulary ./medical_vocabulary.txt
```

> **Note:** Vocabulary expansion depends on model architecture support. If direct tokenizer expansion isn't supported, terms inform data augmentation strategies.

---

## Medical Vocabulary File

The `medical_vocabulary.txt` file contains 200+ German medical terms organized by category:

### Categories

| Category | Examples |
|----------|----------|
| Diagnoses / Diseases | Migräne, Multiple Sklerose, Epilepsie |
| Symptoms | Dyspnoe, Tachykardie, Parästhesie |
| Procedures / Examinations | Elektroenzephalographie, Lumbalpunktion |
| Medications / Brand Names | Ibuprofen, Metoprolol, Levetiracetam |
| Abbreviations | EEG, MRT, CT, EKG, HbA1c |
| Anatomy | Cerebellum, Hypothalamus, Thalamus |
| Medical Specialties | Neurologie, Kardiologie, Onkologie |

### Format

```
# Lines starting with # are comments
Migräne
Migräne mit Aura
Multiple Sklerose
Morbus Parkinson
EEG
MRT
```

### Usage in Training vs. Post-Processing

The vocabulary serves two purposes:

1. **During fine-tuning** — Terms can inform data augmentation to improve model recognition.
2. **During inference** — The diarization server applies case-correcting post-processing, replacing case-insensitive matches with the canonical form from the vocabulary.

---

## Output and Checkpoints

### Directory Structure After Training

```
finetuned_model/
├── parakeet_medical_finetuned.nemo    # Final model (NeMo format)
├── checkpoints/
│   ├── epoch=10-step=5000.ckpt        # Top-3 checkpoints
│   ├── epoch=25-step=12500.ckpt
│   └── epoch=49-step=25000.ckpt
├── lightning_logs/
│   └── version_0/
│       ├── hparams.yaml
│       └── metrics.csv
└── parakeet_medical_finetune/
    └── ... (NeMo experiment manager logs)
```

### Checkpoint Selection

The training saves the **top 3 checkpoints** based on:
- `val_wer` (Word Error Rate) if a validation manifest is provided
- `train_loss` otherwise

The final saved `.nemo` model corresponds to the state at the end of training (last epoch).

---

## Deployment After Fine-Tuning

After training, deploy the fine-tuned model with the diarization server:

```bash
# Set custom model path (if supported by parakeet-mlx)
export PARAKEET_MODEL=./finetuned_model/parakeet_medical_finetuned.nemo

# Start server with medical vocabulary
python parakeet_server_diarization.py \
  --port 8003 \
  --vocabulary ./medical_vocabulary.txt
```

> **Note:** The parakeet-mlx server currently loads models from HuggingFace by ID. To use a locally fine-tuned `.nemo` model, you may need to convert it to the MLX format or serve it via NeMo's inference APIs.

---

## Tips and Best Practices

### Data Quality

- **Transcription accuracy** — Ensure ground-truth text is correct. Errors in labels propagate to the model.
- **Domain coverage** — Include diverse medical scenarios (consultations, diagnoses, medication discussions).
- **Speaker variety** — Use multiple speakers to avoid overfitting to specific voices.
- **Balanced duration** — Mix short (2–5s) and longer (10–20s) utterances.

### Training

- **Start with a small dataset** — Validate the pipeline with 50–100 samples before scaling up.
- **Monitor validation WER** — If it stops improving or increases, training may be overfitting.
- **Lower learning rate for larger datasets** — Use 5e-5 for 10,000+ samples.
- **Increase epochs for small datasets** — Use 100+ epochs for fewer than 500 samples.

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM (Out of Memory) | Reduce `--batch-size` or increase gradient accumulation |
| High WER after training | Check data quality, reduce learning rate |
| Training diverges | Lower learning rate, increase warmup steps |
| Slow training | Enable BF16 (default), use NVMe for data |
| Checkpoint not found on resume | Verify `--output-dir` matches previous run |
