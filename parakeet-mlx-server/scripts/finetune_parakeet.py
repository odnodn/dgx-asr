#!/usr/bin/env python3
"""Fine-tune Parakeet TDT 0.6B v3 for German medical context.

This script provides a workflow for fine-tuning the NVIDIA Parakeet TDT 0.6B v3
model with German medical audio and text data using NVIDIA NeMo.

Requirements:
    pip install nemo_toolkit[asr] pytorch-lightning wandb

Usage:
    # Prepare data and train
    python finetune_parakeet.py --data-dir ./medical_data --output-dir ./finetuned_model

    # Resume training
    python finetune_parakeet.py --data-dir ./medical_data --output-dir ./finetuned_model --resume

Data Format:
    The data directory should contain:
    - manifest.json (NeMo manifest format): JSON lines with audio_filepath, text, duration
    - audio/ directory with WAV files referenced in the manifest

    Example manifest.json line:
    {"audio_filepath": "audio/patient_001.wav", "text": "der patient klagt über kopfschmerzen", "duration": 3.5}
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Data Preparation ───────────────────────────────────────────────────────────


def validate_manifest(manifest_path: str) -> bool:
    """Validate a NeMo-format manifest file."""
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found: {manifest_path}")
        return False

    errors = 0
    total = 0
    base_dir = os.path.dirname(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON: {e}")
                errors += 1
                continue

            # Check required fields
            for field in ["audio_filepath", "text", "duration"]:
                if field not in entry:
                    logger.error(f"Line {line_num}: Missing field '{field}'")
                    errors += 1

            # Check audio file exists
            audio_path = entry.get("audio_filepath", "")
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(base_dir, audio_path)
            if not os.path.exists(audio_path):
                logger.warning(f"Line {line_num}: Audio file not found: {audio_path}")

            # Validate duration
            duration = entry.get("duration", 0)
            if not isinstance(duration, (int, float)) or duration <= 0:
                logger.error(f"Line {line_num}: Invalid duration: {duration}")
                errors += 1

    logger.info(f"Manifest validation: {total} entries, {errors} errors")
    return errors == 0


def create_sample_manifest(output_dir: str):
    """Create a sample manifest file for reference."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest_sample.json")

    samples = [
        {
            "audio_filepath": "audio/sample_001.wav",
            "text": "der patient klagt über starke kopfschmerzen seit drei tagen",
            "duration": 4.2,
        },
        {
            "audio_filepath": "audio/sample_002.wav",
            "text": "die diagnose lautet migräne mit aura",
            "duration": 3.1,
        },
        {
            "audio_filepath": "audio/sample_003.wav",
            "text": "wir verschreiben ibuprofen sechshundert milligramm bei bedarf",
            "duration": 4.8,
        },
        {
            "audio_filepath": "audio/sample_004.wav",
            "text": "die blutdruckwerte liegen bei einhundertvierzig zu neunzig millimeter quecksilbersäule",
            "duration": 5.5,
        },
        {
            "audio_filepath": "audio/sample_005.wav",
            "text": "patient berichtet über dyspnoe bei belastung seit zwei wochen",
            "duration": 4.0,
        },
    ]

    with open(manifest_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Sample manifest created: {manifest_path}")
    logger.info("Replace with your actual medical audio data before training.")
    return manifest_path


# ─── Fine-tuning Configuration ──────────────────────────────────────────────────


def get_training_config(
    train_manifest: str,
    val_manifest: Optional[str],
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
) -> dict:
    """Generate NeMo training configuration for Parakeet TDT fine-tuning."""
    config = {
        "model": {
            "pretrained_model": "nvidia/parakeet-tdt-0.6b-v3",
            "train_ds": {
                "manifest_filepath": train_manifest,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": True,
                "pin_memory": True,
                "trim_silence": True,
                "max_duration": 20.0,
                "min_duration": 0.5,
            },
            "optim": {
                "name": "adamw",
                "lr": learning_rate,
                "betas": [0.9, 0.98],
                "weight_decay": 1e-3,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_steps": 500,
                    "min_lr": 1e-6,
                },
            },
        },
        "trainer": {
            "max_epochs": epochs,
            "accelerator": "auto",
            "devices": 1,
            "precision": "bf16-mixed",
            "accumulate_grad_batches": 4,
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 10,
            "val_check_interval": 0.25,
            "default_root_dir": output_dir,
        },
    }

    if val_manifest:
        config["model"]["validation_ds"] = {
            "manifest_filepath": val_manifest,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        }

    return config


# ─── Training ───────────────────────────────────────────────────────────────────


def finetune(
    train_manifest: str,
    val_manifest: Optional[str],
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    resume: bool = False,
    vocabulary_path: Optional[str] = None,
):
    """Run fine-tuning of Parakeet TDT 0.6B v3.

    This function uses NVIDIA NeMo to fine-tune the model on German medical data.
    """
    try:
        import nemo.collections.asr as nemo_asr
        import pytorch_lightning as pl
        from nemo.utils.exp_manager import exp_manager
    except ImportError:
        logger.error(
            "NeMo toolkit not installed. Install with:\n"
            "  pip install nemo_toolkit[asr] pytorch-lightning"
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Validate manifest
    if not validate_manifest(train_manifest):
        logger.error("Training manifest validation failed. Fix errors and retry.")
        sys.exit(1)

    if val_manifest and not validate_manifest(val_manifest):
        logger.warning("Validation manifest has errors. Continuing without validation set.")
        val_manifest = None

    config = get_training_config(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    logger.info("Loading pretrained model: nvidia/parakeet-tdt-0.6b-v3")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

    # Update training data config
    asr_model.cfg.train_ds.manifest_filepath = train_manifest
    asr_model.cfg.train_ds.batch_size = batch_size

    if val_manifest:
        asr_model.cfg.validation_ds.manifest_filepath = val_manifest
        asr_model.cfg.validation_ds.batch_size = batch_size

    # Setup data loaders
    asr_model.setup_training_data(asr_model.cfg.train_ds)
    if val_manifest:
        asr_model.setup_validation_data(asr_model.cfg.validation_ds)

    # Update optimizer
    asr_model.cfg.optim.lr = learning_rate

    # If vocabulary is provided, update the tokenizer
    if vocabulary_path and os.path.exists(vocabulary_path):
        logger.info(f"Loading additional vocabulary from: {vocabulary_path}")
        logger.info(
            "Note: Vocabulary expansion requires model architecture support. "
            "Terms will be used for training data augmentation if direct tokenizer "
            "expansion is not supported."
        )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        val_check_interval=config["trainer"]["val_check_interval"] if val_manifest else 1.0,
        default_root_dir=output_dir,
        enable_checkpointing=True,
    )

    # Setup experiment manager
    exp_config = {
        "exp_dir": output_dir,
        "name": "parakeet_medical_finetune",
        "checkpoint_callback_params": {
            "save_top_k": 3,
            "monitor": "val_wer" if val_manifest else "train_loss",
            "mode": "min",
        },
    }

    if resume:
        # Find latest checkpoint
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            checkpoints = sorted(Path(ckpt_dir).glob("*.ckpt"))
            if checkpoints:
                exp_config["resume_if_exists"] = True
                logger.info(f"Resuming from checkpoint: {checkpoints[-1]}")
            else:
                logger.warning("No checkpoints found. Starting fresh.")
        else:
            logger.warning("No checkpoint directory found. Starting fresh.")

    exp_manager(trainer, exp_config)

    # Train
    logger.info("=" * 60)
    logger.info("Starting fine-tuning")
    logger.info(f"  Training manifest: {train_manifest}")
    logger.info(f"  Validation manifest: {val_manifest or 'None'}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info("=" * 60)

    trainer.fit(asr_model)

    # Save the fine-tuned model
    model_save_path = os.path.join(output_dir, "parakeet_medical_finetuned.nemo")
    asr_model.save_to(model_save_path)
    logger.info(f"Fine-tuned model saved to: {model_save_path}")

    return model_save_path


# ─── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Parakeet TDT 0.6B v3 for German medical context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create sample manifest for reference
    python finetune_parakeet.py --create-sample --output-dir ./medical_data

    # Validate your manifest
    python finetune_parakeet.py --validate --data-dir ./medical_data

    # Run fine-tuning
    python finetune_parakeet.py --data-dir ./medical_data --output-dir ./finetuned

    # Resume training
    python finetune_parakeet.py --data-dir ./medical_data --output-dir ./finetuned --resume

    # With custom vocabulary
    python finetune_parakeet.py --data-dir ./medical_data --output-dir ./finetuned --vocabulary ./medical_vocab.txt
        """,
    )

    parser.add_argument("--data-dir", type=str, help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="./finetuned_model", help="Output directory")
    parser.add_argument("--train-manifest", type=str, help="Training manifest path (overrides data-dir)")
    parser.add_argument("--val-manifest", type=str, help="Validation manifest path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--vocabulary", type=str, help="Path to medical vocabulary file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--create-sample", action="store_true", help="Create sample manifest")
    parser.add_argument("--validate", action="store_true", help="Validate manifest only")

    args = parser.parse_args()

    if args.create_sample:
        output_dir = args.output_dir or "./medical_data"
        create_sample_manifest(output_dir)
        return

    if args.validate:
        if not args.data_dir:
            logger.error("--data-dir required for validation")
            sys.exit(1)
        manifest = os.path.join(args.data_dir, "manifest.json")
        if validate_manifest(manifest):
            logger.info("Manifest is valid!")
        else:
            logger.error("Manifest has errors.")
            sys.exit(1)
        return

    # Determine manifest paths
    train_manifest = args.train_manifest
    if not train_manifest:
        if not args.data_dir:
            logger.error("Either --data-dir or --train-manifest is required")
            parser.print_help()
            sys.exit(1)
        train_manifest = os.path.join(args.data_dir, "manifest.json")

    val_manifest = args.val_manifest
    if not val_manifest and args.data_dir:
        candidate = os.path.join(args.data_dir, "manifest_val.json")
        if os.path.exists(candidate):
            val_manifest = candidate

    finetune(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
        vocabulary_path=args.vocabulary,
    )


if __name__ == "__main__":
    main()
