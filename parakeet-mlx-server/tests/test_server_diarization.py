"""Unit tests for parakeet_server_diarization.py"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os
import sys

# Mock heavy dependencies before importing
sys.modules["parakeet_mlx"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["pyannote"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torchaudio"] = MagicMock()
sys.modules["nemo"] = MagicMock()
sys.modules["nemo.collections"] = MagicMock()
sys.modules["nemo.collections.asr"] = MagicMock()
sys.modules["nemo.collections.asr.models"] = MagicMock()

from parakeet_server_diarization import (
    assign_speaker_names,
    merge_transcription_with_diarization,
    apply_vocabulary_corrections,
    load_medical_vocabulary,
    DEFAULT_SPEAKER_NAMES,
    DiarizationResponse,
    DiarizationSegment,
    app,
)

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


# ─── Unit Tests for Helper Functions ────────────────────────────────────────────


class TestAssignSpeakerNames:
    def test_two_speakers_default(self):
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00"},
        ]
        result = assign_speaker_names(segments, ["Arzt", "Patient"])
        assert result[0]["speaker"] == "Arzt"
        assert result[1]["speaker"] == "Patient"
        assert result[2]["speaker"] == "Arzt"

    def test_three_speakers(self):
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_02"},
        ]
        result = assign_speaker_names(segments, ["Arzt", "Patient", "Angehöriger"])
        assert result[0]["speaker"] == "Arzt"
        assert result[1]["speaker"] == "Patient"
        assert result[2]["speaker"] == "Angehöriger"

    def test_more_speakers_than_names(self):
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_02"},
        ]
        result = assign_speaker_names(segments, ["Arzt", "Patient"])
        assert result[0]["speaker"] == "Arzt"
        assert result[1]["speaker"] == "Patient"
        assert result[2]["speaker"] == "Speaker 3"

    def test_no_names_provided(self):
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        ]
        result = assign_speaker_names(segments, None)
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"

    def test_empty_segments(self):
        result = assign_speaker_names([], ["Arzt", "Patient"])
        assert result == []


class TestMergeTranscriptionWithDiarization:
    def test_with_asr_segments(self):
        asr_segments = [
            {"text": "Hallo", "start": 0.0, "end": 0.5},
            {"text": "wie geht es Ihnen", "start": 0.5, "end": 2.0},
            {"text": "mir geht es gut", "start": 2.0, "end": 3.5},
        ]
        diar_segments = [
            {"start": 0.0, "end": 2.0, "speaker": "Arzt"},
            {"start": 2.0, "end": 4.0, "speaker": "Patient"},
        ]
        result = merge_transcription_with_diarization(asr_segments, diar_segments, "")
        # First two segments overlap with Arzt, last with Patient
        assert result[0]["speaker"] == "Arzt"
        assert "Hallo" in result[0]["text"]
        assert result[-1]["speaker"] == "Patient"

    def test_without_asr_segments(self):
        diar_segments = [
            {"start": 0.0, "end": 2.0, "speaker": "Arzt"},
            {"start": 2.0, "end": 4.0, "speaker": "Patient"},
        ]
        result = merge_transcription_with_diarization(
            None, diar_segments, "Hallo wie geht es Ihnen"
        )
        assert len(result) == 2
        assert result[0]["speaker"] == "Arzt"
        assert result[1]["speaker"] == "Patient"

    def test_empty_diarization(self):
        result = merge_transcription_with_diarization(None, [], "some text")
        assert len(result) == 1
        assert result[0]["text"] == "some text"

    def test_merges_consecutive_same_speaker(self):
        asr_segments = [
            {"text": "Hallo", "start": 0.0, "end": 0.5},
            {"text": "wie geht es", "start": 0.5, "end": 1.5},
            {"text": "Ihnen heute", "start": 1.5, "end": 2.0},
        ]
        diar_segments = [
            {"start": 0.0, "end": 3.0, "speaker": "Arzt"},
        ]
        result = merge_transcription_with_diarization(asr_segments, diar_segments, "")
        # All should be merged into one segment
        assert len(result) == 1
        assert result[0]["speaker"] == "Arzt"
        assert "Hallo" in result[0]["text"]
        assert "Ihnen heute" in result[0]["text"]


class TestApplyVocabularyCorrections:
    def test_corrects_case(self):
        vocab = ["Migräne", "Ibuprofen", "EEG"]
        text = "der patient hat migräne und bekommt ibuprofen"
        result = apply_vocabulary_corrections(text, vocab)
        assert "Migräne" in result
        assert "Ibuprofen" in result

    def test_no_vocabulary(self):
        text = "some text"
        result = apply_vocabulary_corrections(text, None)
        assert result == text

    def test_empty_vocabulary(self):
        text = "some text"
        result = apply_vocabulary_corrections(text, [])
        assert result == text

    def test_preserves_unmatched_text(self):
        vocab = ["Migräne"]
        text = "der patient ist gesund"
        result = apply_vocabulary_corrections(text, vocab)
        assert result == text


class TestLoadMedicalVocabulary:
    def test_load_from_file(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("Migräne\nIbuprofen\n# comment\n\nEEG\n", encoding="utf-8")
        result = load_medical_vocabulary(str(vocab_file))
        assert result == ["Migräne", "Ibuprofen", "EEG"]

    def test_file_not_found(self):
        result = load_medical_vocabulary("/nonexistent/path.txt")
        assert result is None

    def test_no_path(self):
        result = load_medical_vocabulary(None)
        assert result is None


class TestDefaultSpeakerNames:
    def test_two_speakers(self):
        assert DEFAULT_SPEAKER_NAMES[2] == ["Arzt", "Patient"]

    def test_three_speakers(self):
        assert DEFAULT_SPEAKER_NAMES[3] == ["Arzt", "Patient", "Angehöriger"]

    def test_four_speakers(self):
        assert DEFAULT_SPEAKER_NAMES[4] == ["Arzt", "Patient", "Angehöriger", "Begleitung"]


# ─── API Endpoint Tests ─────────────────────────────────────────────────────────


class TestEndpoints:
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "neuro-parakeet-diarization"
        assert "diarization_strategy" in data

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "diarization_loaded" in data
        assert "diarization_strategy" in data
        assert "medical_vocabulary_loaded" in data

    def test_liveness_endpoint(self, client):
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_config_endpoint(self, client):
        response = client.get("/v1/diarization/config")
        assert response.status_code == 200
        data = response.json()
        assert "diarization_strategy" in data
        assert "default_num_speakers" in data
        assert "default_speaker_names" in data

    def test_transcription_no_model(self, client):
        """Test that transcription returns 503 when model is not loaded."""
        import io

        audio_content = b"\x00" * 1000
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "parakeet-tdt-0.6b-v3", "num_speakers": "2"},
        )
        assert response.status_code == 503

    def test_diarize_only_no_pipeline(self, client):
        """Test diarize-only endpoint returns 503 when pipeline not loaded."""
        import io

        audio_content = b"\x00" * 1000
        response = client.post(
            "/v1/audio/diarize",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"num_speakers": "2"},
        )
        assert response.status_code == 503

    def test_transcription_empty_file(self, client):
        """Test that empty file returns 400."""
        import io

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b""), "audio/wav")},
            data={"model": "parakeet-tdt-0.6b-v3"},
        )
        # Either 400 (empty file) or 503 (no model) is acceptable
        assert response.status_code in [400, 503]

    def test_transcription_invalid_file_type(self, client):
        """Test that invalid file type returns 400 or 503 (if model not loaded)."""
        import io

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.exe", io.BytesIO(b"\x00" * 100), "application/exe")},
            data={"model": "parakeet-tdt-0.6b-v3"},
        )
        assert response.status_code in [400, 503]


class TestDiarizationResponse:
    def test_response_model(self):
        response = DiarizationResponse(
            text="Hallo wie geht es Ihnen",
            segments=[
                DiarizationSegment(speaker="Arzt", text="Hallo wie geht es Ihnen", start=0.0, end=2.0)
            ],
            speakers=["Arzt"],
            num_speakers=1,
            diarization_strategy="pyannote",
        )
        assert response.text == "Hallo wie geht es Ihnen"
        assert len(response.segments) == 1
        assert response.segments[0].speaker == "Arzt"
        assert response.diarization_strategy == "pyannote"
