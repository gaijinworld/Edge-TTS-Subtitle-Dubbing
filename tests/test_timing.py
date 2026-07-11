from __future__ import annotations

import numpy as np
import soundfile as sf
import tempfile
import os

from edge_tts_subtitle_dubbing.timing import (
    adjust_audio_length,
    _exact_sample_length,
    SAMPLE_RATE,
)


class TestExactSampleLength:
    def test_trim(self):
        y = np.ones(100, dtype=np.float32)
        result = _exact_sample_length(y, 50)
        assert len(result) == 50

    def test_pad(self):
        y = np.ones(10, dtype=np.float32)
        result = _exact_sample_length(y, 50)
        assert len(result) == 50
        assert result[10] == 0.0

    def test_zero_target(self):
        y = np.ones(10, dtype=np.float32)
        result = _exact_sample_length(y, 0)
        assert len(result) == 0


class TestAdjustAudioLength:
    def test_returns_correct_length(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            y = np.ones(SAMPLE_RATE, dtype=np.float32)
            sf.write(wav_path, y, SAMPLE_RATE)

            result = adjust_audio_length(
                wav_path,
                desired_length=2.0,
                sample_rate=SAMPLE_RATE,
                max_speed_factor=2.0,
                enable_exact_duration_text_stretching=True,
            )
            assert len(result) == int(2.0 * SAMPLE_RATE)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def test_zero_desired_length_exact(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            y = np.ones(SAMPLE_RATE, dtype=np.float32)
            sf.write(wav_path, y, SAMPLE_RATE)

            result = adjust_audio_length(
                wav_path,
                desired_length=0.0,
                sample_rate=SAMPLE_RATE,
                enable_exact_duration_text_stretching=True,
            )
            assert len(result) == 0
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
