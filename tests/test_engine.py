from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from edge_tts_subtitle_dubbing.engine import srt_to_audio_numpy
from edge_tts_subtitle_dubbing.models import DubbingResult
from edge_tts_subtitle_dubbing.timing import SAMPLE_RATE


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_three_cue_srt_returns_result(tmp_path):
    """Test that srt_to_audio_numpy processes a 3-cue SRT and returns a DubbingResult."""
    srt_path = str(FIXTURES_DIR / "three-cue.srt")
    output_path = str(tmp_path / "test_output.wav")

    async def mock_batch(segments_info, voice, batch_size, retries):
        results = {}
        for text, cache_path, idx in segments_info:
            dummy_audio = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1
            sf.write(cache_path, dummy_audio, SAMPLE_RATE)
            results[idx] = None
        return results

    with patch(
        "edge_tts_subtitle_dubbing.engine.generate_audio_batch",
        side_effect=mock_batch,
    ):
        result = srt_to_audio_numpy(
            srt_path=srt_path,
            output_path=output_path,
            voice="en-US-JennyNeural",
            log_level="ERROR",
        )

    assert isinstance(result, DubbingResult)
    assert result.total_cues == 3
    assert result.output_path == output_path
    assert os.path.exists(output_path)


def test_missing_srt_returns_empty_result():
    """Test that a missing SRT file returns an empty DubbingResult."""
    result = srt_to_audio_numpy(
        srt_path="/nonexistent/file.srt",
        output_path="/tmp/output.wav",
        voice="en-US-JennyNeural",
        log_level="ERROR",
    )
    assert isinstance(result, DubbingResult)
    assert result.total_cues == 0


def test_generation_errors_initialized():
    """Test that generation_errors is always initialized (the known bug fix)."""
    import inspect
    source = inspect.getsource(srt_to_audio_numpy)
    lines = source.split('\n')
    found_init = False
    for line in lines:
        if "generation_errors:" in line and "= {}" in line:
            found_init = True
            break
        if "if segments_to_generate:" in line:
            break
    assert found_init, "generation_errors must be initialized before segments_to_generate block"
