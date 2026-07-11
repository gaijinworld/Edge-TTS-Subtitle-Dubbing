from __future__ import annotations

import logging
import os
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
from audiostretchy.stretch import stretch_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


def _exact_sample_length(
    y: npt.NDArray[np.float32], target_samples: int
) -> npt.NDArray[np.float32]:
    """Trim or zero-pad to an exact sample count (keeps the timeline aligned)."""
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(y) < target_samples:
        return np.pad(y, (0, target_samples - len(y)), mode="constant").astype(np.float32)
    return y[:target_samples].astype(np.float32)


def _adjust_audio_length_upstream_compat(
    wav_path: str,
    desired_length: float,
    sample_rate: int = SAMPLE_RATE,
    max_speed_factor: float = 1.5,
) -> npt.NDArray[np.float32]:
    """Adjusts audio file to desired length using audiostretchy time-stretching (legacy upstream)."""
    try:
        y, sr = librosa.load(wav_path, sr=sample_rate)
    except Exception as e:
        logger.error(f"Failed to load {wav_path}: {e}")
        return np.zeros(0, dtype=np.float32)

    current_length = len(y) / sample_rate

    if desired_length <= 0:
        return y

    ratio = desired_length / current_length

    min_ratio = 1.0 / max_speed_factor
    if ratio < min_ratio:
        ratio = min_ratio
        logger.warning(
            f"  Clamped max speed: Ratio {ratio:.3f} "
            f"(Req: {desired_length:.2f}s from {current_length:.2f}s)"
        )

    input_tmp = wav_path
    output_tmp = wav_path.replace(".mp3", "_stretched.wav").replace(".wav", "_stretched.wav")

    try:
        stretch_audio(input_tmp, output_tmp, ratio=ratio, sample_rate=sample_rate)
        y_stretched, _ = librosa.load(output_tmp, sr=sample_rate)

        target_samples = int(desired_length * sample_rate)

        if len(y_stretched) < target_samples:
            padding = target_samples - len(y_stretched)
            y_stretched = np.pad(y_stretched, (0, padding), "constant")
        elif len(y_stretched) > target_samples:
            y_stretched = y_stretched[:target_samples]

        return y_stretched

    except Exception as e:
        logger.error(f"Stretching failed: {e}")
        return y
    finally:
        if os.path.exists(output_tmp):
            try:
                os.remove(output_tmp)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {output_tmp}: {e}")


def _adjust_audio_length_exact_slot(
    wav_path: str,
    desired_length: float,
    sample_rate: int = SAMPLE_RATE,
    max_speed_factor: float = 1.5,
) -> npt.NDArray[np.float32]:
    """Adjusts audio to cue slot length: audiostretchy first, then strict trim/pad."""
    target_samples = int(desired_length * sample_rate)

    try:
        y, _sr = librosa.load(wav_path, sr=sample_rate)
    except Exception as e:
        logger.error(f"Failed to load {wav_path}: {e}")
        return _exact_sample_length(np.zeros(0, dtype=np.float32), target_samples)

    current_length = len(y) / sample_rate

    if desired_length <= 0:
        return np.zeros(0, dtype=np.float32)

    ratio = desired_length / current_length

    min_ratio = 1.0 / max_speed_factor
    if ratio < min_ratio:
        ratio = min_ratio
        logger.warning(
            f"  Clamped max speed: Ratio {ratio:.3f} "
            f"(Req: {desired_length:.2f}s from {current_length:.2f}s)"
        )

    stem = Path(wav_path).stem
    parent = Path(wav_path).parent
    output_tmp = str(parent / f"{stem}_audiostretch.wav")

    def fallback_librosa_time_stretch() -> npt.NDArray[np.float32]:
        rate = current_length / desired_length
        rate = float(min(max(rate, 1e-6), max_speed_factor))
        try:
            y2 = librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=rate)
            return _exact_sample_length(y2.astype(np.float32), target_samples)
        except Exception as e2:
            logger.error(f"librosa.effects.time_stretch fallback failed: {e2}")
            return np.zeros(target_samples, dtype=np.float32)

    try:
        stretch_audio(wav_path, output_tmp, ratio=ratio, sample_rate=sample_rate)
        y_stretched, _ = librosa.load(output_tmp, sr=sample_rate)
        return _exact_sample_length(y_stretched.astype(np.float32), target_samples)
    except Exception as e:
        logger.error(f"audiostretchy failed ({e}); using librosa time_stretch fallback")
        return fallback_librosa_time_stretch()
    finally:
        if os.path.exists(output_tmp):
            try:
                os.remove(output_tmp)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {output_tmp}: {e}")


def adjust_audio_length(
    wav_path: str,
    desired_length: float,
    sample_rate: int = SAMPLE_RATE,
    max_speed_factor: float = 1.5,
    *,
    enable_exact_duration_text_stretching: bool = False,
) -> npt.NDArray[np.float32]:
    """Dispatch: legacy upstream stretch behavior vs exact-slot + librosa fallback."""
    if enable_exact_duration_text_stretching:
        return _adjust_audio_length_exact_slot(
            wav_path, desired_length, sample_rate, max_speed_factor
        )
    return _adjust_audio_length_upstream_compat(
        wav_path, desired_length, sample_rate, max_speed_factor
    )
