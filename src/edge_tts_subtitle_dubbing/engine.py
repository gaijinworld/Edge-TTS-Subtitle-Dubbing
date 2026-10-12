from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pysrt
import soundfile as sf
from tqdm import tqdm

from .ffmpeg import convert_audio_format, get_media_duration
from .models import CueResult, DubbingResult
from .synthesis import generate_audio_batch
from .text import clean_subtitle_text, parse_duration_str, time_str_to_seconds
from .timing import SAMPLE_RATE, adjust_audio_length

logger = logging.getLogger(__name__)

OVERLAP_THRESHOLD = 0.01
MIN_SEGMENT_DURATION = 0.05
FINAL_PADDING_THRESHOLD = 0.01
EXCESS_AUDIO_WARNING_THRESHOLD = 1.0


def _setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> None:
    import sys
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.handlers.clear()
    logger.setLevel(numeric_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def srt_to_audio_numpy(
    srt_path: str,
    output_path: str,
    voice: str,
    temp_dir: Optional[str] = None,
    keep_temp: bool = False,
    resume: bool = False,
    ref_video: Optional[str] = None,
    expected_duration: Optional[str] = None,
    max_speed: float = 1.5,
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    no_concat: bool = False,
    batch_size: int = 10,
    retries: int = 10,
    enable_exact_duration_text_stretching: bool = False,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
) -> DubbingResult:
    """Convert SRT subtitles to synchronized audio using Edge TTS.

    Returns a DubbingResult with processing statistics and per-cue details.
    """
    _setup_logging(log_file, log_level)

    result = DubbingResult(output_path=output_path)

    if not os.path.exists(srt_path):
        logger.error(f"SRT file not found: {srt_path}")
        return result

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return result

    if not temp_dir:
        with open(srt_path, 'rb') as f:
            file_content = f.read()
        h = hashlib.md5(file_content).hexdigest()
        temp_dir = os.path.join(os.getcwd(), "temp", h)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    logger.info(f"Working in temp ID: {os.path.basename(temp_dir)}")

    try:
        subs = pysrt.open(srt_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            subs = pysrt.open(srt_path, encoding='latin-1')
            logger.warning("SRT file parsed with latin-1 encoding")
        except Exception as e:
            logger.error(f"Failed to parse SRT file with alternative encoding: {e}")
            return result
    except Exception as e:
        logger.error(f"Failed to parse SRT file: {e}")
        return result

    if not subs or len(subs) == 0:
        logger.error("SRT file is empty or contains no valid subtitles")
        return result

    logger.info(f"Loaded {len(subs)} subtitle entries")
    if enable_exact_duration_text_stretching:
        logger.info("Exact-duration text stretching: ON (strict slot + librosa fallback)")
    else:
        logger.info("Exact-duration text stretching: OFF (upstream stretch / raw-on-fail)")

    audio_segments: list[np.ndarray] = []
    current_total_samples = 0

    final_target_sec = 0.0
    if expected_duration:
        final_target_sec = parse_duration_str(expected_duration)
    elif ref_video:
        final_target_sec = get_media_duration(ref_video, ffprobe_path=ffprobe_path)

    logger.info(f"Final Target Duration: {final_target_sec}s")
    logger.info(f"Starting processing of {len(subs)} subtitle segments...")
    logger.info("=" * 60)

    cache_dir = os.path.join(temp_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    stats = {
        'total': len(subs),
        'generated': 0,
        'resumed': 0,
        'cached': 0,
        'failed': 0,
        'empty': 0,
        'overlaps': 0,
        'late_starts': 0,
    }

    segments_to_generate: list[tuple[str, str, int]] = []
    text_to_cache: dict[str, str] = {}

    for i, sub in enumerate(subs):
        text = clean_subtitle_text(sub.text)
        if not text:
            stats['empty'] += 1
            continue

        raw_audio_path = os.path.join(temp_dir, f"raw_{i}.mp3")
        exists = os.path.exists(raw_audio_path) and os.path.getsize(raw_audio_path) > 0

        if resume and exists:
            stats['resumed'] += 1
        else:
            text_hash = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
            cache_path = os.path.join(cache_dir, f"cache_{text_hash}.mp3")

            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                try:
                    shutil.copy2(cache_path, raw_audio_path)
                    stats['cached'] += 1
                    logger.debug(f"Using cached audio for segment {i} (hash: {text_hash[:8]}...)")
                except Exception as e:
                    logger.warning(f"Failed to copy cache for segment {i}: {e}")
                    if text not in text_to_cache:
                        text_to_cache[text] = cache_path
                        segments_to_generate.append((text, cache_path, i))
            else:
                if text not in text_to_cache:
                    text_to_cache[text] = cache_path
                    segments_to_generate.append((text, cache_path, i))

    # FIX: Always initialize generation_errors before any branch that might use it
    generation_errors: dict[int, Optional[Exception]] = {}

    if segments_to_generate:
        unique_count = len(segments_to_generate)
        total_needed = sum(
            1 for i, sub in enumerate(subs)
            if not (resume and os.path.exists(os.path.join(temp_dir, f"raw_{i}.mp3")))
        )
        deduped = total_needed - unique_count

        if deduped > 0:
            logger.info(f"Deduplicated {deduped} identical text segments")

        logger.info(f"Generating {unique_count} unique audio segments (batch size: {batch_size})...")
        generation_errors = asyncio.run(
            generate_audio_batch(segments_to_generate, voice, batch_size, retries)
        )

        for text, cache_path, idx in segments_to_generate:
            raw_audio_path = os.path.join(temp_dir, f"raw_{idx}.mp3")

            if idx in generation_errors and generation_errors[idx] is None:
                try:
                    if os.path.exists(cache_path):
                        shutil.copy2(cache_path, raw_audio_path)
                    stats['generated'] += 1
                except Exception as e:
                    logger.error(f"Failed to copy generated cache for segment {idx}: {e}")
                    stats['failed'] += 1
            else:
                stats['failed'] += 1
                if idx in generation_errors:
                    logger.error(f"Failed to generate audio for segment {idx}: {generation_errors[idx]}")

    logger.info("Copying cached audio to segment locations...")
    for i, sub in enumerate(subs):
        text = clean_subtitle_text(sub.text)
        if not text:
            continue

        raw_audio_path = os.path.join(temp_dir, f"raw_{i}.mp3")

        if os.path.exists(raw_audio_path) and os.path.getsize(raw_audio_path) > 0:
            continue

        text_hash = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, f"cache_{text_hash}.mp3")

        if os.path.exists(cache_path):
            try:
                shutil.copy2(cache_path, raw_audio_path)
            except Exception as e:
                logger.error(f"Failed to copy cache to segment {i}: {e}")

    logger.info("Processing and synchronizing segments...")

    for i, sub in enumerate(tqdm(subs, desc="Processing", unit="sub")):
        start_sec = time_str_to_seconds(sub.start)
        end_sec = time_str_to_seconds(sub.end)
        target_span_sec = end_sec - start_sec
        text = clean_subtitle_text(sub.text)

        cue = CueResult(index=i, text=text, start_sec=start_sec, end_sec=end_sec)

        current_head_sec = current_total_samples / SAMPLE_RATE
        gap_sec = start_sec - current_head_sec

        if gap_sec > 0:
            gap_samples = int(gap_sec * SAMPLE_RATE)
            silence_chunk = np.zeros(gap_samples, dtype=np.float32)
            audio_segments.append(silence_chunk)
            current_total_samples += gap_samples
        elif gap_sec < -OVERLAP_THRESHOLD:
            stats['overlaps'] += 1
            logger.warning(f"Overlap at sub {i+1}: Head {current_head_sec:.2f}s > Start {start_sec:.2f}s")
            cue.status = "overlap"

        if not text:
            stats['empty'] += 1
            cue.status = "empty"
            current_head_sec = current_total_samples / SAMPLE_RATE
            needed_sec = end_sec - current_head_sec
            if needed_sec > 0:
                needed_samples = int(needed_sec * SAMPLE_RATE)
                audio_segments.append(np.zeros(needed_samples, dtype=np.float32))
                current_total_samples += needed_samples
            result.cues.append(cue)
            continue

        raw_audio_path = os.path.join(temp_dir, f"raw_{i}.mp3")

        if i in generation_errors and generation_errors[i] is not None:
            cue.status = "failed"
            cue.error = str(generation_errors[i])
            logger.debug(f"Using silence fallback for segment {i+1}")
            needed_samples = int(target_span_sec * SAMPLE_RATE)
            audio_segments.append(np.zeros(needed_samples, dtype=np.float32))
            current_total_samples += needed_samples
            result.cues.append(cue)
            continue

        current_head_sec = current_total_samples / SAMPLE_RATE
        target_dur_for_segment = end_sec - current_head_sec

        if target_dur_for_segment < MIN_SEGMENT_DURATION:
            stats['late_starts'] += 1
            cue.status = "late_start"
            logger.warning(f"  Sub {i+1} starts late. Forcing max speed catch-up.")
            target_dur_for_segment = MIN_SEGMENT_DURATION

        stretched_wav = adjust_audio_length(
            raw_audio_path,
            target_dur_for_segment,
            sample_rate=SAMPLE_RATE,
            max_speed_factor=max_speed,
            enable_exact_duration_text_stretching=enable_exact_duration_text_stretching,
        )

        audio_segments.append(stretched_wav)
        current_total_samples += len(stretched_wav)
        result.cues.append(cue)

    if final_target_sec > 0:
        current_len_sec = current_total_samples / SAMPLE_RATE
        missing = final_target_sec - current_len_sec

        if missing > FINAL_PADDING_THRESHOLD:
            logger.info(f"Adding final padding: {missing:.2f}s")
            pad_samples = int(missing * SAMPLE_RATE)
            audio_segments.append(np.zeros(pad_samples, dtype=np.float32))
            current_total_samples += pad_samples
        elif missing < -EXCESS_AUDIO_WARNING_THRESHOLD:
            logger.warning(f"Total audio ({current_len_sec:.2f}s) exceeds target ({final_target_sec:.2f}s)!")

    logger.info("=" * 60)
    logger.info("Processing Summary:")
    logger.info(f"  Total segments: {stats['total']}")
    logger.info(f"  Generated: {stats['generated']}")
    logger.info(f"  Cached (text reuse): {stats['cached']}")
    logger.info(f"  Resumed: {stats['resumed']}")
    logger.info(f"  Empty subtitles: {stats['empty']}")
    logger.info(f"  Failed (using silence): {stats['failed']}")
    if stats['overlaps'] > 0:
        logger.info(f"  Overlaps detected: {stats['overlaps']}")
    if stats['late_starts'] > 0:
        logger.info(f"  Late starts (speed-up): {stats['late_starts']}")

    final_audio_duration = current_total_samples / SAMPLE_RATE
    logger.info(f"  Output audio duration: {final_audio_duration:.2f}s")
    if final_target_sec > 0:
        accuracy = (final_audio_duration / final_target_sec * 100) if final_target_sec > 0 else 0
        logger.info(f"  Target match accuracy: {accuracy:.2f}%")
    logger.info("=" * 60)

    if no_concat:
        logger.info("Skipping concatenation (--no-concat mode)")
        logger.info(f"Individual segments saved in: {temp_dir}")
        logger.info("Done.")
        result.total_cues = stats['total']
        result.generated = stats['generated']
        result.cached = stats['cached']
        result.resumed = stats['resumed']
        result.failed = stats['failed']
        result.empty = stats['empty']
        result.overlaps = stats['overlaps']
        result.late_starts = stats['late_starts']
        result.final_duration_sec = final_audio_duration
        result.target_duration_sec = final_target_sec
        result.stats = stats
        return result

    logger.info("Concatenating segments...")
    if audio_segments:
        full_wav = np.concatenate(audio_segments)
    else:
        full_wav = np.zeros(0, dtype=np.float32)

    try:
        output_ext = Path(output_path).suffix.lower().lstrip('.')
    except Exception:
        output_ext = 'wav'

    if output_ext in ['m4a', 'opus']:
        logger.info(f"Saving temporary WAV for conversion to {output_ext}...")
        temp_output_wav = os.path.join(temp_dir, "final_output_temp.wav")
        sf.write(temp_output_wav, full_wav, SAMPLE_RATE)

        logger.info(f"Converting to {output_ext}...")
        if convert_audio_format(temp_output_wav, output_path, output_ext, ffmpeg_path=ffmpeg_path):
            logger.info("Done.")
        else:
            logger.error("Conversion failed.")

        if os.path.exists(temp_output_wav):
            try:
                os.remove(temp_output_wav)
            except Exception:
                pass
    else:
        logger.info(f"Saving final output to {output_path} ...")
        sf.write(output_path, full_wav, SAMPLE_RATE)
        logger.info("Done.")

    if not keep_temp and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info("Cleaned temp (including cache).")
    elif keep_temp:
        cache_dir_check = os.path.join(temp_dir, "cache")
        if os.path.exists(cache_dir_check):
            cache_count = len([f for f in os.listdir(cache_dir_check) if f.startswith('cache_')])
            logger.info(f"Cache preserved: {cache_count} unique text segments in {cache_dir_check}")

    result.total_cues = stats['total']
    result.generated = stats['generated']
    result.cached = stats['cached']
    result.resumed = stats['resumed']
    result.failed = stats['failed']
    result.empty = stats['empty']
    result.overlaps = stats['overlaps']
    result.late_starts = stats['late_starts']
    result.final_duration_sec = final_audio_duration
    result.target_duration_sec = final_target_sec
    result.stats = stats
    return result
