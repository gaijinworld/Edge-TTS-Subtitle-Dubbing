from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_FFPROBE = "ffprobe"
DEFAULT_FFMPEG = "ffmpeg"


def get_media_duration(file_path: Optional[str], ffprobe_path: str = DEFAULT_FFPROBE) -> float:
    """Get duration of a media file using ffprobe."""
    if not file_path:
        logger.warning("No file path provided for duration check")
        return 0.0
    if not os.path.exists(file_path):
        logger.error(f"Media file not found: {file_path}")
        return 0.0
    cmd = [
        ffprobe_path, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.debug(f"Media duration for {os.path.basename(file_path)}: {duration:.2f}s")
        return duration
    except FileNotFoundError:
        logger.error("ffprobe not found. Please install FFmpeg and ensure it's in PATH.")
        return 0.0
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {file_path}: {e.stderr}")
        return 0.0
    except ValueError as e:
        logger.error(f"Invalid duration value from ffprobe: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error reading duration for {file_path}: {e}")
        return 0.0


def convert_audio_format(input_path: str, output_path: str, format: str, ffmpeg_path: str = DEFAULT_FFMPEG) -> bool:
    """Convert audio file to specified format using ffmpeg."""
    try:
        if format == 'm4a':
            cmd = [ffmpeg_path, "-i", input_path, "-c:a", "aac", "-b:a", "192k", "-y", output_path]
        elif format == 'opus':
            cmd = [ffmpeg_path, "-i", input_path, "-c:a", "libopus", "-b:a", "128k", "-y", output_path]
        else:
            cmd = [ffmpeg_path, "-i", input_path, "-y", output_path]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.error(f"Failed to convert audio to {format}: {e}")
        return False
