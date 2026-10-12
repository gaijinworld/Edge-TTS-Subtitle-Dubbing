from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def time_str_to_seconds(t_obj) -> float:
    """Convert pysrt SubRipTime object to seconds."""
    return t_obj.hours * 3600 + t_obj.minutes * 60 + t_obj.seconds + t_obj.milliseconds / 1000.0


def clean_subtitle_text(text: str) -> str:
    """Clean subtitle text: replace newlines with spaces, strip, and convert '...' to empty."""
    cleaned = text.replace('\n', ' ').strip()
    if cleaned == "...":
        cleaned = ""
    return cleaned


def parse_duration_str(duration_str: str | None) -> float:
    """Parse duration string (HH:MM:SS, MM:SS, or seconds) to float seconds."""
    if not duration_str:
        return 0.0
    try:
        if ":" in duration_str:
            parts = duration_str.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
        else:
            return float(duration_str)
    except Exception as e:
        logger.error(f"Invalid duration format: {duration_str}. Use HH:MM:SS or seconds.")
        return 0.0
    return 0.0
