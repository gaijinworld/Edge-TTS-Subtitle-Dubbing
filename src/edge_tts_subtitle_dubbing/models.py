from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DubbingOptions:
    voice: str = "en-US-JennyNeural"
    max_speed: float = 1.5
    exact_duration: bool = False
    output_format: str = "wav"
    batch_size: int = 10
    retries: int = 10
    keep_temp: bool = False
    resume: bool = False
    no_concat: bool = False
    log_level: str = "INFO"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"


@dataclass
class CueResult:
    index: int
    text: str
    start_sec: float
    end_sec: float
    status: str = "ok"
    error: str | None = None


@dataclass
class DubbingResult:
    output_path: str
    total_cues: int = 0
    generated: int = 0
    cached: int = 0
    resumed: int = 0
    failed: int = 0
    empty: int = 0
    overlaps: int = 0
    late_starts: int = 0
    final_duration_sec: float = 0.0
    target_duration_sec: float = 0.0
    cues: list[CueResult] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
