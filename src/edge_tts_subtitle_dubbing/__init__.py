"""edge-tts-subtitle-dubbing: Convert SRT subtitles to synchronized audio using Edge TTS."""

from .engine import srt_to_audio_numpy
from .models import CueResult, DubbingOptions, DubbingResult
from .exceptions import DubbingError, SynthesisError

__all__ = [
    "srt_to_audio_numpy",
    "DubbingOptions",
    "DubbingResult",
    "CueResult",
    "DubbingError",
    "SynthesisError",
]

__version__ = "1.0.0"
