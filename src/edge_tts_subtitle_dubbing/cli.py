from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .engine import srt_to_audio_numpy


def _print_banner():
    banner = """
╔═══════════════════════════════════════════════════╗
║             Edge TTS Subtitle Dubbing             ║
║  github.com/fr0stb1rd/Edge-TTS-Subtitle-Dubbing   ║
║           License: MIT © 2026 fr0stb1rd           ║
╚═══════════════════════════════════════════════════╝
    """
    print(banner)


def main(argv: list[str] | None = None) -> int:
    _print_banner()

    parser = argparse.ArgumentParser(description="SRT to Audio (Numpy/Librosa Precise Sync)")
    parser.add_argument("srt_file")
    parser.add_argument("output_file")
    parser.add_argument("--voice", default="en-US-JennyNeural")
    parser.add_argument("--temp", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ref_video", default=None)
    parser.add_argument("--expected_duration", default=None)
    parser.add_argument("--max_speed", type=float, default=1.5)
    parser.add_argument("--log_file", default=None, help="Path to log file.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--no-concat", action="store_true", help="Generate segments only, skip final merge")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of concurrent TTS requests")
    parser.add_argument("--retries", type=int, default=10, help="Number of retry attempts")
    parser.add_argument("--format", choices=['wav', 'm4a', 'opus'], default=None, help="Output format override")
    parser.add_argument(
        "--enable-exact-duration-text-stretching",
        action="store_true",
        help="Force each TTS segment to exact cue length; librosa fallback if audiostretchy fails.",
    )

    args = parser.parse_args(argv)

    log_file = args.log_file
    if log_file is None:
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        output_basename = os.path.splitext(os.path.basename(args.output_file))[0]
        log_file = os.path.join(output_dir, f"{output_basename}.log")

    final_output = args.output_file
    if args.format:
        path = Path(final_output)
        if path.suffix.lower().lstrip('.') != args.format:
            final_output = f"{final_output}.{args.format}"
            print(f"Output filename changed to match format: {final_output}")

    srt_to_audio_numpy(
        args.srt_file,
        final_output,
        args.voice,
        temp_dir=args.temp,
        keep_temp=args.keep_temp,
        resume=args.resume,
        ref_video=args.ref_video,
        expected_duration=args.expected_duration,
        max_speed=args.max_speed,
        log_file=log_file,
        log_level=args.log_level,
        no_concat=args.no_concat,
        batch_size=args.batch_size,
        retries=args.retries,
        enable_exact_duration_text_stretching=args.enable_exact_duration_text_stretching,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
