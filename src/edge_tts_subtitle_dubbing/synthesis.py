from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

import edge_tts

logger = logging.getLogger(__name__)


async def generate_audio_segment(
    text: str,
    output_file: str,
    voice: str,
    rate: str = "+0%",
    retries: int = 10,
) -> None:
    """Generate audio for a single text segment using Edge TTS with retry logic."""
    for i in range(retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(output_file)
            return
        except Exception as e:
            if i < retries:
                wait_time = (i + 1) * random.uniform(1, 3)
                logger.warning(
                    f"Connection error for TTS ('{text}'), retrying in {wait_time:.1f}s... "
                    f"({i+1}/{retries}) | Error: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to generate audio for ('{text}') after {retries} retries: {e}")
                raise e


async def generate_audio_batch(
    segments_info: list[tuple[str, str, int]],
    voice: str,
    batch_size: int = 10,
    retries: int = 10,
) -> dict[int, Optional[Exception]]:
    """Generate multiple audio segments in parallel using async batch processing."""
    async def generate_one(text: str, output_path: str, index: int) -> tuple[int, Optional[Exception]]:
        try:
            await generate_audio_segment(text, output_path, voice, retries=retries)
            return (index, None)
        except Exception as e:
            return (index, e)

    all_results: dict[int, Optional[Exception]] = {}

    for i in range(0, len(segments_info), batch_size):
        batch = segments_info[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(segments_info) + batch_size - 1) // batch_size

        logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} segments)...")

        tasks = [generate_one(text, path, idx) for text, path, idx in batch]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for idx, error in results:
            all_results[idx] = error

    return all_results
