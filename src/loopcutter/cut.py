"""Loop extraction.

Two things stop loops clicking at the seam:

1. Snapping the start to a nearby zero crossing, so the waveform begins near
   silence rather than mid-swing.
2. A sub-millisecond fade at each edge to catch whatever the snap missed.

The snap moves the start, so the end moves with it by exactly the same amount.
The length in samples is never recomputed - that would reintroduce drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .manifest import LoopSpec
from .timing import resolve_window

DEFAULT_SNAP_MS = 2.0
DEFAULT_FADE_MS = 0.5


@dataclass
class CutResult:
    spec: LoopSpec
    output: Path
    sample_rate: int
    length_samples: int
    snap_offset_samples: int
    peak: float


def _find_zero_crossing(audio: np.ndarray, centre: int, radius: int) -> int:
    """Nearest sample to `centre` where the summed waveform crosses zero.

    Searches outward so the closest candidate wins. Returns `centre` unchanged
    if nothing is found inside the radius.
    """
    if radius <= 0:
        return centre

    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    lo = max(1, centre - radius)
    hi = min(len(mono) - 1, centre + radius)
    if lo >= hi:
        return centre

    for offset in range(0, hi - lo + 1):
        for candidate in (centre + offset, centre - offset):
            if lo <= candidate <= hi:
                if mono[candidate - 1] <= 0.0 <= mono[candidate]:
                    return candidate
                if mono[candidate - 1] >= 0.0 >= mono[candidate]:
                    return candidate
    return centre


def _apply_edge_fades(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0 or len(audio) < fade_samples * 2:
        return audio

    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=out.dtype)
    if out.ndim > 1:
        ramp = ramp[:, None]
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out


def cut_loop(
    spec: LoopSpec,
    out_dir: str | Path,
    fmt: str = "aiff",
    subtype: str = "PCM_24",
    snap_ms: float = DEFAULT_SNAP_MS,
    fade_ms: float = DEFAULT_FADE_MS,
    filename: str | None = None,
) -> CutResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not spec.source.exists():
        raise FileNotFoundError(f"row {spec.row_number}: source not found: {spec.source}")

    info = sf.info(str(spec.source))
    window = resolve_window(
        spec.start_seconds, spec.bars, spec.bpm, info.samplerate, spec.beats_per_bar
    )

    snap_radius = int(round(snap_ms / 1000.0 * info.samplerate))
    read_start = max(0, window.start_sample - snap_radius)
    read_stop = min(info.frames, window.end_sample + snap_radius + 1)

    if read_start >= info.frames:
        raise ValueError(
            f"row {spec.row_number}: start {spec.start_seconds:.3f}s is past the end of "
            f"{spec.source.name} ({info.duration:.3f}s)"
        )

    block, _ = sf.read(
        str(spec.source), start=read_start, stop=read_stop, dtype="float32", always_2d=True
    )

    centre = window.start_sample - read_start
    snapped = _find_zero_crossing(block, centre, snap_radius)
    offset = snapped - centre

    if snapped + window.length_samples > len(block):
        # Not enough audio after the snap; fall back to the unsnapped start.
        snapped, offset = centre, 0

    if snapped + window.length_samples > len(block):
        raise ValueError(
            f"row {spec.row_number}: {spec.source.name} is too short for a "
            f"{spec.bars}-bar loop at {spec.bpm} BPM starting at {spec.start_seconds:.3f}s"
        )

    audio = block[snapped : snapped + window.length_samples]
    fade_samples = int(round(fade_ms / 1000.0 * info.samplerate))
    audio = _apply_edge_fades(audio, fade_samples)

    name = filename or f"{spec.slug}.{fmt}"
    destination = out_dir / name
    sf.write(str(destination), audio, info.samplerate,
             format=fmt.upper(), subtype=subtype)

    return CutResult(
        spec=spec,
        output=destination,
        sample_rate=info.samplerate,
        length_samples=len(audio),
        snap_offset_samples=offset,
        peak=float(np.max(np.abs(audio))) if len(audio) else 0.0,
    )
