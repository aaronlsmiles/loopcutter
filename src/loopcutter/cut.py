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
    start_sample: int
    source_peak: float
    trim_db: float = 0.0
    timing_source: Path | None = None     # the full mix, when this loop was cut from a stem


def _find_zero_crossing(audio: np.ndarray, centre: int, radius: int,
                        direction: str = "both") -> int:
    """Nearest sample to `centre` where the summed waveform crosses zero.

    `direction="backward"` only looks earlier. Returns `centre` unchanged if
    nothing is found inside the radius.
    """
    if radius <= 0:
        return centre
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    lo = max(1, centre - radius)
    hi = min(len(mono) - 1, centre + radius)
    if lo >= hi:
        return centre
    for offset in range(0, hi - lo + 1):
        candidates = (centre - offset,) if direction == "backward" else (centre + offset, centre - offset)
        for candidate in candidates:
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
    trim_db: float = 0.0,
    xfade_ms: float = 0.0,
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
    xfade = int(round(xfade_ms / 1000.0 * info.samplerate))
    read_start = max(0, window.start_sample - snap_radius - xfade)
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
    snapped = _find_zero_crossing(block, centre, snap_radius, direction="backward")
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
    pre = block[snapped - xfade : snapped] if xfade and snapped >= xfade else None
    source_peak = max(float(np.max(np.abs(audio))) if len(audio) else 0.0,
                      float(np.max(np.abs(pre))) if pre is not None else 0.0)
    if pre is not None:
        # The tail fades into the audio just before the start, so the wrap plays
        # exactly what the source played there: no dip, no step.
        ramp = np.linspace(0.0, 1.0, xfade, dtype=audio.dtype)[:, None]
        audio = audio.copy()
        audio[-xfade:] = audio[-xfade:] * (1.0 - ramp) + pre * ramp
    else:
        audio = _apply_edge_fades(audio, int(round(fade_ms / 1000.0 * info.samplerate)))
    if trim_db:
        audio = np.clip(audio * 10.0 ** (trim_db / 20.0), -1.0, 1.0)

    destination = out_dir / (filename or f"{spec.slug}.{fmt}")
    sf.write(str(destination), audio, info.samplerate, format=fmt.upper(), subtype=subtype)

    return CutResult(
        spec=spec, output=destination, sample_rate=info.samplerate,
        length_samples=len(audio), snap_offset_samples=offset,
        peak=float(np.max(np.abs(audio))) if len(audio) else 0.0,
        start_sample=read_start + snapped, source_peak=source_peak, trim_db=trim_db,
    )


def cut_oneshot(spec: LoopSpec, out_dir, fmt: str = "aiff", subtype: str = "PCM_24",
                snap_ms: float = DEFAULT_SNAP_MS, fade_out_ms: float = 10.0,
                filename: str | None = None, trim_db: float = 0.0) -> CutResult:
    """A single hit or phrase from start to end: snapped backward to a zero crossing,
    a sub-millisecond fade in, a short fade out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not spec.source.exists():
        raise FileNotFoundError(f"row {spec.row_number}: source not found: {spec.source}")
    info = sf.info(str(spec.source))
    start = int(round(spec.start_seconds * info.samplerate))
    end = min(info.frames, int(round(spec.end_seconds * info.samplerate)))
    radius = int(round(snap_ms / 1000.0 * info.samplerate))
    read_start = max(0, start - radius)
    block, _ = sf.read(str(spec.source), start=read_start, stop=end, dtype="float32", always_2d=True)
    snapped = _find_zero_crossing(block, start - read_start, radius, direction="backward")
    audio = block[snapped:].copy()
    source_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    fade_in = int(round(DEFAULT_FADE_MS / 1000.0 * info.samplerate))
    fade_out = min(len(audio) // 2, int(round(fade_out_ms / 1000.0 * info.samplerate)))
    audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=audio.dtype)[:, None]
    audio[len(audio) - fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=audio.dtype)[:, None]
    if trim_db:
        audio = np.clip(audio * 10.0 ** (trim_db / 20.0), -1.0, 1.0)
    destination = out_dir / (filename or f"{spec.slug}.{fmt}")
    sf.write(str(destination), audio, info.samplerate, format=fmt.upper(), subtype=subtype)
    return CutResult(spec=spec, output=destination, sample_rate=info.samplerate,
                     length_samples=len(audio), snap_offset_samples=snapped - (start - read_start),
                     peak=float(np.max(np.abs(audio))) if len(audio) else 0.0,
                     start_sample=read_start + snapped, source_peak=source_peak, trim_db=trim_db)
