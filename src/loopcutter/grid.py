"""Beat grids.

Fit a constant-tempo grid to detected beats, find which beat is the one, then
move the grid onto the audio's own onsets, 2 ms early, because a late start
clips the attack on every wrap.

beat_this reports beats on a 50 fps frame grid, so single intervals are
quantised to 20 ms. The first estimate of the period therefore comes from a
line fitted through the longest unbroken run of beats, not from the median
interval. That run can still hide a stretch the detector followed in triplets
or swing, so the estimate is then refined against every beat at once: the
period, within 5%, at which the beats line up best.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from .model import Grid
from .onsets import attacks_near, detect_onsets

INLIER_S = 0.030
MIN_BEATS = 16
PHASE_WINDOW_S = 0.040
MIN_PHASE_AGREEMENT = 0.35      # 10 of 11 calibration tracks agreed at 0.38-0.76
ONSET_LEAD_S = 0.002
LOCAL_SPAN_BEATS = 64
PERIOD_SEARCH = 0.05

Detector = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


class GridError(ValueError):
    """Raised when the detected beats can't support a grid."""


def nominal_bpm(bpm: float) -> float:
    """Machine-made tracks sit on whole BPMs; keep anything else to 0.01."""
    whole = round(bpm)
    return float(whole) if abs(bpm - whole) < 0.01 else round(bpm, 2)


def _initial_period(beats: np.ndarray) -> float:
    intervals = np.diff(beats)
    rough = float(np.median(intervals))
    breaks = np.nonzero((intervals > 1.5 * rough) | (intervals < 0.5 * rough))[0]
    edges = np.concatenate([[0], breaks + 1, [beats.size]])
    start, stop = max(zip(edges[:-1], edges[1:]), key=lambda e: e[1] - e[0])
    run = beats[start:stop]
    if run.size < 8:
        return rough
    slope, _ = np.polyfit(np.arange(run.size), run, 1)
    return float(slope)


def _coherence(beats: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """How well the beats line up on each candidate period (1 = every beat on the grid)."""
    t = beats - beats[0]
    blocks = np.array_split(periods, max(1, periods.size // 256))     # bounded memory on long files
    return np.concatenate([np.abs(np.exp(2j * np.pi * t[None, :] / b[:, None]).mean(axis=1))
                           for b in blocks])


def _refine_period(beats: np.ndarray, rough: float) -> tuple[float, float]:
    """The period within PERIOD_SEARCH of `rough` at which the most beats line up, and
    the time of one beat on it. Every beat votes, so quantisation, stray detections and
    gaps average out; the narrow search can't land on half or double time."""
    step = 0.2 * rough / (beats[-1] - beats[0])                  # a fifth of the peak's width
    coarse = rough * (1 + np.arange(-PERIOD_SEARCH, PERIOD_SEARCH + step, step))
    best = coarse[np.argmax(_coherence(beats, coarse))]
    fine = best + np.linspace(-1, 1, 41) * step * rough
    period = float(fine[np.argmax(_coherence(beats, fine))])
    angle = np.angle(np.mean(np.exp(2j * np.pi * (beats - beats[0]) / period)))
    return period, float(beats[0] + angle / (2 * np.pi) * period)


def fit_grid(beats, beats_per_bar: int = 4) -> Grid:
    beats = np.sort(np.asarray(beats, dtype=float))
    if beats.size < MIN_BEATS:
        raise GridError(f"only {beats.size} beats detected; a grid needs {MIN_BEATS}")
    slope, intercept = _refine_period(beats, _initial_period(beats))
    keep = np.ones(beats.size, dtype=bool)
    for _ in range(4):
        n = np.round((beats - intercept) / slope)
        keep = np.abs(beats - (intercept + n * slope)) < INLIER_S
        if keep.sum() < MIN_BEATS:
            break
        slope, intercept = np.polyfit(n[keep], beats[keep], 1)
    return Grid(period=float(slope), phase=float(intercept % slope),
                beats_per_bar=beats_per_bar, inlier_ratio=float(keep.mean()))


def bar_phase(grid: Grid, downbeats) -> tuple[int, float]:
    downbeats = np.asarray(downbeats, dtype=float)
    if downbeats.size == 0:
        return 0, 0.0
    n = np.round((downbeats - grid.phase) / grid.period).astype(int)
    on_grid = np.abs(downbeats - (grid.phase + n * grid.period)) < INLIER_S
    if not on_grid.any():
        return 0, 0.0
    counts = np.bincount(n[on_grid] % grid.beats_per_bar, minlength=grid.beats_per_bar)
    best = int(counts.argmax())
    return best, float(counts[best] / counts.sum())


def local_grid(beats, downbeats, t: float, span_beats: int = LOCAL_SPAN_BEATS,
               beats_per_bar: int = 4) -> Grid:
    """A grid fitted only to the beats around time t, for tracks whose tempo moves."""
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if beats.size < MIN_BEATS:
        raise GridError(f"only {beats.size} beats detected; a grid needs {MIN_BEATS}")
    half = span_beats * float(np.median(np.diff(beats))) / 2
    grid = fit_grid(beats[np.abs(beats - t) <= half], beats_per_bar)
    phase, agreement = bar_phase(grid, downbeats[np.abs(downbeats - t) <= half])
    return replace(grid, bar_phase=phase, bar_agreement=agreement)


def attack_phase(mono, sr: int, grid: Grid) -> tuple[float, float]:
    """Where the attack sits against the grid's beats (the strongest onset within 40 ms of
    each beat, strength-weighted median), and the share of beats that agree to within 5 ms."""
    times, strengths = detect_onsets(mono, sr)
    beats = grid.beat_time(np.arange(int((len(mono) / sr - grid.phase) / grid.period)))
    found = attacks_near(times, strengths, beats, PHASE_WINDOW_S)
    if len(found) < 2:
        return 0.0, 0.0
    return found.placement, found.agreement(beats.size)


@lru_cache(maxsize=2)
def _model(device: str):
    from beat_this.inference import Audio2Beats

    return Audio2Beats(checkpoint_path="final0", device=device, dbn=False)


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    beats, downbeats = _model(_device())(np.asarray(mono, dtype=np.float32), sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def analyse_grid(mono: np.ndarray, sr: int, beats_per_bar: int = 4,
                 detector: Detector | None = None) -> tuple[Grid, np.ndarray, np.ndarray]:
    beats, downbeats = (detector or detect)(mono, sr)
    grid = fit_grid(beats, beats_per_bar)
    phase, bar_share = bar_phase(grid, downbeats)
    grid = replace(grid, bar_phase=phase, bar_agreement=bar_share)
    offset, agreement = attack_phase(mono, sr, grid)
    if agreement >= MIN_PHASE_AGREEMENT:
        grid = grid.shifted(offset - ONSET_LEAD_S)
    return replace(grid, phase_agreement=agreement), np.asarray(beats), np.asarray(downbeats)


def compare_grids(grid: Grid, app_beats) -> dict:
    """How a DJ app's beat times sit against ours. app_offset_ms = app minus ours.

    The offset is a circular mean, so a grid half a beat out reads as half a
    beat whichever way rounding would have fallen.
    """
    app = np.asarray(app_beats, dtype=float)
    if app.size < 2:
        return {}
    resultant = np.mean(np.exp(2j * np.pi * (app - grid.phase) / grid.period))
    offset = float(np.angle(resultant)) / (2 * np.pi) * grid.period
    # Across the whole grid, not one interval: rekordbox keeps beat times in whole
    # milliseconds, so single intervals alternate (468 and 469 ms at 128 BPM).
    span = float(app[-1] - app[0])
    beats = max(1, round(span / float(np.median(np.diff(app)))))
    return {"app_bpm": 60.0 * beats / span,
            "app_offset_ms": offset * 1000,
            "half_beat": abs(offset) > 0.4 * grid.period,
            "app_agreement": float(abs(resultant))}       # 1 = every app beat at the same offset


def save_beats(path, beats, downbeats) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, beats=np.asarray(beats), downbeats=np.asarray(downbeats))


def load_beats(path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["beats"], data["downbeats"]
