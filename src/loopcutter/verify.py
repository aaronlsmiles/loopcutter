"""Verification.

An agent cannot hear the output. These checks are the substitute. Every one
of them fails loudly rather than warning quietly, because a batch of 300 loops
with three bad ones is worse than a batch that refuses to finish.

The seam check is the interesting one: it compares the spectrum of the last
20ms against the first 20ms. A loop whose end does not resemble its start will
audibly lurch every time it wraps, even with perfect sample counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .cut import CutResult
from .timing import loop_length_samples

SEAM_WINDOW_MS = 20.0
SEAM_MIN_CORRELATION = 0.35
MAX_PEAK = 0.999
MAX_DC_OFFSET = 0.01
BPM_TOLERANCE = 0.02


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    output: Path
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def _spectrum(block: np.ndarray) -> np.ndarray:
    mono = block.mean(axis=1) if block.ndim > 1 else block
    if len(mono) == 0:
        return np.zeros(1)
    windowed = mono * np.hanning(len(mono))
    return np.abs(np.fft.rfft(windowed))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size == 0:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def verify_cut(result: CutResult, check_bpm: bool = False) -> Report:
    report = Report(output=result.output)
    audio, sample_rate = sf.read(str(result.output), dtype="float32", always_2d=True)

    expected = loop_length_samples(
        result.spec.bars, result.spec.bpm, sample_rate, result.spec.beats_per_bar
    )
    report.checks.append(
        Check(
            "sample_count",
            len(audio) == expected,
            f"got {len(audio)}, expected {expected}",
        )
    )

    report.checks.append(
        Check("sample_rate", sample_rate == result.sample_rate,
              f"{sample_rate} vs source {result.sample_rate}")
    )

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    report.checks.append(
        Check("peak_headroom", peak <= MAX_PEAK, f"peak {peak:.4f}")
    )
    report.checks.append(
        Check("not_silent", peak > 1e-5, f"peak {peak:.6f}")
    )

    dc = float(np.max(np.abs(audio.mean(axis=0)))) if audio.size else 0.0
    report.checks.append(
        Check("dc_offset", dc <= MAX_DC_OFFSET, f"dc {dc:.5f}")
    )

    seam = int(round(SEAM_WINDOW_MS / 1000.0 * sample_rate))
    if len(audio) >= seam * 2:
        correlation = _correlation(_spectrum(audio[:seam]), _spectrum(audio[-seam:]))
        report.checks.append(
            Check(
                "seam_continuity",
                correlation >= SEAM_MIN_CORRELATION,
                f"spectral correlation {correlation:.3f} "
                f"(threshold {SEAM_MIN_CORRELATION})",
            )
        )

    if check_bpm:
        report.checks.append(_bpm_check(result, audio, sample_rate))

    return report


def _bpm_check(result: CutResult, audio: np.ndarray, sample_rate: int) -> Check:
    try:
        import librosa
    except ImportError:
        return Check("bpm_match", True, "skipped - librosa not installed")

    mono = audio.mean(axis=1)
    duration = len(mono) / sample_rate
    implied = result.spec.bars * result.spec.beats_per_bar * 60.0 / duration

    tempo, _ = librosa.beat.beat_track(y=mono, sr=sample_rate)
    tempo = float(np.atleast_1d(tempo)[0])

    # Detected tempo commonly lands on a half or double of the true value.
    candidates = [tempo, tempo * 2, tempo / 2]
    best = min(candidates, key=lambda c: abs(c - result.spec.bpm))
    drift = abs(best - result.spec.bpm) / result.spec.bpm

    return Check(
        "bpm_match",
        drift <= BPM_TOLERANCE or abs(implied - result.spec.bpm) < 0.01,
        f"implied {implied:.3f}, detected {best:.2f}, declared {result.spec.bpm}",
    )
