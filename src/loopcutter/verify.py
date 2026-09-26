"""Verification.

An agent cannot hear the output. These checks are the substitute, and every
one has a test proving it can fail:

- sample_count, sample_rate: the file is exactly the declared length.
- peak_headroom: the loop is no louder than its source, after any trim.
- not_silent, dc_offset.
- beat_alignment: the strongest attack near each of the loop's beats, found in
  the source around the loop. Placement catches a start in the wrong place,
  drift a wrong tempo. Gross errors fail, and fine ones warn.
- bpm_match (automatic when the track has been analysed, or with --check-bpm):
  how far the loop's end lands from where the next bar begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .cut import CutResult
from .onsets import HOP, attacks_near, detect_onsets, onset_envelope
from .timing import loop_length_samples

MAX_DC_OFFSET = 0.01
PEAK_EPS = 1e-4
CONTEXT_S = 0.1
ALIGN_WINDOW_S = 0.060
MIN_AGREEMENT = 0.5                  # below this there is no steady attack to judge (breakdowns, pads)
PASS_PLACEMENT_MS = (-3.0, 8.0)      # an attack before the start means the start cuts into it
FAIL_PLACEMENT_MS = (-10.0, 15.0)
DRIFT_WARN_MS = 4.0
DRIFT_FAIL_MS = 10.0
REFERENCE_END_ERROR_MS = 2.0
MEASURED_END_ERROR_MS = 5.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    warn: bool = False


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

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.passed and c.warn]


def _librosa_available() -> bool:
    try:
        import librosa  # noqa: F401
    except ImportError:
        return False
    return True


def verify_cut(result: CutResult, check_bpm: bool = False,
               reference_bpm: float | None = None) -> Report:
    report = Report(output=result.output)
    audio, sample_rate = sf.read(str(result.output), dtype="float32", always_2d=True)

    loop = result.spec.kind == "loop"
    expected = (loop_length_samples(result.spec.bars, result.spec.bpm, sample_rate,
                                    result.spec.beats_per_bar) if loop else result.length_samples)
    report.checks.append(Check("sample_count", len(audio) == expected,
                               f"got {len(audio)}, expected {expected}"))
    report.checks.append(Check("sample_rate", sample_rate == result.sample_rate,
                               f"{sample_rate} vs source {result.sample_rate}"))

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    ceiling = result.source_peak * 10.0 ** (result.trim_db / 20.0)
    trim = f" with {result.trim_db:+.1f} dB trim" if result.trim_db else ""
    report.checks.append(Check("peak_headroom",
                               ceiling <= 1.0 + PEAK_EPS and peak <= ceiling + PEAK_EPS,
                               f"peak {peak:.4f}; source {result.source_peak:.4f}{trim}"))
    report.checks.append(Check("not_silent", peak > 1e-5, f"peak {peak:.6f}"))
    dc = float(np.max(np.abs(audio.mean(axis=0)))) if audio.size else 0.0
    report.checks.append(Check("dc_offset", dc <= MAX_DC_OFFSET, f"dc {dc:.5f}"))
    if not loop:
        return report                  # one-shots have no beats to align and no tempo to match

    if not _librosa_available():
        report.checks.append(Check("beat_alignment", True, "not judged - librosa not installed"))
        if check_bpm and reference_bpm:
            report.checks.append(_bpm_from_reference(result, reference_bpm))
        return report

    mono, lead = _timing_window(result)
    report.checks.append(_beat_alignment_check(result, mono, lead, sample_rate))
    if check_bpm:
        report.checks.append(_bpm_from_reference(result, reference_bpm) if reference_bpm
                             else _bpm_from_audio(result, mono, sample_rate))
    return report


def _timing_window(result: CutResult) -> tuple[np.ndarray, float]:
    """The source around the loop (the full mix for a stem) as mono, with context either
    side so the onset detector sees the first attack. Returns it and where the loop starts."""
    path = result.timing_source or result.spec.source
    info = sf.info(str(path))
    pad = int(round(CONTEXT_S * info.samplerate))
    lo = max(0, result.start_sample - pad)
    hi = min(info.frames, result.start_sample + result.length_samples + pad)
    audio, _ = sf.read(str(path), start=lo, stop=hi, dtype="float32", always_2d=True)
    return audio.mean(axis=1), (result.start_sample - lo) / info.samplerate


def _beat_alignment_check(result: CutResult, mono: np.ndarray, lead: float, sr: int) -> Check:
    """The strongest attack within 60 ms of each of the loop's beats. Placement is their
    strength-weighted median; drift is how the kick train moves from first beat to last."""
    period = 60.0 / result.spec.bpm
    n = int(round(result.spec.bars * result.spec.beats_per_bar))
    if n < 2:
        return Check("beat_alignment", True, "not judged - shorter than two beats")
    times, strengths = detect_onsets(mono, sr)
    times = times - lead
    inside = (times > -ALIGN_WINDOW_S) & (times < n * period)
    found = attacks_near(times[inside], strengths[inside], np.arange(n) * period, ALIGN_WINDOW_S)
    if len(found) < max(2, n // 2):
        if inside.sum() >= n:
            return Check("beat_alignment", False,
                         f"attacks don't sit on the loop's beats: only {len(found)} of {n} beats "
                         f"have one within {ALIGN_WINDOW_S * 1000:.0f} ms - check the start and the tempo")
        return Check("beat_alignment", True, f"not judged - only {int(inside.sum())} onsets")
    agreement = found.agreement(n)
    if agreement < MIN_AGREEMENT:
        return Check("beat_alignment", True,
                     f"not judged - no steady attack on the beats ({agreement:.0%} agree)")
    placement = found.placement * 1000
    drift = (found.drift(n) or 0.0) * 1000
    detail = f"attacks sit {placement:+.1f} ms from the beats, drifting {drift:+.1f} ms over the loop"
    lo, hi = FAIL_PLACEMENT_MS
    if not lo <= placement <= hi or abs(drift) > DRIFT_FAIL_MS:
        return Check("beat_alignment", False, detail)
    lo, hi = PASS_PLACEMENT_MS
    if not lo <= placement <= hi or abs(drift) > DRIFT_WARN_MS:
        return Check("beat_alignment", True, detail + " - worth a listen", warn=True)
    return Check("beat_alignment", True, detail)


def _bpm_from_reference(result: CutResult, reference_bpm: float) -> Check:
    beats = result.spec.bars * result.spec.beats_per_bar
    end_ms = beats * abs(60 / result.spec.bpm - 60 / reference_bpm) * 1000
    return Check("bpm_match", end_ms <= REFERENCE_END_ERROR_MS,
                 f"declared {result.spec.bpm:g}, analysed {reference_bpm:.3f}: "
                 f"the loop's end is {end_ms:.1f} ms from the next bar")


def _bpm_from_audio(result: CutResult, mono: np.ndarray, sr: int) -> Check:
    declared = result.spec.bpm
    n = int(round(result.spec.bars * result.spec.beats_per_bar))
    measured = _measured_bpm(onset_envelope(mono, sr), sr, declared, n)
    if measured is None:
        return Check("bpm_match", True, "not judged - no clear pulse near the declared tempo")
    end_ms = n * abs(60 / declared - 60 / measured) * 1000
    return Check("bpm_match", end_ms <= MEASURED_END_ERROR_MS,
                 f"declared {declared:g}, measured {measured:.2f}: "
                 f"the loop's end is {end_ms:.1f} ms from the next bar")


def _measured_bpm(envelope: np.ndarray, sr: int, declared: float, n_beats: int) -> float | None:
    """Tempo from the audio: the autocorrelation peak of the onset envelope near K beats
    of the declared tempo, refined between frames. K beats rather than one beat divides
    the frame quantisation by K."""
    x = envelope - envelope.mean()
    size = x.size
    spectrum = np.fft.rfft(x, 2 * size)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:size] / np.arange(size, 0, -1)
    k = max(1, min(4, n_beats // 2))
    width = min(0.15, 0.45 / k)                 # keeps the half-beat peaks outside the search
    centre = k * 60.0 / declared * sr / HOP
    lo, hi = int(centre * (1 - width)), int(np.ceil(centre * (1 + width)))
    if lo < 1 or hi + 1 >= size or ac[0] <= 0:
        return None
    peak = lo + int(np.argmax(ac[lo : hi + 1]))
    if peak in (lo, hi) or ac[peak] < 0.2 * ac[0]:
        return None
    a, b, c = ac[peak - 1], ac[peak], ac[peak + 1]
    denom = a - 2 * b + c
    frac = 0.5 * (a - c) / denom if denom else 0.0
    return 60.0 * sr * k / (HOP * (peak + frac))
