"""Sample-accurate timing arithmetic.

Everything downstream depends on this module being exactly right. The rule
that matters: a loop's length is always derived from BPM and bar count, never
from a second timecode. Two timecodes rounded to milliseconds drift by a few
samples per loop, which is inaudible alone and catastrophic once eight loops
are stacked and left to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TIMECODE = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d{1,2}(?:\.\d+)?)$"
)


def parse_position(value: str | float | int) -> float:
    """Parse a position into seconds.

    Accepts bare seconds ("12.5"), M:SS.mmm ("1:23.456") and H:MM:SS.mmm.
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError("empty position")

    match = _TIMECODE.match(text)
    if match:
        hours = int(match.group("h") or 0)
        minutes = int(match.group("m"))
        seconds = float(match.group("s"))
        if minutes >= 60 or seconds >= 60:
            raise ValueError(f"invalid timecode {text!r}")
        return hours * 3600 + minutes * 60 + seconds

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"cannot parse position {text!r}") from exc


def samples_per_beat(bpm: float, sample_rate: int) -> float:
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return sample_rate * 60.0 / bpm


def loop_length_samples(
    bars: float, bpm: float, sample_rate: int, beats_per_bar: int = 4
) -> int:
    """Exact integer sample count for a loop of `bars` bars at `bpm`.

    Rounds once, at the end. Never round per-beat and multiply.
    """
    if bars <= 0:
        raise ValueError(f"bars must be positive, got {bars}")
    beats = bars * beats_per_bar
    return int(round(beats * samples_per_beat(bpm, sample_rate)))


def bar_start_seconds(
    downbeat: float, bar_index: int, bpm: float, beats_per_bar: int = 4
) -> float:
    """Seconds at which `bar_index` begins, counting the downbeat as bar 1."""
    if bar_index < 1:
        raise ValueError(f"bar_index is 1-based, got {bar_index}")
    seconds_per_bar = beats_per_bar * 60.0 / bpm
    return downbeat + (bar_index - 1) * seconds_per_bar


@dataclass(frozen=True)
class CutWindow:
    """A resolved, sample-accurate extraction window."""

    start_sample: int
    length_samples: int

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.length_samples


def resolve_window(
    start_seconds: float,
    bars: float,
    bpm: float,
    sample_rate: int,
    beats_per_bar: int = 4,
) -> CutWindow:
    if start_seconds < 0:
        raise ValueError(f"start must be >= 0, got {start_seconds}")
    return CutWindow(
        start_sample=int(round(start_seconds * sample_rate)),
        length_samples=loop_length_samples(bars, bpm, sample_rate, beats_per_bar),
    )


def tiling_error_samples(
    long_bars: float,
    short_bars: float,
    bpm: float,
    sample_rate: int,
    beats_per_bar: int = 4,
) -> int:
    """How far a short loop drifts against a long one over the long one's span.

    Zero means they tile exactly. Non-zero means the short loop's rounded
    sample count does not divide the long one, so repeated playback slowly
    slips. This is arithmetic, not a bug: at 128 BPM and 44.1kHz a bar is
    82687.5 samples, and you cannot have half a sample.

    Which rate tiles depends on the tempo. At 48 kHz half bars come out exact
    at 120, 125, 128, 144, 150 and 160 BPM; at 44.1 kHz at 120, 125, 126, 135,
    140, 144, 147, 150 and 160. `rates_that_tile` answers it for any tempo.
    """
    long_len = loop_length_samples(long_bars, bpm, sample_rate, beats_per_bar)
    short_len = loop_length_samples(short_bars, bpm, sample_rate, beats_per_bar)
    if short_len <= 0 or short_len > long_len:
        return 0
    repeats = round(long_bars / short_bars)
    return abs(short_len * repeats - long_len)


def tiling_error_ms(long_bars: float, short_bars: float, bpm: float, sample_rate: int,
                    beats_per_bar: int = 4) -> float:
    """tiling_error_samples expressed in milliseconds."""
    return tiling_error_samples(long_bars, short_bars, bpm, sample_rate, beats_per_bar) \
        / sample_rate * 1000


def rates_that_tile(bars: float, bpm: float, beats_per_bar: int = 4,
                    rates: tuple[int, ...] = (44100, 48000)) -> list[int]:
    """Sample rates at which `bars` at `bpm` is a whole number of samples."""
    exact = []
    for rate in rates:
        samples = bars * beats_per_bar * rate * 60 / bpm
        if abs(samples - round(samples)) < 1e-6:
            exact.append(rate)
    return exact
