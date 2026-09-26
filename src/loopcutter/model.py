"""Shared data types. Every module codes against these; they change only in Wave 0."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Grid:
    """A constant-tempo beat grid. Beat n sits at phase + n * period seconds."""

    period: float
    phase: float
    bar_phase: int = 0
    beats_per_bar: int = 4
    inlier_ratio: float = 1.0
    bar_agreement: float = 1.0
    phase_agreement: float = 1.0
    phase_offset: float = 0.0

    @property
    def bpm(self) -> float:
        return 60.0 / self.period

    def beat_time(self, n):
        return self.phase + n * self.period

    def beat_index(self, t: float) -> int:
        return int(round((t - self.phase) / self.period))

    def nearest_beat(self, t: float) -> float:
        return self.beat_time(self.beat_index(t))

    def is_bar_start(self, n: int) -> bool:
        return (n - self.bar_phase) % self.beats_per_bar == 0

    def nearest_bar(self, t: float) -> float:
        n = self.beat_index(t)
        below = n - ((n - self.bar_phase) % self.beats_per_bar)
        best = min((below, below + self.beats_per_bar), key=lambda m: abs(self.beat_time(m) - t))
        return self.beat_time(best)

    def shifted(self, seconds: float) -> "Grid":
        """The same grid moved later by `seconds`. Beat n becomes beat n + wraps."""
        raw = self.phase + seconds
        wraps = math.floor(raw / self.period)
        return replace(self, phase=raw - wraps * self.period,
                       bar_phase=(self.bar_phase + wraps) % self.beats_per_bar,
                       phase_offset=self.phase_offset + seconds)


@dataclass
class TrackRecord:
    """One row of tracks.csv. scan writes the analysis; the override fields are yours."""

    track_id: str
    source: str
    master: str
    sample_rate: int
    duration: float
    bpm: float
    phase: float
    bar_phase: int
    bpm_fitted: float = 0.0
    beats_per_bar: int = 4
    inlier_ratio: float = 0.0
    bar_agreement: float = 0.0
    phase_agreement: float = 0.0
    phase_offset_ms: float = 0.0
    key: str = ""
    key_source: str = ""
    key_alt: str = ""
    app: str = ""
    app_bpm: float | None = None
    app_offset_ms: float | None = None
    flags: str = ""
    override_bpm: float | None = None
    override_phase_ms: float | None = None


@dataclass(frozen=True)
class Marker:
    """A cue or loop placed in DJ software; times are in that app's own timeline."""

    path: Path
    start: float
    end: float | None
    name: str
    app: str

    @property
    def kind(self) -> str:
        return "loop" if self.end is not None else "cue"


@dataclass(frozen=True)
class MarkerNote:
    label: str | None
    stem: str | None
    variations: tuple[float, ...]


@dataclass(frozen=True)
class SnapResult:
    original: float
    snapped: float
    beat_index: int
    on_bar: bool

    @property
    def shift_ms(self) -> float:
        return (self.snapped - self.original) * 1000
