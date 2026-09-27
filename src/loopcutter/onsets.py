"""Onsets: where attacks start, to within a millisecond, and how strong they are.

Spectral flux on 64 mel bands at a 32-sample hop, peak-picked and backtracked
to the start of each attack; on clicks the bias measured +0.02 ms. The grid,
the analysis and the checks all use this one detector.

Dense masters give 10-20 onsets a second, so "the onset nearest a beat" is
noise. What locates a beat is the *strongest* onset near it, and the
strength-weighted median of those across many beats.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

HOP = 32
CHUNK_S = 60.0


def onset_envelope(mono, sr: int) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Empty filters detected")
        return librosa.onset.onset_strength(y=np.ascontiguousarray(mono, dtype=np.float32), sr=sr,
                                            hop_length=HOP, n_fft=512, n_mels=64)


def _track_envelope(mono: np.ndarray, sr: int, chunk_s: float) -> np.ndarray:
    """The onset envelope of the whole track, built from overlapping chunks so a
    long track never needs a full-resolution spectrogram in memory at once. Chunk
    edges sit on the hop grid, so the pieces join into one frame grid."""
    total = 1 + mono.size // HOP
    step = max(HOP, int(chunk_s * sr) // HOP * HOP)
    pad = max(HOP, sr // HOP * HOP)
    envelope = np.empty(total, dtype=np.float32)
    for start in range(0, mono.size, step):
        lo = max(0, start - pad)
        piece = onset_envelope(mono[lo : start + step + pad], sr)
        first = start // HOP
        last = total if start + step >= mono.size else (start + step) // HOP
        envelope[first:last] = piece[first - lo // HOP : last - lo // HOP]
    return envelope


def detect_onsets(mono, sr: int, chunk_s: float = CHUNK_S) -> tuple[np.ndarray, np.ndarray]:
    """Onset times in seconds (sorted) and each onset's strength (the flux at its peak).

    Peaks are picked once, on the whole track's envelope: onset_detect scales an
    envelope by its own maximum, so picking per chunk would make weak onsets come
    and go with the chunk size."""
    import librosa

    mono = np.asarray(mono, dtype=np.float32)
    if mono.size == 0:
        return np.array([]), np.array([])
    envelope = _track_envelope(mono, sr, chunk_s)
    peaks = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=HOP, units="frames")
    if peaks.size == 0:
        return np.array([]), np.array([])
    starts = librosa.onset.onset_backtrack(peaks, envelope)
    return librosa.frames_to_time(starts, sr=sr, hop_length=HOP), envelope[peaks]


@dataclass(frozen=True)
class Attacks:
    """The strongest onset within a window of each target time (usually each beat)."""

    index: np.ndarray        # which target each attack belongs to
    offset: np.ndarray       # seconds, attack minus target
    strength: np.ndarray

    def __len__(self) -> int:
        return int(self.index.size)

    @property
    def placement(self) -> float:
        """Strength-squared-weighted median offset: where the beats' attack sits."""
        order = np.argsort(self.offset)
        weights = np.cumsum(self.strength[order] ** 2)
        return float(self.offset[order][np.searchsorted(weights, weights[-1] / 2)])

    def agreement(self, targets: int, tolerance: float = 0.005) -> float:
        """Share of all targets whose attack sits within `tolerance` of the placement."""
        if not len(self):
            return 0.0
        return float(np.sum(np.abs(self.offset - self.placement) <= tolerance) / targets)

    def drift(self, targets: int, tolerance: float = 0.008) -> float | None:
        """How far the attack moves from the first target to the last, in seconds, fitted
        only through attacks near the placement (the kick train)."""
        near = np.abs(self.offset - self.placement) <= tolerance
        if near.sum() < 4:
            return None
        slope = np.polyfit(self.index[near], self.offset[near], 1, w=self.strength[near])[0]
        return float(slope * targets)


def attacks_near(times, strengths, targets, window: float) -> Attacks:
    """For each target, the strongest onset within +-window of it. `times` must be sorted."""
    times = np.asarray(times, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    targets = np.asarray(targets, dtype=float)
    index, offset, strength = [], [], []
    lo = np.searchsorted(times, targets - window, side="left")
    hi = np.searchsorted(times, targets + window, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            j = a + int(np.argmax(strengths[a:b]))
            index.append(i)
            offset.append(times[j] - targets[i])
            strength.append(strengths[j])
    return Attacks(np.array(index, dtype=int), np.array(offset, dtype=float),
                   np.array(strength, dtype=float))
