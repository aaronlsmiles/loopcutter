"""Stem separation before cutting.

audio-separator runs Demucs and the RoFormer models behind one API; the Demucs
command line stays as a fallback. Separators work at 44.1 kHz, so stems are
resampled to the master's rate and proven to line up with it and match its level
before any loop is cut from them. Raw output is cached per track.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import soxr

from .audio_io import read_audio

DEFAULT_ENGINE = "audio-separator"
DEFAULT_MODEL = "htdemucs_6s.yaml"
STEM_NAMES = ("drums", "bass", "other", "vocals", "guitar", "piano", "instrumental")
ALIGN_SECONDS = 20.0
GAIN_TOLERANCE_DB = 1.0
MIN_STEM_SHARE = 0.01  # a stem under -20 dB of the master is too faint to align on its own
STEM_LAG_S = 0.001      # one stem against the full mix: band-limited stems (bass) correlate a few
                        # samples off with no timing error; the sum is held to one sample
COMPLETE_MARKER = "complete.json"
_STEM_IN_NAME = re.compile(r"\((\w+)\)")

Runner = Callable[[Path, Path, str], list[Path]]


class ConformedStems(dict):
    """{stem: AIFF path}, plus what the proof measured: gain_db is the level of the
    stems' sum against the master (least squares, 0 means matched) and clipped counts
    the samples per stem that exceeded full scale and were clipped when written."""

    def __init__(self, paths: dict[str, Path], gain_db: float, clipped: dict[str, int]):
        super().__init__(paths)
        self.gain_db = gain_db
        self.clipped = clipped


def _audio_separator(source: Path, out_dir: Path, model: str) -> list[Path]:
    from audio_separator.separator import Separator

    # 1.0 is the loosest threshold it accepts: stems are scaled only if they peak above
    # full scale. soundfile writes the float result at the source's own bit depth.
    separator = Separator(output_dir=str(out_dir), output_format="WAV",
                          normalization_threshold=1.0, use_soundfile=True)
    separator.load_model(model_filename=model)
    return [out_dir / Path(name).name for name in separator.separate(str(source))]


def _demucs(source: Path, out_dir: Path, model: str) -> list[Path]:
    subprocess.run(["demucs", "-n", model, "--float32", "--clip-mode", "none",
                    "-o", str(out_dir), str(source)], check=True)
    return sorted((out_dir / model / source.stem).glob("*.wav"))


ENGINES: dict[str, Runner] = {"audio-separator": _audio_separator, "demucs": _demucs}


def collect_stems(files) -> dict[str, Path]:
    """{stem: file} from separator output named 'Song_(Bass)_model.wav' or 'bass.wav'."""
    found: dict[str, Path] = {}
    for file in map(Path, files):
        tags = _STEM_IN_NAME.findall(file.name)
        name = (tags[-1] if tags else file.stem).lower()
        if name in STEM_NAMES:
            found[name] = file
    return found


def separate(source, out_dir, engine: str = DEFAULT_ENGINE, model: str = DEFAULT_MODEL,
             runner: Runner | None = None) -> dict[str, Path]:
    """Raw separator output for one source, cached in
    out_dir/<source name>/raw/<engine>-<model>. The cache counts only once a finished
    run has marked it, and only for the source file it was made from."""
    source = Path(source)
    raw = Path(out_dir) / source.stem / "raw" / f"{engine}-{model}"
    marker = raw / COMPLETE_MARKER
    stat = source.stat()
    stamp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    try:
        finished = json.loads(marker.read_text()) == stamp
    except (OSError, ValueError):
        finished = False
    cached = collect_stems(sorted(raw.rglob("*.wav"))) if finished else {}
    if cached:
        return cached
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    found = collect_stems((runner or ENGINES[engine])(source, raw, model))
    if not found:
        raise RuntimeError(f"{engine} produced no stems for {source.name}")
    marker.write_text(json.dumps(stamp))
    return found


def conform_stems(stems: dict[str, Path], master, out_dir) -> ConformedStems:
    """Resample each stem to the master's rate and prove the stems' sum lines up with
    the master to the sample and matches its level, all in memory; only then swap the
    set in as out_dir/<stem>.aiff (24-bit). Any failure leaves out_dir with no stems."""
    out_dir = Path(out_dir)
    partial = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        master_audio, rate = read_audio(master)
        audio = {name: _at_rate(path, rate) for name, path in sorted(stems.items())}
        gain_db = _prove(audio, master_audio, rate, Path(master).name)
        partial.mkdir(parents=True)
        clipped = {}
        for name, x in audio.items():
            clipped[name] = int(np.count_nonzero(np.abs(x) > 1.0))
            sf.write(str(partial / f"{name}.aiff"), np.clip(x, -1.0, 1.0), rate,
                     format="AIFF", subtype="PCM_24")
        _swap_in(partial, out_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        for stale in out_dir.glob("*.aiff"):
            stale.unlink()
        raise
    return ConformedStems({name: out_dir / f"{name}.aiff" for name in audio}, gain_db, clipped)


def _at_rate(path: Path, rate: int) -> np.ndarray:
    audio, sr = read_audio(path)
    return audio if sr == rate else soxr.resample(audio, sr, rate, quality="VHQ")


def _prove(audio: dict[str, np.ndarray], master: np.ndarray, rate: int, label: str) -> float:
    """Raise unless the stems' sum, and each stem loud enough to judge, line up with the
    master over its loudest stretch, and the sum matches its level; return that level
    in dB."""
    total = None
    for x in audio.values():
        total = x if total is None else total[: len(x)] + x[: len(total)]
    length = min(len(total), len(master))
    span = min(length, int(ALIGN_SECONDS * rate))
    window = _loudest(master[:length].mean(axis=1), span, hop=max(1, rate // 10))
    ref = master[window].mean(axis=1)
    energy = float(np.dot(ref, ref))
    if energy == 0.0:
        raise RuntimeError(f"{label} is silent")
    lag, corr = _lag(total[window].mean(axis=1), ref)
    if abs(lag) > 1:
        raise RuntimeError(f"stems don't line up with {label}: {lag} samples out")
    for name, x in audio.items():
        part = x[window].mean(axis=1)
        if np.dot(part, part) < MIN_STEM_SHARE * energy:
            continue
        stem_lag, _ = _lag(part, ref)
        if abs(stem_lag) > STEM_LAG_S * rate:
            raise RuntimeError(f"{name} doesn't line up with {label}: {stem_lag} samples out")
    # least squares: the k that best fits mix = k * master at the aligned lag
    gain = corr[lag] / energy
    gain_db = 20 * np.log10(gain) if gain > 0 else float("-inf")
    if not abs(gain_db) <= GAIN_TOLERANCE_DB:
        raise RuntimeError(f"the stems' level doesn't match the master: {gain_db:+.1f} dB"
                           " - was the separator normalising?")
    return float(gain_db)


def _loudest(mono: np.ndarray, span: int, hop: int) -> slice:
    """The loudest span samples of mono, found to within hop samples."""
    blocks = np.add.reduceat(np.square(mono), np.arange(0, len(mono), hop), dtype=np.float64)
    sums = np.convolve(blocks, np.ones(max(1, span // hop)), "valid")
    start = min(int(np.argmax(sums)) * hop, len(mono) - span)
    return slice(start, start + span)


def _lag(x: np.ndarray, ref: np.ndarray) -> tuple[int, np.ndarray]:
    """Samples by which x trails ref, with the cross-correlation it was read from."""
    n = 2 * len(ref)
    corr = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(ref, n)), n)
    lag = int(np.argmax(corr))
    return (lag - n if lag > len(ref) else lag), corr


def _swap_in(new: Path, dest: Path) -> None:
    """Replace dest with new, carrying over what isn't a stem (the separator's raw cache
    lives in dest when the track id is the master's name)."""
    old = dest.with_name(dest.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        dest.rename(old)
        for child in old.iterdir():
            if child.suffix != ".aiff":
                child.rename(new / child.name)
    new.rename(dest)
    shutil.rmtree(old, ignore_errors=True)
