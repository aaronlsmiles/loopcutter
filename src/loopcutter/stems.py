"""Stem separation before cutting.

audio-separator runs Demucs and the RoFormer models behind one API; the Demucs
command line stays as a fallback. Separators work at 44.1 kHz, so stems are
resampled to the master's rate and proven to line up with it before any loop is
cut from them. Raw output is cached per track.
"""

from __future__ import annotations

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
_STEM_IN_NAME = re.compile(r"\((\w+)\)")

Runner = Callable[[Path, Path, str], list[Path]]


def _audio_separator(source: Path, out_dir: Path, model: str) -> list[Path]:
    from audio_separator.separator import Separator

    separator = Separator(output_dir=str(out_dir), output_format="WAV")
    separator.load_model(model_filename=model)
    return [out_dir / Path(name).name for name in separator.separate(str(source))]


def _demucs(source: Path, out_dir: Path, model: str) -> list[Path]:
    subprocess.run(["demucs", "-n", model, "-o", str(out_dir), str(source)], check=True)
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
    """Raw separator output for one source, cached in out_dir/<source name>/raw."""
    source = Path(source)
    raw = Path(out_dir) / source.stem / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cached = collect_stems(sorted(raw.rglob("*.wav")))
    if cached:
        return cached
    found = collect_stems((runner or ENGINES[engine])(source, raw, model))
    if not found:
        raise RuntimeError(f"{engine} produced no stems for {source.name}")
    return found


def conform_stems(stems: dict[str, Path], master, out_dir) -> dict[str, Path]:
    """Resample each stem to the master's rate and prove the stems' sum lines up with
    the master to the sample, all in memory; only then swap the set in as
    out_dir/<stem>.aiff (24-bit). Any failure leaves out_dir with no stems."""
    out_dir = Path(out_dir)
    partial = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        master_audio, rate = read_audio(master)
        audio = {name: _at_rate(path, rate) for name, path in sorted(stems.items())}
        _check_alignment(audio, master_audio, rate, Path(master).name)
        partial.mkdir(parents=True)
        for name, x in audio.items():
            sf.write(str(partial / f"{name}.aiff"), np.clip(x, -1.0, 1.0), rate,
                     format="AIFF", subtype="PCM_24")
        _swap_in(partial, out_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        for stale in out_dir.glob("*.aiff"):
            stale.unlink()
        raise
    return {name: out_dir / f"{name}.aiff" for name in audio}


def _at_rate(path: Path, rate: int) -> np.ndarray:
    audio, sr = read_audio(path)
    return audio if sr == rate else soxr.resample(audio, sr, rate, quality="VHQ")


def _check_alignment(audio: dict[str, np.ndarray], master: np.ndarray, rate: int,
                     label: str) -> None:
    total = None
    for x in audio.values():
        total = x if total is None else total[: len(x)] + x[: len(total)]
    span = min(len(total), len(master), int(ALIGN_SECONDS * rate))
    mix, ref = total[:span].mean(axis=1), master[:span].mean(axis=1)
    corr = np.fft.irfft(np.fft.rfft(mix, 2 * span) * np.conj(np.fft.rfft(ref, 2 * span)))
    lag = int(np.argmax(corr))
    lag = lag - 2 * span if lag > span else lag
    if abs(lag) > 1:
        raise RuntimeError(f"stems don't line up with {label}: {lag} samples out")


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
