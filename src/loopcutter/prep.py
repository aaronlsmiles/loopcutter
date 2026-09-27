"""Working masters: decode once, resample once, write 24-bit AIFF at one rate.

Every DJ app then reads identical PCM, so markers carry no decoder offset.
Masters are never overwritten in place, because Mixed In Key or a DJ app may
have written tags to them. The one exception is a master built from a lossy
file when a lossless copy of the same track appears: the old master is moved
aside and its tags carried over (see `cli`). A changed source is otherwise
reported; delete the master to rebuild it. Headroom can be taken off first,
and any sample still over full scale is clipped and counted.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .audio_io import read_audio
from .naming import sanitise

LOSSLESS_SUFFIXES = {".wav", ".aif", ".aiff", ".flac"}
CONTAINER_SUFFIXES = {".m4a", ".mp4", ".caf"}      # lossless only when they hold ALAC
SAME_TRACK_S = 0.25                                  # a lossless copy of a lossy file is this close in length
AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".mp4", ".aac",
                  ".ogg", ".opus"}


@dataclass(frozen=True)
class PrepResult:
    source: Path
    master: Path
    source_rate: int | None
    rate: int
    frames: int
    reused: bool
    clipped: int = 0
    stale: bool = False
    gain_db: float = 0.0


def master_path(source, masters_dir) -> Path:
    return Path(masters_dir) / f"{sanitise(Path(source).stem)}.aiff"


def find_sources(paths) -> list[Path]:
    found: list[Path] = []
    for item in (Path(p).expanduser() for p in paths):
        if item.is_dir():
            found.extend(sorted(p for p in item.rglob("*")
                                if p.suffix.lower() in AUDIO_SUFFIXES and not p.name.startswith(".")))
        elif item.suffix.lower() in AUDIO_SUFFIXES and not item.name.startswith("."):
            found.append(item)
    return found


def _codec(path) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    run = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True)
    return (run.stdout.strip() or None) if run.returncode == 0 else None


def is_lossless(path) -> bool | None:
    """True or False, or None when it can't be told (an .m4a whose codec can't be read)."""
    suffix = Path(path).suffix.lower()
    if suffix in LOSSLESS_SUFFIXES:
        return True
    if suffix in CONTAINER_SUFFIXES:
        codec = _codec(path)
        return None if codec is None else codec == "alac"
    return False


def choose_sources(sources, masters_dir) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """One source per master. A lossless file always wins over lossy copies of the same
    track; returns the sources to use and (skipped, used instead) pairs. Masters are
    grouped as a case-insensitive filesystem sees them, and a path given twice counts
    once. Any other clash (two lossless files, two lossy ones, or a file whose kind
    can't be told) needs a human and is refused."""
    ordered, seen = [], set()
    for source in map(Path, sources):
        if source.resolve() not in seen:
            seen.add(source.resolve())
            ordered.append(source)
    by_master = defaultdict(list)
    for source in ordered:
        by_master[master_path(source, masters_dir).name.casefold()].append(source)
    winners, skipped, clashes = {}, [], {}
    for key, candidates in by_master.items():
        kinds = [is_lossless(c) for c in candidates]
        if len(candidates) == 1:
            winners[key] = candidates[0]
        elif kinds.count(True) == 1 and kinds.count(False) == len(kinds) - 1:
            winner = candidates[kinds.index(True)]
            winners[key] = winner
            skipped += [(c, winner) for c in candidates if c is not winner]
        else:
            clashes[key] = candidates
    if clashes:
        lines = [f"{master_path(srcs[0], masters_dir).name} <- " + ", ".join(str(s) for s in srcs)
                 for srcs in clashes.values()]
        raise ValueError("two sources would write the same master:\n  " + "\n  ".join(lines))
    kept = [s for s in ordered if winners[master_path(s, masters_dir).name.casefold()] is s]
    return kept, skipped


def source_seconds(path) -> float | None:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except (RuntimeError, sf.LibsndfileError):
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    run = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(run.stdout.strip())
    except ValueError:
        return None


def same_track(source, master_seconds: float) -> bool:
    """Whether a source is plausibly the audio a master of this length was made from."""
    seconds = source_seconds(source)
    return seconds is not None and abs(seconds - master_seconds) <= SAME_TRACK_S


def set_aside(master: Path, replaced_dir: Path) -> Path:
    """Move a master out of the way before it's rebuilt, as <name>_<date>.aiff."""
    replaced_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    dest = replaced_dir / f"{master.name}_{stamp}{master.suffix}"
    count = 2
    while dest.exists():
        dest = replaced_dir / f"{master.name}_{stamp}-{count}{master.suffix}"
        count += 1
    shutil.move(str(master), str(dest))
    return dest


def prepare_master(source, masters_dir, rate: int = 48000, subtype: str = "PCM_24",
                   gain_db: float = 0.0) -> PrepResult:
    """`gain_db` (usually negative) is applied before writing, so decoding and resampling
    overs above full scale fit."""
    source = Path(source)
    dest = master_path(source, masters_dir)
    if dest.exists():
        info = sf.info(str(dest))
        return PrepResult(source, dest, None, info.samplerate, info.frames, reused=True,
                          stale=source.stat().st_mtime > dest.stat().st_mtime)
    audio, source_rate = read_audio(source)
    if source_rate != rate:
        audio = soxr.resample(audio, source_rate, rate, quality="VHQ")
    if gain_db:
        audio = audio * 10.0 ** (gain_db / 20.0)
    clipped = int(np.count_nonzero(np.abs(audio) > 1.0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.stem + ".partial.aiff")
    sf.write(str(temp), np.clip(audio, -1.0, 1.0), rate, format="AIFF", subtype=subtype)
    temp.replace(dest)
    return PrepResult(source, dest, source_rate, rate, len(audio), reused=False, clipped=clipped,
                      gain_db=gain_db)
