"""Working masters: decode once, resample once, write 24-bit AIFF at one rate.

Every DJ app then reads identical PCM, so markers carry no decoder offset.
Existing masters are never overwritten, because Mixed In Key or a DJ app may
have written tags to them. A changed source is reported as stale; delete the
master to rebuild it. Overs created by resampling are clipped and counted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .audio_io import read_audio
from .naming import sanitise

LOSSLESS_SUFFIXES = {".wav", ".aif", ".aiff", ".flac"}
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


def is_lossless(path) -> bool:
    return Path(path).suffix.lower() in LOSSLESS_SUFFIXES


def choose_sources(sources, masters_dir) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """One source per master. A lossless file always wins over lossy copies of the same
    track; returns the sources to use and (skipped, used instead) pairs. Any other clash
    (two lossless files, or two lossy ones) needs a human and is refused."""
    by_master = defaultdict(list)
    for source in map(Path, sources):
        by_master[master_path(source, masters_dir)].append(source)
    winners, skipped, clashes = {}, [], {}
    for master, candidates in by_master.items():
        lossless = [s for s in candidates if is_lossless(s)]
        if len(candidates) == 1 or len(lossless) == 1:
            winner = candidates[0] if len(candidates) == 1 else lossless[0]
            winners[master] = winner
            skipped += [(s, winner) for s in candidates if s != winner]
        else:
            clashes[master] = candidates
    if clashes:
        lines = [f"{m.name} <- " + ", ".join(str(s) for s in srcs) for m, srcs in clashes.items()]
        raise ValueError("two sources would write the same master:\n  " + "\n  ".join(lines))
    kept = [s for s in map(Path, sources) if winners.get(master_path(s, masters_dir)) == s]
    return kept, skipped


def check_collisions(sources, masters_dir) -> None:
    choose_sources(sources, masters_dir)


def prepare_master(source, masters_dir, rate: int = 48000, subtype: str = "PCM_24",
                   gain_db: float = 0.0, replace: bool = False) -> PrepResult:
    """`gain_db` (usually negative) is taken off before writing, so decoding and resampling
    overs above full scale fit. `replace` rebuilds an existing master."""
    source = Path(source)
    dest = master_path(source, masters_dir)
    if dest.exists() and not replace:
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
