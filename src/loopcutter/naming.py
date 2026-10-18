"""Output filename convention.

The bracketed label, tempo, bar count, key and stem make a loop library
sortable and searchable by eye in any file browser. DAWs take tempo from the
audio, not the name. Keep the pattern stable: renaming files breaks any set
or sampler project that points at them.

    Artist - Track [16.2][128][4bar][bass].aiff
    Artist - Track [16.2][128][4bar][bass][LC-1a2b3c4d].aiff    (with a provenance code)

The code, last so the library still sorts by artist, is a short hash of what the loop is: track, label, length,
stem and kind. Re-marking a start keeps it; a different loop gets a different one. It lets a loop be traced back
to its record from a set, a collected project or a sampler that only kept the file name.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from .manifest import LoopSpec

_ILLEGAL = re.compile(r'[/\\:*?"<>|]')


def sanitise(text: str) -> str:
    cleaned = _ILLEGAL.sub("-", text)
    return re.sub(r"\s+", " ", cleaned).strip(" .")


def loop_code(spec: LoopSpec) -> str:
    """LC- and 8 hex of a SHA-1 over the loop's identity (not its start, so re-marking keeps the code)."""
    bars = f"{float(spec.bars):g}"
    identity = "|".join([spec.track_id or spec.source.stem, spec.label, getattr(spec, "kind", "loop"), bars,
                         spec.stem or "full"])
    identity = unicodedata.normalize("NFC", identity)       # macOS file names come back NFD
    return "LC-" + hashlib.sha1(identity.encode()).hexdigest()[:8]


CODE_SUFFIX = re.compile(r"\[LC-[0-9a-f]{8}\]$")


def same_name(a: str, b: str) -> bool:
    """File names compared the way APFS does: NFC and NFD spellings are the same file."""
    return unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)


def copies_of(out_dir: Path, spec: LoopSpec, fmt: str) -> list[Path]:
    """Files in out_dir that are this loop: its plain name, or its name with any provenance code."""
    plain = unicodedata.normalize("NFC", build_filename(spec, fmt))
    stem = plain[: -len(fmt) - 1]
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    found = []
    for p in out_dir.iterdir():
        name = unicodedata.normalize("NFC", p.name)
        if name == plain or (name.endswith("." + fmt) and CODE_SUFFIX.sub("", name[: -len(fmt) - 1]) == stem
                             and name != plain):
            found.append(p)
    return sorted(found)


def build_filename(spec: LoopSpec, fmt: str = "aiff", code: str | None = None) -> str:
    stub_parts = [p for p in (spec.artist, spec.track) if p]
    stub = " - ".join(stub_parts) if stub_parts else spec.source.stem

    if getattr(spec, "kind", "loop") == "oneshot":
        tags = [spec.label, "oneshot"]
    else:
        bpm = int(spec.bpm) if float(spec.bpm).is_integer() else round(spec.bpm, 2)
        bars = int(spec.bars) if float(spec.bars).is_integer() else spec.bars
        tags = [spec.label, str(bpm), f"{bars}bar"]
    if spec.key:
        tags.append(spec.key)
    if spec.stem:
        tags.append(spec.stem)
    if code:
        tags.append(code)

    suffix = "".join(f"[{sanitise(str(t))}]" for t in tags)
    return f"{sanitise(stub)} {suffix}.{fmt}"


def bpm_band(bpm: float, width: int = 5) -> str:
    low = int(bpm // width) * width
    return f"{low}-{low + width - 1}"


def library_subdir(spec) -> Path:
    """<stem>/<bpm band>, e.g. bass/125-129; loops without a stem under full/, one-shots in oneshots/."""
    stem = Path(sanitise(spec.stem or "full"))
    return stem / "oneshots" if getattr(spec, "kind", "loop") == "oneshot" else stem / bpm_band(spec.bpm)
