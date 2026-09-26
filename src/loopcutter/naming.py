"""Output filename convention.

The bracketed label, tempo, bar count, key and stem make a loop library
sortable and searchable by eye in any file browser. DAWs take tempo from the
audio, not the name. Keep the pattern stable: renaming files breaks any set
or sampler project that points at them.

    Artist - Track [16.2][128][4bar][bass].aiff
"""

from __future__ import annotations

import re
from pathlib import Path

from .manifest import LoopSpec

_ILLEGAL = re.compile(r'[/\\:*?"<>|]')


def sanitise(text: str) -> str:
    cleaned = _ILLEGAL.sub("-", text)
    return re.sub(r"\s+", " ", cleaned).strip(" .")


def build_filename(spec: LoopSpec, fmt: str = "aiff") -> str:
    stub_parts = [p for p in (spec.artist, spec.track) if p]
    stub = " - ".join(stub_parts) if stub_parts else spec.source.stem

    bpm = int(spec.bpm) if float(spec.bpm).is_integer() else round(spec.bpm, 2)
    bars = int(spec.bars) if float(spec.bars).is_integer() else spec.bars

    tags = [spec.label, str(bpm), f"{bars}bar"]
    if spec.key:
        tags.append(spec.key)
    if spec.stem:
        tags.append(spec.stem)

    suffix = "".join(f"[{sanitise(str(t))}]" for t in tags)
    return f"{sanitise(stub)} {suffix}.{fmt}"


def bpm_band(bpm: float, width: int = 5) -> str:
    low = int(bpm // width) * width
    return f"{low}-{low + width - 1}"


def library_subdir(spec) -> Path:
    """<stem>/<bpm band>, e.g. bass/125-129; loops without a stem file under full/."""
    return Path(sanitise(spec.stem or "full")) / bpm_band(spec.bpm)
