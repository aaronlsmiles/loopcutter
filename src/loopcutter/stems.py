"""Optional Demucs pre-pass.

Separate first, then cut. A single 4-bar window across six stems gives six
usable loops from one cue point instead of one, which is the difference
between a library you can EQ apart and a pile of full-range fragments.

This shells out rather than importing Demucs so the core package stays light.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_MODEL = "htdemucs_6s"
STEM_NAMES = ("drums", "bass", "other", "vocals", "guitar", "piano")


def separate(
    source: str | Path,
    out_dir: str | Path,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> dict[str, Path]:
    """Run Demucs and return {stem_name: path}. Skips work already done."""
    source = Path(source)
    out_dir = Path(out_dir)
    expected = out_dir / model / source.stem

    if not expected.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["demucs", "-n", model, "-d", device, "-o", str(out_dir), str(source)],
            check=True,
        )

    found = {}
    for name in STEM_NAMES:
        for ext in (".wav", ".flac", ".mp3"):
            candidate = expected / f"{name}{ext}"
            if candidate.exists():
                found[name] = candidate
                break
    if not found:
        raise RuntimeError(f"demucs produced no stems under {expected}")
    return found
