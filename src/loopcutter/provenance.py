"""Where each loop came from: the code in its name, TXXX tags, and a provenance.csv next to the loops.

`cut` writes all three for every loop that passes its checks; `loopcutter provenance` does the same for loops
cut before codes existed. The index is keyed on the code and rewritten in place, so it always lists each loop once.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

INDEX = "provenance.csv"
FIELDS = ["code", "file", "artist", "track", "label", "kind", "bars", "bpm", "stem", "key", "start_seconds",
          "source_file", "manifest", "row", "cut_at"]


def source_text(spec, source_file) -> str:
    """One line for the SOURCE tag: what, from where, at which point."""
    who = " - ".join(p for p in (spec.artist, spec.track) if p) or Path(source_file).stem
    if getattr(spec, "kind", "loop") == "oneshot":
        span = "one-shot"
    else:
        span = f"{float(spec.bars):g} bars @ {float(spec.bpm):g} BPM"
    stem = f", {spec.stem} stem" if spec.stem else ""
    return f"{who} | {source_file} | from {spec.start_seconds:.6f} s, {span}{stem} | label {spec.label}"


def index_row(spec, code: str, path: Path, out_root: Path, source_file, manifest) -> dict[str, str]:
    return {"code": code, "file": path.relative_to(out_root).as_posix(), "artist": spec.artist or "",
            "track": spec.track or "", "label": spec.label, "kind": getattr(spec, "kind", "loop"),
            "bars": f"{float(spec.bars):g}", "bpm": f"{float(spec.bpm):g}", "stem": spec.stem or "",
            "key": spec.key or "", "start_seconds": f"{spec.start_seconds:.6f}", "source_file": str(source_file),
            "manifest": Path(manifest).name, "row": str(spec.row_number),
            "cut_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def update_index(out_root: Path, rows: list[dict[str, str]]) -> Path:
    """Upsert rows by code into out_root/provenance.csv (written whole, then swapped in)."""
    path = Path(out_root) / INDEX
    current: dict[str, dict[str, str]] = {}
    if path.exists():
        with open(path, newline="") as f:
            current = {r["code"]: r for r in csv.DictReader(f)}
    for row in rows:
        current[row["code"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.csv")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(current.values(), key=lambda r: r["file"]))
    os.replace(tmp, path)
    return path
