"""tracks.csv: one row of analysis per source track.

scan proposes these values; the override_* columns are yours and scan never
touches them. The cut path never reads a model, only the manifest.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, fields
from pathlib import Path

from .model import TrackRecord

FIELDS = [f.name for f in fields(TrackRecord)]
_TYPES = {f.name: f.type for f in fields(TrackRecord)}      # strings, e.g. "float | None"


def _parse(kind: str, raw: str | None):
    if kind == "str":
        return raw or ""
    if raw in ("", None):
        return None if "None" in kind else (0 if kind == "int" else 0.0)
    return int(float(raw)) if kind == "int" else float(raw)


def _format(value):
    if value is None:
        return ""
    return f"{value:.6f}" if isinstance(value, float) else value


def track_id_for(master) -> str:
    return Path(master).stem


def load_tracks(path) -> dict[str, TrackRecord]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        records = [TrackRecord(**{n: _parse(_TYPES[n], row.get(n)) for n in FIELDS})
                   for row in csv.DictReader(handle)]
    return {r.track_id: r for r in records}


def save_tracks(path, records: dict[str, TrackRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.csv")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in sorted(records.values(), key=lambda r: r.track_id):
            writer.writerow({k: _format(v) for k, v in asdict(record).items()})
    temp.replace(path)


def find_record(records: dict[str, TrackRecord], path) -> TrackRecord | None:
    target = Path(path).expanduser().resolve()
    for record in records.values():
        if target in (Path(record.source).expanduser().resolve(),
                      Path(record.master).expanduser().resolve()):
            return record
    return None
