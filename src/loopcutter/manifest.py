"""Manifest parsing.

The manifest is the source of truth. Keep it in git. Never edit outputs by
hand - change the manifest and re-cut, so every loop on disk is reproducible
from a single text file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .timing import bar_start_seconds, parse_position

REQUIRED = {"source", "label", "bars", "bpm"}

# A row must resolve a start position by exactly one of these routes.
START_BY_TIME = {"start"}
START_BY_BAR = {"downbeat", "start_bar"}


@dataclass
class LoopSpec:
    source: Path
    label: str
    bars: float
    bpm: float
    start_seconds: float
    beats_per_bar: int = 4
    stem: str | None = None
    artist: str | None = None
    track: str | None = None
    key: str | None = None
    notes: str = ""
    row_number: int = 0
    extra: dict[str, str] = field(default_factory=dict)
    track_id: str | None = None
    snap: str = ""
    kind: str = "loop"
    end_seconds: float | None = None
    xfade_ms: float | None = None      # None: use the command line's --xfade-ms

    @property
    def slug(self) -> str:
        parts = [p for p in (self.artist, self.track) if p]
        stub = " - ".join(parts) if parts else self.source.stem
        stem = f"[{self.stem}]" if self.stem else ""
        if self.kind == "oneshot":
            return f"{stub} [{self.label}][oneshot]{stem}"
        bars = int(self.bars) if float(self.bars).is_integer() else self.bars
        return f"{stub} [{self.label}][{bars}bar]{stem}"


class ManifestError(ValueError):
    """Raised with the offending row number so failures are addressable."""


def _require(row: dict[str, str], row_number: int) -> None:
    missing = REQUIRED - {k for k, v in row.items() if v not in (None, "")}
    if missing:
        raise ManifestError(
            f"row {row_number}: missing required column(s) {sorted(missing)}"
        )


def _resolve_start(row: dict[str, str], bpm: float, beats_per_bar: int,
                   row_number: int) -> float:
    has_time = bool(row.get("start"))
    has_bar = bool(row.get("downbeat")) and bool(row.get("start_bar"))

    if has_time and has_bar:
        raise ManifestError(
            f"row {row_number}: give either 'start' or 'downbeat'+'start_bar', not both"
        )
    if has_time:
        return parse_position(row["start"])
    if has_bar:
        return bar_start_seconds(
            downbeat=parse_position(row["downbeat"]),
            bar_index=int(row["start_bar"]),
            bpm=bpm,
            beats_per_bar=beats_per_bar,
        )
    raise ManifestError(
        f"row {row_number}: no start position - give 'start', or 'downbeat' and 'start_bar'"
    )


def _parse_variations(row: dict[str, str], row_number: int) -> list[float]:
    """Bar lengths this row should produce.

    A `variations` cell like "4,2,1,0.5" expands one cue point into four loops
    that all share a start, so they can live in a Launchpad column and be
    launched in Legato mode for beat-division rolls. Each is a real file cut to
    an exact sample count, so they stay locked to the grid in a way a mapped
    loop-length control does not.

    `bars` is always included, and duplicates are dropped, so "4" with bars=4
    yields one loop rather than two identical ones.
    """
    base = float(row["bars"])
    raw = (row.get("variations") or "").strip()
    if not raw:
        return [base]

    lengths = [base]
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ManifestError(
                f"row {row_number}: variations must be bar numbers, got {token!r}"
            ) from exc
        if value <= 0:
            raise ManifestError(
                f"row {row_number}: variation bar counts must be positive, got {value}"
            )
        lengths.append(value)

    seen: list[float] = []
    for value in lengths:
        if value not in seen:
            seen.append(value)
    return sorted(seen, reverse=True)


def _xfade(row: dict[str, str], row_number: int) -> float | None:
    raw = row.get("xfade_ms")
    if raw in (None, ""):
        return None
    value = float(raw)
    if value < 0:
        raise ManifestError(f"row {row_number}: xfade_ms can't be negative, got {value:g}")
    return value


def load_manifest(path: str | Path, audio_root: str | Path | None = None) -> list[LoopSpec]:
    """Load a manifest.

    Relative `source` paths resolve against `audio_root`, which defaults to the
    current working directory - not the manifest's own directory - so that
    manifests/session.csv can refer to audio/track.aiff the way you would type
    it from the repo root.
    """
    path = Path(path)
    root = Path(audio_root) if audio_root else Path.cwd()

    specs: list[LoopSpec] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError(f"{path} has no header row")

        known = REQUIRED | START_BY_TIME | START_BY_BAR | {
            "beats_per_bar", "stem", "artist", "track", "key", "notes",
            "variations", "track_id", "snap", "kind", "end", "xfade_ms",
        }

        for offset, row in enumerate(reader, start=2):
            row = {k: (v.strip() if isinstance(v, str) else v)
                   for k, v in row.items() if k}
            if not any(row.values()):
                continue
            if row.get("source", "").startswith("#"):
                continue

            kind = (row.get("kind") or "loop").lower()
            if kind not in {"loop", "oneshot"}:
                raise ManifestError(f"row {offset}: kind must be loop or oneshot, got {kind!r}")
            if kind == "oneshot":
                missing = [c for c in ("source", "label", "start", "end") if not row.get(c)]
                if missing:
                    raise ManifestError(f"row {offset}: a oneshot needs {missing}")
                start, end = parse_position(row["start"]), parse_position(row["end"])
                if end <= start:
                    raise ManifestError(f"row {offset}: end must be after start")
                source = Path(row["source"])
                specs.append(LoopSpec(
                    source=source if source.is_absolute() else root / source, label=row["label"],
                    bars=0.0, bpm=float(row.get("bpm") or 0), start_seconds=start,
                    stem=row.get("stem") or None, artist=row.get("artist") or None,
                    track=row.get("track") or None, key=row.get("key") or None,
                    notes=row.get("notes") or "", row_number=offset, kind="oneshot",
                    end_seconds=end, track_id=row.get("track_id") or None))
                continue

            snap = (row.get("snap") or "").lower()
            if snap in {"beat", "bar"}:
                raise ManifestError(f"row {offset}: start not snapped yet - run "
                                    f"`loopcutter resolve {path}`")
            if snap:
                raise ManifestError(f"row {offset}: snap must be beat, bar or empty, got {snap!r}")
            if row.get("track_id") and not (row.get("bpm") and row.get("source")):
                raise ManifestError(f"row {offset}: source or bpm missing - run "
                                    f"`loopcutter resolve {path}` to fill them from tracks.csv")
            _require(row, offset)
            bpm = float(row["bpm"])
            beats_per_bar = int(row.get("beats_per_bar") or 4)
            lengths = _parse_variations(row, offset)

            source = Path(row["source"])
            if not source.is_absolute():
                source = root / source

            start = _resolve_start(row, bpm, beats_per_bar, offset)
            for bars in lengths:
                specs.append(
                    LoopSpec(
                        source=source,
                        label=row["label"],
                        bars=bars,
                        bpm=bpm,
                        start_seconds=start,
                        beats_per_bar=beats_per_bar,
                        stem=row.get("stem") or None,
                        artist=row.get("artist") or None,
                        track=row.get("track") or None,
                        key=row.get("key") or None,
                        notes=row.get("notes") or "",
                        row_number=offset,
                        extra={k: v for k, v in row.items() if k not in known and v},
                        track_id=row.get("track_id") or None,
                        snap=snap,
                        xfade_ms=_xfade(row, offset),
                    )
                )

    if not specs:
        raise ManifestError(f"{path} contains no usable rows")

    labels = [s.slug for s in specs]
    duplicates = {name for name in labels if labels.count(name) > 1}
    if duplicates:
        raise ManifestError(f"duplicate output names would collide: {sorted(duplicates)}")

    return specs
