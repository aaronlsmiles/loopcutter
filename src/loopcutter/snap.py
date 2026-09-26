"""Snap starts onto the analysed grid, and turn DJ-app markers into manifest rows.

The DJ app says roughly where and which beat; the grid says exactly when. The
app's measured offset is removed first, every shift is reported, and shifts
beyond the limit are refused rather than applied. Rows come out resolved, so
`cut` never has to move a start.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .grid import GridError, local_grid, local_inliers, nominal_bpm
from .markers import parse_note
from .model import Grid, Marker, SnapResult, TrackRecord
from .trackdb import find_record

COMMON_BARS = (0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 32)
BAR_TOLERANCE = 0.03
FILE_START_SLACK_S = 0.005
LOCAL_FIT = 0.9                 # the share of nearby beats that must sit on the whole-track grid
MANIFEST_FIELDS = ["source", "track_id", "label", "artist", "track", "bars", "bpm", "start",
                   "snap", "variations", "stem", "key", "notes"]


class SnapError(ValueError):
    """Raised when a start is further from the grid than the limit allows."""


@dataclass
class ImportResult:
    rows: list[dict] = field(default_factory=list)
    moves_ms: list[float] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def track_grid(record: TrackRecord) -> Grid:
    """The track's grid at its fitted tempo (not the rounded one), unless you've overridden it."""
    bpm = record.override_bpm or record.bpm_fitted or record.bpm
    grid = Grid(period=60.0 / bpm, phase=record.phase, bar_phase=record.bar_phase,
                beats_per_bar=record.beats_per_bar, inlier_ratio=record.inlier_ratio,
                bar_agreement=record.bar_agreement, phase_agreement=record.phase_agreement)
    return grid.shifted(record.override_phase_ms / 1000) if record.override_phase_ms else grid


def grid_at(record: TrackRecord, t: float,
            beats_for: Callable[[TrackRecord], tuple | None] | None = None) -> tuple[Grid, float]:
    """The grid to snap against near time t, and the tempo a loop starting there has.

    A track whose beats don't fit one steady grid (flagged grid-fit) gets a grid
    fitted to the cached beats around t, moved onto the attack by the track's
    measured offset and your override, as the whole-track grid was.
    """
    if not (record.override_bpm or record.bpm_fitted or record.bpm):
        raise SnapError("not analysed yet - run `loopcutter scan`")
    if beats_for is None or "grid-fit" not in record.flags.split(";") or record.override_bpm:
        return track_grid(record), record.override_bpm or record.bpm
    cached = beats_for(record)
    if cached is None:
        raise SnapError("its beats don't fit one steady grid and there is no beat cache - "
                        f"run `loopcutter scan --force \"{record.track_id}\"`")
    whole = track_grid(record)
    detected = replace(record, override_phase_ms=None)                 # the grid the raw beats were fitted to
    detected = track_grid(detected).shifted(-record.phase_offset_ms / 1000)
    if local_inliers(detected, cached[0], t) >= LOCAL_FIT:
        # Flagged for a break or an intro elsewhere; here the whole-track grid, fitted to far
        # more beats, still holds, and its tempo is the more accurate.
        return whole, record.bpm
    try:
        grid = local_grid(*cached, t=t)
    except GridError as exc:
        raise SnapError(str(exc)) from None
    shift = (record.phase_offset_ms + (record.override_phase_ms or 0.0)) / 1000
    return (grid.shifted(shift) if shift else grid), nominal_bpm(grid.bpm)


def snap_time(t: float, grid: Grid, mode: str = "beat", max_shift_ms: float = 60.0) -> SnapResult:
    if mode == "off":
        n = grid.beat_index(t)
        return SnapResult(t, t, n, grid.is_bar_start(n))
    target = grid.nearest_bar(t) if mode == "bar" else grid.nearest_beat(t)
    n = grid.beat_index(target)
    if target < 0:
        # The first beat can sit a hair before the file because the grid leads the attack.
        if target < -FILE_START_SLACK_S:
            raise SnapError(f"{t:.3f}s snaps to {target:.3f}s, before the start of the file")
        target = 0.0
    result = SnapResult(t, target, n, grid.is_bar_start(n))
    if abs(result.shift_ms) > max_shift_ms:
        before = grid.beat_time(math.floor((t - grid.phase) / grid.period))
        raise SnapError(f"{t:.3f}s is {result.shift_ms:+.0f} ms from the nearest {mode} "
                        f"(limit {max_shift_ms:.0f} ms); nearby beats {before:.3f}s and "
                        f"{before + grid.period:.3f}s, nearest bar {grid.nearest_bar(t):.3f}s")
    return result


def bars_from_length(seconds: float, grid: Grid) -> float | None:
    bars = seconds / (grid.period * grid.beats_per_bar)
    best = min(COMMON_BARS, key=lambda b: abs(b - bars))
    return best if abs(bars - best) <= BAR_TOLERANCE * best else None


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _same_file(a, b) -> bool:
    return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()


def _app_correction(record: TrackRecord, marker: Marker) -> float:
    """The app's measured offset, for marks on the file it was measured on (rekordbox reads
    the master; Serato the source it tagged)."""
    flags = set(record.flags.split(";"))
    same_app = record.app and marker.app.startswith(record.app)
    measured_on = record.master if record.app.startswith("rekordbox") else record.source
    if (same_app and _same_file(marker.path, measured_on) and record.app_offset_ms is not None
            and not flags & {"half-beat", "app-bpm"}):
        return record.app_offset_ms / 1000
    return 0.0


def markers_to_rows(markers: list[Marker], tracks: dict[str, TrackRecord], *,
                    default_bars: float = 4.0, max_shift_ms: float = 60.0,
                    beats_for: Callable[[TrackRecord], tuple | None] | None = None) -> ImportResult:
    result = ImportResult()
    counters: dict[str, int] = {}
    for marker in markers:
        record = find_record(tracks, marker.path)
        if record is None:
            result.problems.append(f"{marker.path.name}: not in tracks.csv - run `loopcutter prep` and `scan`")
            continue
        where = f"{marker.path.name} at {marker.start:.3f}s"
        flags = set(record.flags.split(";"))
        t = marker.start - _app_correction(record, marker)
        try:
            grid, bpm = grid_at(record, t, beats_for)
        except SnapError as exc:
            result.problems.append(f"{where}: {exc}")
            continue
        unsure = "phase" in flags and record.override_phase_ms is None
        if unsure and marker.app.startswith("rekordbox") and _same_file(marker.path, record.master):
            # rekordbox reads the master's own samples, so its quantised mark is the best phase we have
            snap = SnapResult(marker.start, marker.start, grid.beat_index(marker.start), False)
        elif unsure:
            result.problems.append(f"{where}: this track's grid phase isn't confident - mark it in "
                                   "rekordbox on the master with quantise on, or set override_phase_ms")
            continue
        else:
            try:
                snap = snap_time(t, grid, "beat", max_shift_ms)
            except SnapError as exc:
                result.problems.append(f"{where}: {exc}")
                continue
        bars = bars_from_length(marker.end - marker.start, grid) if marker.end else default_bars
        if bars is None:
            length = (marker.end - marker.start) / (grid.period * grid.beats_per_bar)
            result.problems.append(f"{where}: the loop is {length:.2f} bars, not a standard "
                                   "length - fix it in the app or add the row by hand")
            continue
        note = parse_note(marker.name)
        counters[record.track_id] = counters.get(record.track_id, 0) + 1
        artist, _, title = Path(record.master).stem.partition(" - ")
        move_ms = (snap.snapped - marker.start) * 1000
        result.rows.append({
            "source": record.master, "track_id": record.track_id,
            "label": note.label or f"M{counters[record.track_id]}",
            "artist": artist if title else "", "track": title or artist,
            "bars": _fmt(bars), "bpm": _fmt(bpm), "start": f"{snap.snapped:.6f}", "snap": "",
            "variations": ",".join(_fmt(v) for v in note.variations if v != bars),
            "stem": note.stem or "", "key": record.key,
            "notes": f"{marker.app} mark at {marker.start:.3f}s, moved {move_ms:+.1f} ms"
                     + (" (kept: grid phase not confident)" if unsure else ""),
        })
        result.moves_ms.append(move_ms)
    return result


def write_manifest(rows: list[dict], path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; choose another name")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
