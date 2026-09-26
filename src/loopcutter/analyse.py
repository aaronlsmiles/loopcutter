"""scan: analyse each master once, record grid, phase, key and doubts in tracks.csv,
and cache the raw beats so snapping can fit local grids without re-running the model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .audio_io import AudioReadError, read_audio
from .grid import (MIN_PHASE_AGREEMENT, Detector, GridError, analyse_grid, compare_grids,
                   nominal_bpm, save_beats)
from .keydetect import KeyResult, detect_key, key_from_keyfinder
from .model import Grid, TrackRecord
from .trackdb import load_tracks, save_tracks

LOSSLESS = {".wav", ".aif", ".aiff", ".flac"}
APP_FLAGS = ("app-bpm", "half-beat", "app-offset")
MIN_APP_AGREEMENT = 0.5         # below this the app's beats scatter against ours, so its offset means nothing


@dataclass
class ScanReport:
    analysed: list[str] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    flagged: dict[str, list[str]] = field(default_factory=dict)
    warnings: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        lines = [f"{len(self.analysed)} analysed, {len(self.compared)} compared with the app, "
                 f"{len(self.skipped)} already done, {len(self.flagged)} flagged, "
                 f"{len(self.errors)} failed"]
        lines += [f"  FLAG {tid}: {', '.join(f)}" for tid, f in sorted(self.flagged.items())]
        lines += [f"  WARN {tid}: {w}" for tid, w in sorted(self.warnings.items())]
        lines += [f"  FAIL {tid}: {e}" for tid, e in sorted(self.errors.items())]
        return "\n".join(lines)


def flags_for(grid: Grid, key: KeyResult, app: dict, lossless: bool) -> list[str]:
    flags = []
    if grid.inlier_ratio < 0.9:
        flags.append("grid-fit")
    if grid.bar_agreement < 0.8:
        flags.append("bar-phase")
    if grid.phase_agreement < MIN_PHASE_AGREEMENT:
        flags.append("phase")
    if key.disagrees:
        flags.append("key")
    return flags + _app_flags(grid.bpm, app, lossless)


def _app_flags(bpm: float, app: dict, lossless: bool) -> list[str]:
    if not app:
        return []
    flags = ["app-bpm"] if abs(app["app_bpm"] - bpm) > 0.05 else []
    if app.get("app_agreement", 1.0) < MIN_APP_AGREEMENT:
        return flags
    if app["half_beat"]:
        flags.append("half-beat")
    elif lossless and abs(app["app_offset_ms"]) > 20:
        flags.append("app-offset")
    return flags


def _app_fields(app: str, compared: dict) -> dict:
    trusted = compared.get("app_agreement", 1.0) >= MIN_APP_AGREEMENT
    return {"app": app if compared else "", "app_bpm": compared.get("app_bpm"),
            "app_offset_ms": compared.get("app_offset_ms") if compared and trusted else None}


def _app_file_is_lossless(record: TrackRecord, app: str) -> bool:
    """rekordbox is compared on the master; Serato on the source file it has tagged."""
    measured = record.master if app.startswith("rekordbox") else record.source
    return Path(measured).suffix.lower() in LOSSLESS


def _compare(grid: Grid, app_beats) -> dict:
    return compare_grids(grid, app_beats) if app_beats is not None and len(app_beats) > 1 else {}


def compare_app(record: TrackRecord, app: str, app_beats) -> TrackRecord:
    """Only the DJ-app cross-check, for a track already analysed: no model, no audio."""
    grid = Grid(period=60.0 / (record.bpm_fitted or record.bpm), phase=record.phase,
                bar_phase=record.bar_phase, beats_per_bar=record.beats_per_bar)
    compared = _compare(grid, app_beats)
    kept = [f for f in record.flags.split(";") if f and f not in APP_FLAGS]
    flags = kept + _app_flags(grid.bpm, compared, _app_file_is_lossless(record, app))
    return replace(record, **_app_fields(app, compared), flags=";".join(flags))


def analyse_record(record: TrackRecord, *, detector: Detector | None = None,
                   keyfinder: Callable = key_from_keyfinder, app_beats=None, app: str = "",
                   beats_dir=None) -> TrackRecord:
    audio, sr = read_audio(record.master)
    mono = audio.mean(axis=1)
    grid, beats, downbeats = analyse_grid(mono, sr, detector=detector)
    if beats_dir is not None:
        save_beats(Path(beats_dir) / f"{record.track_id}.npz", beats, downbeats)
    key = detect_key(record.master, keyfinder=keyfinder)
    compared = _compare(grid, app_beats)
    flags = flags_for(grid, key, compared, _app_file_is_lossless(record, app))
    return replace(
        record, sample_rate=sr, duration=len(mono) / sr, bpm=nominal_bpm(grid.bpm),
        bpm_fitted=grid.bpm, phase=grid.phase, bar_phase=grid.bar_phase,
        beats_per_bar=grid.beats_per_bar, inlier_ratio=grid.inlier_ratio,
        bar_agreement=grid.bar_agreement, phase_agreement=grid.phase_agreement,
        phase_offset_ms=grid.phase_offset * 1000,
        key=key.key.camelot if key.key else "", key_source=key.source,
        key_alt=key.alternative.camelot if key.alternative else "",
        **_app_fields(app, compared), flags=";".join(flags))


def scan_tracks(tracks_csv, *, only: set[str] | None = None, force: bool = False,
                detector: Detector | None = None, keyfinder: Callable = key_from_keyfinder,
                app: str = "", app_beats_for: Callable | None = None, beats_dir=None) -> ScanReport:
    records = load_tracks(tracks_csv)
    report = ScanReport()
    for track_id, record in sorted(records.items()):
        if only and track_id not in only:
            continue
        done = bool(record.bpm) and not force
        if done and not (app_beats_for and not record.app):
            report.skipped.append(track_id)
            continue
        beats = None
        if app_beats_for:
            try:
                beats = app_beats_for(record)
            except Exception as exc:        # a malformed tag in one file mustn't stop the scan
                report.warnings[track_id] = f"can't read the {app} grid ({type(exc).__name__}: {exc})"
        if done and beats is None:
            report.skipped.append(track_id)
            continue
        try:
            if done:
                updated = compare_app(record, app, beats)
            else:
                updated = analyse_record(record, detector=detector, keyfinder=keyfinder,
                                         app_beats=beats, app=app, beats_dir=beats_dir)
        except (GridError, AudioReadError, OSError) as exc:
            report.errors[track_id] = str(exc)
            continue
        except Exception as exc:            # one bad file mustn't stop the other tracks
            report.errors[track_id] = f"{type(exc).__name__}: {exc}"
            continue
        records[track_id] = updated
        (report.compared if done else report.analysed).append(track_id)
        if updated.flags:
            report.flagged[track_id] = updated.flags.split(";")
        save_tracks(tracks_csv, records)
    return report
