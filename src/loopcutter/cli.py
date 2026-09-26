"""Command line interface.

    loopcutter init ~/loops-workspace          private workspace with its folders
    loopcutter prep                            lossless working masters
    loopcutter scan --app rekordbox            grid, phase, key and doubts per track
    loopcutter import --from rekordbox         your marked loops, snapped, as a manifest
    loopcutter resolve manifests/set.csv       fill and snap hand-typed rows
    loopcutter cut manifests/set.csv           cut, verify, tag and file
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import soundfile as _sf

from .cut import cut_loop, cut_oneshot
from .keyreport import load_keys, render, render_histogram, render_sweep
from .keys import (COMFORT_SEMITONES, MAX_POSSIBLE_SEMITONES, KeyParseError, parse_key,
                   plan_sessions, sweep)
from .manifest import ManifestError, load_manifest
from .model import TrackRecord
from .naming import build_filename, library_subdir
from .timing import parse_position, rates_that_tile, tiling_error_ms, tiling_error_samples
from .trackdb import load_tracks, save_tracks, track_id_for
from .verify import Check, verify_cut
from .workspace import DIRS, WorkspaceError, find_workspace, init_workspace

LIVE_INFERS_BARS = {1, 2, 4, 8, 16}


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _workspace_or_exit():
    ws = find_workspace()
    if ws is None:
        print("error: not inside a loopcutter workspace - run `loopcutter init DIR` and cd into it",
              file=sys.stderr)
        raise SystemExit(2)
    return ws


def _tiling_warnings(specs) -> list[str]:
    """Variation sets whose rounded lengths don't divide evenly. A warning only: DAWs
    that warp clips re-sync them, but it is worth knowing before stacking them."""
    groups = defaultdict(list)
    for spec in specs:
        groups[(str(spec.source), spec.label, spec.start_seconds)].append(spec)
    messages = []
    for members in groups.values():
        members = [m for m in members if m.kind == "loop"]       # one-shots have no tempo to tile
        if len(members) < 2:
            continue
        try:
            rate = _sf.info(str(members[0].source)).samplerate
        except RuntimeError:
            continue
        longest = max(members, key=lambda s: s.bars)
        for spec in (m for m in members if m is not longest):
            drift = tiling_error_samples(longest.bars, spec.bars, spec.bpm, rate, spec.beats_per_bar)
            if drift:
                ms = tiling_error_ms(longest.bars, spec.bars, spec.bpm, rate, spec.beats_per_bar)
                exact = [r for r in rates_that_tile(spec.bars, spec.bpm, spec.beats_per_bar) if r != rate]
                hint = f" Exact at {exact[0]} Hz." if exact else ""
                messages.append(f"{spec.slug}: drifts {drift} sample(s) ({ms:.3f} ms) per "
                                f"{longest.bars:g}-bar cycle at {spec.bpm:g} BPM / {rate} Hz; DAWs "
                                f"that warp clips re-sync it.{hint}")
    return messages


def _cmd_init(args) -> int:
    try:
        ws = init_workspace(args.directory)
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"workspace ready at {ws.root}")
    for sub in DIRS:
        print(f"  {sub}/")
    return 0


def _cmd_prep(args) -> int:
    from .prep import check_collisions, find_sources, prepare_master
    from .tags import copy_basic_tags

    ws = _workspace_or_exit()
    sources = find_sources(args.sources or ws.setting("sources", "paths", []))
    if not sources:
        print("no audio found - pass files or folders, or set [sources] paths in loopcutter.toml",
              file=sys.stderr)
        return 2
    try:
        check_collisions(sources, ws.masters)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    records = load_tracks(ws.tracks_csv)
    made = stale = 0
    for source in sources:
        result = prepare_master(source, ws.masters, rate=int(ws.setting("audio", "sample_rate", 48000)),
                                subtype=ws.setting("audio", "subtype", "PCM_24"))
        if result.stale:
            stale += 1
            print(f"  STALE {result.master.name}: its source changed; delete the master to rebuild it")
        if not result.reused:
            copy_basic_tags(source, result.master)
            made += 1
            clip = f"  CLIPPED {result.clipped} sample(s)" if result.clipped else ""
            print(f"  {result.master.name}  {result.source_rate} -> {result.rate} Hz{clip}")
        track_id = track_id_for(result.master)
        if track_id not in records:
            records[track_id] = TrackRecord(track_id=track_id, source=str(Path(source).resolve()),
                                            master=str(result.master), sample_rate=result.rate,
                                            duration=result.frames / result.rate,
                                            bpm=0.0, phase=0.0, bar_phase=0)
        elif Path(records[track_id].source).resolve() != Path(source).resolve():
            print(f"  CHANGED {result.master.name}: built from {records[track_id].source}; "
                  "delete the master and its tracks.csv row to rebuild it from this source")
    save_tracks(ws.tracks_csv, records)
    print(f"{made} master(s) written, {len(sources) - made} already there ({stale} stale), in {ws.masters}")
    return 0


def _app_beats(app):
    if app == "serato":
        from .markers import serato_beats

        return lambda record: serato_beats(record.source)
    if app == "rekordbox":
        from .markers import open_rekordbox, rekordbox_beats

        db = open_rekordbox()
        return lambda record: rekordbox_beats(db, record.master)
    return None


def _cmd_scan(args) -> int:
    from .analyse import scan_tracks

    ws = _workspace_or_exit()
    report = scan_tracks(ws.tracks_csv, only=set(args.tracks) or None, force=args.force,
                         app=args.app or "", app_beats_for=_app_beats(args.app),
                         beats_dir=ws.beats_dir)
    text = report.text()
    print(text)
    ws.reports.mkdir(parents=True, exist_ok=True)
    (ws.reports / f"scan-{_stamp()}.txt").write_text(text + "\n")
    return 1 if report.errors else 0


def _beats_for(ws):
    from .grid import load_beats

    def load(record):
        path = ws.beats_dir / f"{record.track_id}.npz"
        return load_beats(path) if path.exists() else None
    return load


def _cmd_import(args) -> int:
    from .markers import from_rekordbox_db, from_rekordbox_xml, from_serato
    from .snap import markers_to_rows, write_manifest

    ws = _workspace_or_exit()
    tracks = load_tracks(ws.tracks_csv)
    playlist = args.playlist or ws.setting("marking", "playlist", "")
    try:
        if args.source == "serato":
            paths = [Path(p) for p in args.files] or [Path(r.source) for r in tracks.values()]
            markers = from_serato(paths)
        elif args.source == "rekordbox":
            if not playlist:
                raise ValueError("name the playlist with --playlist or [marking] playlist")
            markers = from_rekordbox_db(playlist)
        else:
            if not args.xml:
                raise ValueError("--from rekordbox-xml needs --xml FILE")
            markers = from_rekordbox_xml(Path(args.xml), playlist or None)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = markers_to_rows(markers, tracks, beats_for=_beats_for(ws),
                             max_shift_ms=args.max_shift_ms or ws.setting("snap", "max_shift_ms", 60))
    out = Path(args.out) if args.out else ws.manifests / f"import-{args.source}-{_stamp()}.csv"
    try:
        write_manifest(result.rows, out)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{len(markers)} marker(s) -> {len(result.rows)} row(s) in {out}")
    if result.moves_ms:
        moves = result.moves_ms
        print(f"  moved onto the grid: median {statistics.median(moves):+.1f} ms, "
              f"range {min(moves):+.1f} to {max(moves):+.1f} ms")
    for problem in result.problems:
        print(f"  SKIP {problem}", file=sys.stderr)
    return 1 if result.problems else 0


def _cmd_resolve(args) -> int:
    from .snap import SnapError, grid_at, snap_time

    ws = _workspace_or_exit()
    tracks = load_tracks(ws.tracks_csv)
    beats_for = _beats_for(ws)
    path = Path(args.manifest)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, restkey="_extra", restval="")
        fields, rows = list(reader.fieldnames or []), list(reader)
    ragged = [n for n, row in enumerate(rows, start=2) if "_extra" in row]
    if ragged:
        for number in ragged:
            print(f"  FAIL row {number}: more cells than the header - quote any value "
                  "that contains a comma", file=sys.stderr)
        print(f"{path} left unchanged", file=sys.stderr)
        return 2
    for column in ("source", "bpm", "key", "snap", "notes"):
        if column not in fields:
            fields.append(column)
    limit = args.max_shift_ms or ws.setting("snap", "max_shift_ms", 60)
    failures = changed = 0
    for number, row in enumerate(rows, start=2):
        track_id = row.get("track_id")
        mode = (row.get("snap") or "").strip().lower()
        if not track_id:
            if mode:
                print(f"  FAIL row {number}: snap needs a track_id", file=sys.stderr)
                failures += 1
            continue
        record = tracks.get(track_id)
        if record is None:
            print(f"  FAIL row {number}: {track_id!r} is not in tracks.csv", file=sys.stderr)
            failures += 1
            continue
        try:
            if mode not in {"", "beat", "bar"}:
                raise ValueError("snap must be beat, bar or empty")
            if mode and not row.get("start"):
                raise ValueError("snap needs a start")
            done = not mode and row.get("bpm") and row.get("source")
            if not row.get("start") and not row.get("bpm") and "grid-fit" in record.flags.split(";"):
                raise ValueError("this track's tempo moves, so give a start (or a bpm) for this row")
            at = row.get("start") or row.get("downbeat")
            start = parse_position(at) if at else 0.0
            grid, bpm = (None, 0.0) if done else grid_at(record, start, beats_for)
            if mode and "phase" in record.flags.split(";") and record.override_phase_ms is None:
                raise ValueError("this track's grid phase isn't confident - set override_phase_ms "
                                 "in tracks.csv, or mark it in rekordbox and import")
            moved = snap_time(start, grid, mode, limit) if mode else None
        except (SnapError, ValueError) as exc:
            print(f"  FAIL row {number}: {exc}", file=sys.stderr)
            failures += 1
            continue
        row["source"] = row.get("source") or record.master
        row["bpm"] = row.get("bpm") or f"{bpm:g}"
        row["key"] = row.get("key") or record.key
        if moved is not None:
            row["start"], row["snap"] = f"{moved.snapped:.6f}", ""
            row["notes"] = (row.get("notes", "") + f" resolved {moved.shift_ms:+.1f} ms").strip()
            print(f"  row {number}: start moved {moved.shift_ms:+.1f} ms onto the {mode}")
        changed += 1
    ws.reports.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, ws.reports / f"{path.stem}.{_stamp()}.bak.csv")
    temp = path.with_suffix(".partial.csv")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    print(f"{changed} row(s) resolved in {path}; backup in {ws.reports}")
    return 1 if failures else 0


def _cmd_stems(args) -> int:
    from .stems import DEFAULT_MODEL, conform_stems, separate

    ws = _workspace_or_exit()
    tracks = load_tracks(ws.tracks_csv)
    model = args.model or (DEFAULT_MODEL if args.engine == "audio-separator" else "htdemucs_6s")
    failures = 0
    for track_id in args.tracks:
        record = tracks.get(track_id)
        if record is None:
            print(f"  FAIL {track_id}: not in tracks.csv", file=sys.stderr)
            failures += 1
            continue
        try:
            raw = separate(record.master, ws.stems, engine=args.engine, model=model)
            stems = conform_stems(raw, record.master, ws.stems / track_id)
        except ImportError as exc:
            print(f"error: {exc.name or exc} isn't installed - install loopcutter[stems]", file=sys.stderr)
            return 2
        except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            print(f"  FAIL {track_id}: {exc}", file=sys.stderr)
            failures += 1
            continue
        clipped = sum(getattr(stems, "clipped", {}).values())
        level = f"{getattr(stems, 'gain_db', 0.0):+.2f} dB against the master"
        print(f"  {track_id}: " + ", ".join(sorted(stems)) + f" ({level}"
              + (f", {clipped} sample(s) clipped" if clipped else "") + ")")
    return 1 if failures else 0


def _tag(result, spec) -> None:
    from .tags import write_loop_tags

    try:
        key = parse_key(spec.key) if spec.key else None
    except KeyParseError:
        key = None
    length = "oneshot" if spec.kind == "oneshot" else f"{spec.bars:g}bar"
    write_loop_tags(result.output, bpm=spec.bpm or None, key=key, stem=spec.stem,
                    title=f"{spec.track or spec.source.stem} [{spec.label}][{length}]",
                    artist=spec.artist, comment=spec.label)


def _cmd_cut(args) -> int:
    ws = find_workspace()
    tracks_csv = args.tracks or (ws.tracks_csv if ws else None)
    tracks = load_tracks(tracks_csv) if tracks_csv else {}
    try:
        specs = load_manifest(args.manifest, args.audio_root)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    out_root = Path(args.out) if args.out else (ws.loops / args.format if ws else Path("loops"))
    layout = args.layout or ("library" if ws else "flat")
    tagging = args.tag if args.tag is not None else (ws is not None and args.format == "aiff")
    if tagging and args.format == "flac":
        print("error: --tag writes ID3 tags, which FLAC files don't use", file=sys.stderr)
        return 2

    print(f"{len(specs)} loop(s) in {args.manifest}")
    if importlib.util.find_spec("librosa") is None:
        print("  note: librosa isn't installed, so beat_alignment can't be judged; "
              "install loopcutter[verify]")
    for line in _tiling_warnings(specs):
        print(f"  note: {line}")
    for spec in specs:
        if spec.kind == "loop" and spec.bars not in LIVE_INFERS_BARS:
            print(f"  note: {spec.slug}: Live infers tempo only for 1, 2, 4, 8 and 16-bar clips; "
                  f"set this one to {spec.bpm:g} BPM by hand")
    if args.dry_run:
        for spec in specs:
            print(f"  row {spec.row_number}: {spec.slug}  {spec.bars:g} bars @ {spec.bpm:g} BPM "
                  f"from {spec.start_seconds:.6f}s  <- {spec.source.name}")
        return 0

    failures = 0
    for spec in specs:
        timing_source = None
        if spec.stem and spec.stem != "full" and spec.track_id and ws is not None:
            stem_file = ws.stems / spec.track_id / f"{spec.stem}.aiff"
            if not stem_file.exists():
                print(f"  FAIL row {spec.row_number}: no {spec.stem} stem yet - run "
                      f"`loopcutter stems \"{spec.track_id}\"`", file=sys.stderr)
                failures += 1
                continue
            # keep the track's name: the file is now bass.aiff, but the loop is still this track's
            timing_source = spec.source
            spec = replace(spec, source=stem_file, track=spec.track or spec.source.stem)
        elif spec.stem and spec.stem != "full":
            print(f"  note: row {spec.row_number}: the {spec.stem} row is cut from its source as given; "
                  "give it a track_id in a workspace to cut it from separated stems")
        record = tracks.get(spec.track_id) if spec.track_id else None
        out_dir = out_root / library_subdir(spec) if layout == "library" else out_root
        try:
            if spec.kind == "oneshot":
                result = cut_oneshot(spec, out_dir=out_dir, fmt=args.format, subtype=args.subtype,
                                     snap_ms=args.snap_ms, trim_db=args.trim_db,
                                     filename=build_filename(spec, args.format))
            else:
                result = cut_loop(spec, out_dir=out_dir, fmt=args.format, subtype=args.subtype,
                                  snap_ms=args.snap_ms, fade_ms=args.fade_ms, trim_db=args.trim_db,
                                  xfade_ms=args.xfade_ms if spec.xfade_ms is None else spec.xfade_ms,
                                  filename=build_filename(spec, args.format))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  FAIL row {spec.row_number}: {exc}", file=sys.stderr)
            failures += 1
            continue
        result.timing_source = timing_source
        # A track whose tempo moves has no single tempo to check against; measure the audio.
        steady = record is not None and ("grid-fit" not in record.flags.split(";") or record.override_bpm)
        reference = (record.override_bpm or record.bpm_fitted or record.bpm) if steady else None
        report = verify_cut(result, check_bpm=args.check_bpm or record is not None,
                            reference_bpm=reference)
        if record is not None and not steady and spec.kind == "loop":
            from .snap import same_tempo

            track_bpm = record.bpm_fitted or record.bpm
            if track_bpm and not same_tempo(spec.bpm, track_bpm):
                report.checks.append(Check("bpm_match", False,
                                           f"declared {spec.bpm:g}, the track runs at {track_bpm:.2f}: "
                                           "a half- or double-time reading of the beats"))
        if args.lenient:
            for check in report.checks:
                if check.name == "beat_alignment" and not check.passed:
                    check.passed, check.warn = True, True
        if tagging and report.ok:
            _tag(result, spec)
            if _sf.info(str(result.output)).frames != result.length_samples:
                report.checks.append(Check("tagging", False, "tagging changed the audio length"))
        if not report.ok:
            failures += 1
            result.output.unlink(missing_ok=True)          # a loop that failed its checks never reaches the library
            print(f"  FAIL {result.output.name} (not kept)", file=sys.stderr)
            for check in report.failures:
                print(f"         {check.name}: {check.detail}", file=sys.stderr)
            continue
        snap = f" snap {result.snap_offset_samples:+d}" if result.snap_offset_samples else ""
        print(f"  ok  {result.output.relative_to(out_root)}  {result.length_samples} samples{snap}")
        for check in report.warnings:
            print(f"      warn {check.name}: {check.detail}")

    if failures:
        print(f"\n{failures} of {len(specs)} failed", file=sys.stderr)
        return 1
    print(f"\nall {len(specs)} loops written to {out_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopcutter")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create a private workspace (masters, manifests, loops ...)")
    init.add_argument("directory", nargs="?", default=".")
    init.set_defaults(func=_cmd_init)

    prep = sub.add_parser("prep", help="make lossless working masters at the workspace rate")
    prep.add_argument("sources", nargs="*", help="files or folders (default: [sources] paths)")
    prep.set_defaults(func=_cmd_prep)

    scan = sub.add_parser("scan", help="analyse masters: grid, phase, key, app cross-check")
    scan.add_argument("tracks", nargs="*", help="track_ids to analyse (default: all pending)")
    scan.add_argument("--app", choices=["serato", "rekordbox"], default=None)
    scan.add_argument("--force", action="store_true", help="re-analyse tracks already done")
    scan.set_defaults(func=_cmd_scan)

    imp = sub.add_parser("import", help="turn loops marked in DJ software into a resolved manifest")
    imp.add_argument("--from", dest="source", required=True, choices=["rekordbox", "rekordbox-xml", "serato"])
    imp.add_argument("files", nargs="*", help="Serato: audio files (default: every source)")
    imp.add_argument("--playlist", default=None)
    imp.add_argument("--xml", default=None, help="rekordbox XML export (for rekordbox-xml)")
    imp.add_argument("--out", default=None)
    imp.add_argument("--max-shift-ms", type=float, default=None)
    imp.set_defaults(func=_cmd_import)

    res = sub.add_parser("resolve", help="fill rows from tracks.csv and snap starts marked snap=beat|bar")
    res.add_argument("manifest")
    res.add_argument("--max-shift-ms", type=float, default=None)
    res.set_defaults(func=_cmd_resolve)

    cut = sub.add_parser("cut", help="cut every loop in a resolved manifest")
    cut.add_argument("manifest")
    cut.add_argument("--out", default=None, help="default: loops/<format> in a workspace, else ./loops")
    cut.add_argument("--audio-root", default=None,
                     help="base directory for relative source paths "
                          "(defaults to the current working directory)")
    cut.add_argument("--tracks", default=None, help="tracks.csv for the tempo check (default: the workspace's)")
    cut.add_argument("--layout", choices=["flat", "library"], default=None,
                     help="library files loops as <stem>/<tempo band>/ (default in a workspace)")
    cut.add_argument("--tag", action=argparse.BooleanOptionalAction, default=None,
                     help="write BPM, key and stem tags (default: on for AIFF in a workspace)")
    cut.add_argument("--format", default="aiff", choices=["aiff", "wav", "flac"])
    cut.add_argument("--subtype", default="PCM_24")
    cut.add_argument("--snap-ms", type=float, default=2.0)
    cut.add_argument("--fade-ms", type=float, default=0.5)
    cut.add_argument("--trim-db", type=float, default=0.0, help="gain for every loop, e.g. -1.0")
    cut.add_argument("--xfade-ms", type=float, default=0.0,
                     help="crossfade the loop's tail into the audio before its start (a row's xfade_ms wins)")
    cut.add_argument("--check-bpm", action="store_true",
                     help="measure the tempo from the audio (automatic for analysed tracks)")
    cut.add_argument("--lenient", action="store_true",
                     help="report beat_alignment failures as warnings (for unsteady material)")
    cut.add_argument("--dry-run", action="store_true")
    cut.set_defaults(func=_cmd_cut)

    stems = sub.add_parser("stems", help="separate masters into stems at the master's rate")
    stems.add_argument("tracks", nargs="+", help="track_ids from tracks.csv")
    stems.add_argument("--engine", choices=["audio-separator", "demucs"], default="audio-separator")
    stems.add_argument("--model", default=None)
    stems.set_defaults(func=_cmd_stems)

    keys = sub.add_parser(
        "keys",
        help="work out the best session key(s) for a library of loops")
    keys.add_argument("path", help="CSV, XLSX or manifest with a key column")
    keys.add_argument("--keys", type=int, default=3,
                      help="how many session keys to plan for (default 3)")
    keys.add_argument("--semitones", type=int, default=3,
                      help="transposition limit either way, 0-6 (default 3). "
                           "4 and 5 are past the usual comfort zone but are "
                           "guidance, not physics")
    keys.add_argument("--comfort", type=int, default=COMFORT_SEMITONES,
                      help="shifts beyond this are flagged for audition "
                           f"(default {COMFORT_SEMITONES})")
    keys.add_argument("--sweep", action="store_true",
                      help="tabulate coverage at every limit from 0 to 6")
    keys.add_argument("--relative", action="store_true",
                      help="also allow relative major/minor as a free match")
    keys.add_argument("--sheet", default=None,
                      help="worksheet name, for multi-sheet xlsx files")
    keys.add_argument("--histogram", action="store_true",
                      help="also print the key distribution")
    keys.add_argument("-v", "--verbose", action="store_true",
                      help="list every loop and its transposition")
    keys.set_defaults(func=_cmd_keys)

    inspect = sub.add_parser("inspect", help="print audio file facts")
    inspect.add_argument("path")
    inspect.add_argument("--bpm", type=float, default=None)
    inspect.set_defaults(func=_cmd_inspect)

    return parser


def _cmd_inspect(args: argparse.Namespace) -> int:
    import soundfile as sf

    info = sf.info(args.path)
    print(f"{args.path}")
    print(f"  {info.samplerate} Hz, {info.channels} ch, {info.subtype}")
    print(f"  {info.frames} frames, {info.duration:.3f} s")
    if args.bpm:
        beats = info.duration * args.bpm / 60.0
        print(f"  at {args.bpm} BPM that is {beats:.4f} beats "
              f"({beats / 4:.4f} bars)")
    return 0


def _cmd_keys(args: argparse.Namespace) -> int:
    try:
        load = load_keys(args.path, sheet=args.sheet)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not load.entries:
        print("No readable keys found.", file=sys.stderr)
        return 1

    if not 0 <= args.semitones <= MAX_POSSIBLE_SEMITONES:
        print(f"error: --semitones must be 0 to {MAX_POSSIBLE_SEMITONES}. "
              "A tritone is the furthest two keys can be; past that you are "
              "travelling the other way round the circle.", file=sys.stderr)
        return 2

    plan = plan_sessions(
        load.entries,
        max_keys=args.keys,
        max_semitones=args.semitones,
        allow_relative=args.relative,
        total_rows=load.total_rows,
    )
    print(render(plan, load, verbose=args.verbose, comfort=args.comfort))

    if args.sweep:
        print()
        print(render_sweep(
            sweep(load.entries, max_keys=args.keys,
                  allow_relative=args.relative, comfort=args.comfort),
            max_keys=args.keys, comfort=args.comfort))

    if args.histogram:
        print()
        print(render_histogram(load.entries))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        # Piped into head or less and the reader went away. Not an error.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
