"""Command line interface.

    loopcutter cut manifests/session-01.csv --out loops/
    loopcutter cut manifests/session-01.csv --out loops/ --dry-run
    loopcutter inspect path/to/track.aiff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from collections import defaultdict

import soundfile as _sf

from .cut import cut_loop
from .manifest import ManifestError, load_manifest
from .keyreport import load_keys, render, render_histogram, render_sweep
from .keys import COMFORT_SEMITONES, MAX_POSSIBLE_SEMITONES, plan_sessions, sweep
from .naming import build_filename
from .timing import tiling_error_samples
from .verify import verify_cut


def _cmd_cut(args: argparse.Namespace) -> int:
    try:
        specs = load_manifest(args.manifest, args.audio_root)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2

    print(f"{len(specs)} loop(s) in {args.manifest}")
    for line in _tiling_warnings(specs):
        print(f"  note: {line}")

    if args.dry_run:
        for spec in specs:
            print(
                f"  row {spec.row_number}: {spec.slug}  "
                f"{spec.bars} bars @ {spec.bpm} BPM from {spec.start_seconds:.3f}s  "
                f"<- {spec.source.name}"
            )
        return 0

    failures = 0
    for spec in specs:
        try:
            result = cut_loop(
                spec,
                out_dir=args.out,
                fmt=args.format,
                subtype=args.subtype,
                snap_ms=args.snap_ms,
                fade_ms=args.fade_ms,
                filename=build_filename(spec, args.format),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"  FAIL row {spec.row_number}: {exc}", file=sys.stderr)
            failures += 1
            continue

        report = verify_cut(result, check_bpm=args.check_bpm)
        if report.ok:
            snap = (f" snap {result.snap_offset_samples:+d}"
                    if result.snap_offset_samples else "")
            print(f"  ok  {result.output.name}  "
                  f"{result.length_samples} samples{snap}")
        else:
            failures += 1
            print(f"  FAIL {result.output.name}", file=sys.stderr)
            for check in report.failures:
                print(f"         {check.name}: {check.detail}", file=sys.stderr)

    if failures:
        print(f"\n{failures} of {len(specs)} failed", file=sys.stderr)
        return 1

    print(f"\nall {len(specs)} loops written to {args.out}")
    return 0


def _tiling_warnings(specs) -> list[str]:
    """Flag variation sets whose rounded lengths do not divide evenly.

    Only a warning: Live re-warps to its own grid, so the practical impact is
    small. But it is worth knowing before you stack a 1-bar loop against a
    4-bar one for five minutes.
    """
    groups = defaultdict(list)
    for spec in specs:
        groups[(str(spec.source), spec.label, spec.start_seconds)].append(spec)

    messages = []
    for members in groups.values():
        if len(members) < 2:
            continue
        try:
            sample_rate = _sf.info(str(members[0].source)).samplerate
        except Exception:
            continue
        longest = max(members, key=lambda s: s.bars)
        for spec in members:
            if spec is longest:
                continue
            drift = tiling_error_samples(
                longest.bars, spec.bars, spec.bpm, sample_rate, spec.beats_per_bar
            )
            if drift:
                messages.append(
                    f"{spec.slug}: {spec.bars}-bar does not tile exactly into "
                    f"{longest.bars}-bar at {spec.bpm} BPM / {sample_rate} Hz "
                    f"({drift} sample(s) out). 48 kHz sources avoid this."
                )
    return messages


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopcutter")
    sub = parser.add_subparsers(dest="command", required=True)

    cut = sub.add_parser("cut", help="cut every loop in a manifest")
    cut.add_argument("manifest")
    cut.add_argument("--out", default="loops", help="output directory")
    cut.add_argument("--audio-root", default=None,
                     help="base directory for relative source paths "
                          "(defaults to the current working directory)")
    cut.add_argument("--format", default="aiff", choices=["aiff", "wav", "flac"])
    cut.add_argument("--subtype", default="PCM_24")
    cut.add_argument("--snap-ms", type=float, default=2.0)
    cut.add_argument("--fade-ms", type=float, default=0.5)
    cut.add_argument("--check-bpm", action="store_true",
                     help="run the librosa tempo check (slow)")
    cut.add_argument("--dry-run", action="store_true")
    cut.set_defaults(func=_cmd_cut)

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
