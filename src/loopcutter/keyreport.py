"""Reading key data out of a spreadsheet and rendering the plan.

Accepts CSV, XLSX or a loopcutter manifest. It looks for a column named key,
camelot, initial key or initialkey, case-insensitively, so it works on exports
from Mixed In Key, Serato and rekordbox without editing them first.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .keys import (COMFORT_SEMITONES, Key, KeyParseError, Plan, SweepRow,
                   key_histogram, parse_key)

KEY_COLUMNS = ("key", "camelot", "initial key", "initialkey", "initial_key", "tone")
LABEL_COLUMNS = ("label", "track", "title", "name", "sample", "file", "filename")
ARTIST_COLUMNS = ("artist", "source")


@dataclass
class LoadResult:
    entries: list[tuple[str, Key]]
    total_rows: int
    unparsed: list[tuple[str, str]]
    no_key: int


def _pick(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _rows_from_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        return list(reader), list(reader.fieldnames)


def _rows_from_xlsx(path: Path, sheet: str | None) -> tuple[list[dict], list[str]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "reading .xlsx needs openpyxl: pip install 'loopcutter[sheets]'"
        ) from exc

    book = load_workbook(str(path), data_only=True)
    ws = book[sheet] if sheet else book[book.sheetnames[0]]

    header_row = None
    for row in ws.iter_rows(min_row=1, max_row=12, values_only=True):
        cells = [str(c).strip().lower() if c is not None else "" for c in row]
        if any(c in KEY_COLUMNS for c in cells):
            header_row = row
            break
    if header_row is None:
        raise ValueError(
            f"no key column found in the first 12 rows of {path.name}"
            f" (sheet {ws.title!r}). Looked for: {', '.join(KEY_COLUMNS)}"
        )

    headers = [str(c).strip() if c is not None else "" for c in header_row]
    start = None
    for index, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row == header_row:
            start = index + 1
            break

    rows = []
    for row in ws.iter_rows(min_row=start, values_only=True):
        if all(c is None or str(c).strip() == "" for c in row):
            continue
        rows.append({h: ("" if v is None else str(v))
                     for h, v in zip(headers, row) if h})
    return rows, headers


def load_keys(path: str | Path, sheet: str | None = None) -> LoadResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        rows, headers = _rows_from_xlsx(path, sheet)
    else:
        rows, headers = _rows_from_csv(path)

    key_col = _pick(headers, KEY_COLUMNS)
    if key_col is None:
        raise ValueError(
            f"no key column in {path.name}. Looked for: {', '.join(KEY_COLUMNS)}"
        )
    label_col = _pick(headers, LABEL_COLUMNS)
    artist_col = _pick(headers, ARTIST_COLUMNS)

    entries: list[tuple[str, Key]] = []
    unparsed: list[tuple[str, str]] = []
    no_key = 0

    for number, row in enumerate(rows, start=2):
        raw = (row.get(key_col) or "").strip()
        parts = [row.get(artist_col) or "", row.get(label_col) or ""]
        label = " - ".join(p for p in parts if p) or f"row {number}"

        if not raw:
            no_key += 1
            continue
        try:
            entries.append((label, parse_key(raw)))
        except KeyParseError:
            unparsed.append((label, raw))

    return LoadResult(entries=entries, total_rows=len(rows),
                      unparsed=unparsed, no_key=no_key)


def _shift_text(shift: int, comfort: int = COMFORT_SEMITONES) -> str:
    if shift == 0:
        return "as is"
    text = f"{shift:+d} semitone{'s' if abs(shift) != 1 else ''}"
    if abs(shift) > comfort:
        text += "  <-- stretched, audition this"
    return text


def render(plan: Plan, load: LoadResult, verbose: bool = False,
           comfort: int = COMFORT_SEMITONES) -> str:
    out: list[str] = []
    add = out.append

    add(f"{load.total_rows} rows, {plan.total_keyed} with a readable key")
    if load.no_key:
        add(f"  {load.no_key} with no key (percussion and texture, presumably)")
    if load.unparsed:
        add(f"  {len(load.unparsed)} unreadable:")
        for label, raw in load.unparsed[:8]:
            add(f"    {label}: {raw!r}")
        if len(load.unparsed) > 8:
            add(f"    ...and {len(load.unparsed) - 8} more")
    add("")
    add(f"Transposition limit: +/-{plan.max_semitones} semitones"
        + (", relative major/minor allowed" if plan.allow_relative else ""))
    if plan.max_semitones > comfort:
        add(f"  Beyond +/-{comfort} is past the usual comfort zone. Loops needing")
        add("  more than that are marked below. Trust your ears, not the table.")
    add("")

    if not plan.options:
        add("No session key covers anything. Check the key column.")
        return "\n".join(out)

    running = 0
    for index, option in enumerate(plan.options, start=1):
        running += option.count
        share = 100 * option.count / plan.total_keyed if plan.total_keyed else 0
        add(f"Session key {index}: {option.key}")
        noun = "loop" if option.count == 1 else "loops"
        add(f"  covers {option.count} {noun} ({share:.0f}%), "
            f"{option.untransposed} of them untransposed")
        add(f"  running total: {running}/{plan.total_keyed} "
            f"({100 * running / plan.total_keyed:.0f}%)")

        if index == 1 and plan.runners_up:
            close = ", ".join(f"{k.camelot} ({n})" for k, n in plan.runners_up)
            add(f"  next best were: {close}")

        if verbose:
            add("  loops:")
            for a in sorted(option.covered, key=lambda x: (abs(x.shift), x.label)):
                via = " via relative" if a.via_relative else ""
                add(f"    {a.label}  [{a.key.camelot}]  "
                    f"{_shift_text(a.shift, comfort)}{via}")
        add("")

    if plan.unreachable:
        add(f"Not covered by {len(plan.options)} key"
            f"{'s' if len(plan.options) != 1 else ''}: {len(plan.unreachable)} loops")
        for label, key in sorted(plan.unreachable, key=lambda e: e[1].camelot):
            add(f"  {label}  [{key.camelot}] {key.name}")
        add("")
        add("Options: raise the transposition limit, add a fourth key, allow")
        add("relative major/minor with --relative, or leave these out of the set.")
    else:
        add(f"All {plan.total_keyed} keyed loops are covered.")

    return "\n".join(out)


def render_histogram(entries: list[tuple[str, Key]]) -> str:
    rows = key_histogram(entries)
    if not rows:
        return "No keys to chart."
    widest = max(count for _, count in rows)
    lines = ["Key distribution:"]
    for key, count in rows:
        bar = "#" * max(1, round(24 * count / widest))
        lines.append(f"  {key.camelot:>3}  {key.name:<9} {count:>4}  {bar}")
    return "\n".join(lines)


def render_sweep(rows: list[SweepRow], max_keys: int,
                 comfort: int = COMFORT_SEMITONES) -> str:
    """Show what relaxing the transposition limit actually buys."""
    lines = [
        f"Coverage by transposition limit (with {max_keys} session key"
        f"{'s' if max_keys != 1 else ''}):",
        "",
        "  limit   covered        keys to cover all   stretched",
    ]
    previous = None
    for row in rows:
        full = str(row.keys_for_full) if row.keys_for_full else ">8"
        gain = ""
        if previous is not None and row.covered > previous:
            gain = f"  (+{row.covered - previous})"
        flag = " *" if row.semitones > comfort else "  "
        lines.append(
            f"  +/-{row.semitones}{flag}  {row.covered:>3}/{row.total} "
            f"({row.share:>3.0f}%){gain:<7}      {full:>3}"
            f"              {row.stretched if row.stretched else '-':>3}"
        )
        previous = row.covered

    lines.append("")
    lines.append(f"  * beyond the usual +/-{comfort} comfort zone")
    lines.append("")
    lines.append("Read the middle column first. Dropping from three session keys")
    lines.append("to two is usually worth more than a few percent of coverage.")
    return "\n".join(lines)
