"""Session-key planning.

Given a library of loops with known keys, work out which session key lets you
use the most of them, which ones that excludes, and what second and third keys
would pick up the remainder.

The model is transposition, not harmonic mixing. A loop is usable in a session
key if you can pitch it there without audible artifacts—consensus is about
three semitones either way. Beyond that you should find a different loop rather
than force the one you have.

Relative major/minor is offered as an option (--relative) because the two share
the same notes: an A minor loop sits inside C major untransposed. It is off by
default because sharing notes is not the same as sharing a tonic, and a loop
with a strong root note will still pull against the session.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# C = 0 .. B = 11
PITCH_CLASSES = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot wheel. The A series is minor, the B series its relative major.
CAMELOT_MINOR = {1: 8, 2: 3, 3: 10, 4: 5, 5: 0, 6: 7,
                 7: 2, 8: 9, 9: 4, 10: 11, 11: 6, 12: 1}
CAMELOT_MAJOR = {1: 11, 2: 6, 3: 1, 4: 8, 5: 3, 6: 10,
                 7: 5, 8: 0, 9: 7, 10: 2, 11: 9, 12: 4}

_TO_CAMELOT = {}
for _n, _pc in CAMELOT_MINOR.items():
    _TO_CAMELOT[(_pc, "minor")] = f"{_n}A"
for _n, _pc in CAMELOT_MAJOR.items():
    _TO_CAMELOT[(_pc, "major")] = f"{_n}B"

_CAMELOT_RE = re.compile(r"^\s*(\d{1,2})\s*([AB])\s*$", re.IGNORECASE)
_NAMED_RE = re.compile(
    r"^\s*([A-Ga-g][#b♯♭]?)\s*"
    r"(m|min|minor|maj|major|M)?\s*$"
)

MAX_SEMITONES_DEFAULT = 3
COMFORT_SEMITONES = 3
MAX_POSSIBLE_SEMITONES = 6  # a tritone; beyond this you are going the other way


class KeyParseError(ValueError):
    """Raised when a key string cannot be interpreted."""


@dataclass(frozen=True)
class Key:
    """A musical key as a tonic pitch class plus a mode."""

    pitch_class: int
    mode: str  # "major" or "minor"

    @property
    def camelot(self) -> str:
        return _TO_CAMELOT[(self.pitch_class, self.mode)]

    @property
    def name(self) -> str:
        suffix = "minor" if self.mode == "minor" else "major"
        return f"{PC_NAMES[self.pitch_class]} {suffix}"

    def __str__(self) -> str:
        return f"{self.name} ({self.camelot})"

    @property
    def relative(self) -> "Key":
        """The relative major of a minor key, or relative minor of a major."""
        if self.mode == "minor":
            return Key((self.pitch_class + 3) % 12, "major")
        return Key((self.pitch_class - 3) % 12, "minor")


def parse_key(text: str) -> Key:
    """Parse Camelot ("8A") or named ("Am", "F# minor", "Bb") notation.

    Open Key notation (1m / 1d) is not supported—it is offset from Camelot by
    five positions and silently misreading it would be worse than refusing.
    """
    if text is None:
        raise KeyParseError("empty key")
    raw = str(text).strip().replace("♯", "#").replace("♭", "b")
    if not raw:
        raise KeyParseError("empty key")

    match = _CAMELOT_RE.match(raw)
    if match:
        number = int(match.group(1))
        letter = match.group(2).upper()
        if not 1 <= number <= 12:
            raise KeyParseError(f"Camelot number out of range: {raw!r}")
        if letter == "A":
            return Key(CAMELOT_MINOR[number], "minor")
        return Key(CAMELOT_MAJOR[number], "major")

    match = _NAMED_RE.match(raw)
    if match:
        tonic = match.group(1)
        quality = (match.group(2) or "").strip()
        pc = PITCH_CLASSES.get(tonic.upper())
        if pc is None:
            raise KeyParseError(f"unknown tonic in {raw!r}")
        # Bare "m" is minor; bare "M", "maj" or nothing at all is major.
        mode = "minor" if quality.lower() in {"m", "min", "minor"} else "major"
        if quality == "M":
            mode = "major"
        return Key(pc, mode)

    raise KeyParseError(f"cannot parse key {raw!r}")


def semitone_distance(a: int, b: int) -> int:
    """Shortest distance between two pitch classes, 0 to 6."""
    gap = abs(a - b) % 12
    return min(gap, 12 - gap)


def signed_shift(source: int, target: int) -> int:
    """Semitones to move `source` onto `target`, choosing the shorter route.

    Positive is up. Ties (a tritone) resolve downward, which is the gentler
    choice for loops with bass content.
    """
    up = (target - source) % 12
    down = up - 12
    return up if up < abs(down) else down


def is_reachable(loop: Key, session: Key, max_semitones: int,
                 allow_relative: bool) -> bool:
    if loop.mode == session.mode:
        return semitone_distance(loop.pitch_class, session.pitch_class) <= max_semitones
    if allow_relative:
        rel = loop.relative
        return semitone_distance(rel.pitch_class, session.pitch_class) <= max_semitones
    return False


def all_keys() -> list[Key]:
    return [Key(pc, mode) for mode in ("minor", "major") for pc in range(12)]


@dataclass
class Assignment:
    """One loop placed in a session key."""

    label: str
    key: Key
    shift: int
    via_relative: bool = False


@dataclass
class Option:
    """A candidate session key and what it covers."""

    key: Key
    covered: list[Assignment] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.covered)

    @property
    def untransposed(self) -> int:
        return sum(1 for a in self.covered if a.shift == 0)


@dataclass
class Plan:
    """The full result: ranked session keys and what each picks up."""

    options: list[Option]
    unreachable: list[tuple[str, Key]]
    total_keyed: int
    total_rows: int
    max_semitones: int
    allow_relative: bool
    runners_up: list[tuple[Key, int]] = field(default_factory=list)

    @property
    def covered_count(self) -> int:
        return sum(o.count for o in self.options)


def _assign(label: str, loop: Key, session: Key,
            max_semitones: int, allow_relative: bool) -> Assignment | None:
    if loop.mode == session.mode:
        if semitone_distance(loop.pitch_class, session.pitch_class) <= max_semitones:
            return Assignment(label, loop,
                              signed_shift(loop.pitch_class, session.pitch_class))
        return None
    if allow_relative:
        rel = loop.relative
        if semitone_distance(rel.pitch_class, session.pitch_class) <= max_semitones:
            return Assignment(label, loop,
                              signed_shift(rel.pitch_class, session.pitch_class),
                              via_relative=True)
    return None


def plan_sessions(
    entries: list[tuple[str, Key]],
    max_keys: int = 3,
    max_semitones: int = MAX_SEMITONES_DEFAULT,
    allow_relative: bool = False,
    total_rows: int | None = None,
) -> Plan:
    """Greedy set cover: pick the key covering most loops, then repeat.

    Greedy is not guaranteed optimal for set cover, but with 24 candidate keys
    and a coverage function this smooth, the first pick is always the true
    maximum and later picks are near it. Exhaustive search over triples is
    2024 combinations and would also be fine—greedy is chosen for readable
    output ordering, not speed.
    """
    if max_keys < 1:
        raise ValueError("max_keys must be at least 1")

    remaining = list(entries)
    options: list[Option] = []
    runners_up: list[tuple[Key, int]] = []

    for round_index in range(max_keys):
        if not remaining:
            break

        scored: list[tuple[int, int, Key, list[Assignment]]] = []
        for session in all_keys():
            hits = [
                a for a in (
                    _assign(label, loop, session, max_semitones, allow_relative)
                    for label, loop in remaining
                ) if a is not None
            ]
            # Rank by coverage, then by how many need no transposition at all.
            scored.append((len(hits), sum(1 for h in hits if h.shift == 0),
                           session, hits))

        scored.sort(key=lambda s: (-s[0], -s[1], s[2].pitch_class, s[2].mode))
        best = scored[0]
        if best[0] == 0:
            break

        if round_index == 0:
            runners_up = [(s[2], s[0]) for s in scored[1:4]]

        options.append(Option(key=best[2], covered=best[3]))
        taken = {id_ for id_ in range(len(remaining))
                 if _assign(remaining[id_][0], remaining[id_][1], best[2],
                            max_semitones, allow_relative) is not None}
        remaining = [e for i, e in enumerate(remaining) if i not in taken]

    return Plan(
        options=options,
        unreachable=remaining,
        total_keyed=len(entries),
        total_rows=total_rows if total_rows is not None else len(entries),
        max_semitones=max_semitones,
        allow_relative=allow_relative,
        runners_up=runners_up,
    )


def key_histogram(entries: list[tuple[str, Key]]) -> list[tuple[Key, int]]:
    counts = Counter(key for _, key in entries)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].pitch_class))


@dataclass
class SweepRow:
    """Coverage at one transposition limit."""

    semitones: int
    covered: int
    total: int
    keys_used: int
    keys_for_full: int | None
    stretched: int  # covered only by exceeding the comfort threshold

    @property
    def share(self) -> float:
        return 100 * self.covered / self.total if self.total else 0.0


def sweep(
    entries: list[tuple[str, Key]],
    max_keys: int = 3,
    allow_relative: bool = False,
    comfort: int = COMFORT_SEMITONES,
    full_cover_ceiling: int = 8,
) -> list[SweepRow]:
    """Coverage at every transposition limit from 0 to a tritone.

    Two numbers per row that answer different questions. `covered` is how much
    of the library `max_keys` session keys reach at that limit. `keys_for_full`
    is how many session keys it would take to reach everything—which is often
    the more useful figure, because relaxing the limit by one semitone can drop
    you from three session keys to two.

    `stretched` counts loops that only come along because the limit exceeds the
    comfort threshold. Those are the ones to audition before trusting them.
    """
    rows: list[SweepRow] = []
    total = len(entries)

    for limit in range(MAX_POSSIBLE_SEMITONES + 1):
        plan = plan_sessions(entries, max_keys=max_keys,
                             max_semitones=limit, allow_relative=allow_relative)
        stretched = sum(
            1 for option in plan.options for a in option.covered
            if abs(a.shift) > comfort
        )

        keys_for_full = None
        for candidate in range(1, full_cover_ceiling + 1):
            trial = plan_sessions(entries, max_keys=candidate,
                                  max_semitones=limit,
                                  allow_relative=allow_relative)
            if not trial.unreachable:
                keys_for_full = candidate
                break

        rows.append(SweepRow(
            semitones=limit,
            covered=plan.covered_count,
            total=total,
            keys_used=len(plan.options),
            keys_for_full=keys_for_full,
            stretched=stretched,
        ))

    return rows
