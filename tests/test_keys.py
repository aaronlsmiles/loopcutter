"""Key parsing and session-cover tests.

The Camelot mappings are the part most likely to be quietly wrong, so they are
pinned against known anchors rather than trusted.
"""

import pytest

from loopcutter.keys import (
    Key,
    KeyParseError,
    parse_key,
    plan_sessions,
    semitone_distance,
    signed_shift,
)


def test_camelot_anchors():
    """8A is A minor and 8B is C major. Everything else follows from these."""
    assert parse_key("8A") == Key(9, "minor")
    assert parse_key("8B") == Key(0, "major")
    assert parse_key("1A") == Key(8, "minor")   # Ab minor
    assert parse_key("1B") == Key(11, "major")  # B major
    assert parse_key("12A") == Key(1, "minor")  # Db minor
    assert parse_key("7B") == Key(5, "major")   # F major


def test_b_series_is_the_relative_major_of_the_a_series():
    for number in range(1, 13):
        minor = parse_key(f"{number}A")
        major = parse_key(f"{number}B")
        assert minor.relative == major
        assert major.relative == minor


def test_named_forms():
    assert parse_key("Am") == Key(9, "minor")
    assert parse_key("A minor") == Key(9, "minor")
    assert parse_key("Amin") == Key(9, "minor")
    assert parse_key("C") == Key(0, "major")
    assert parse_key("C major") == Key(0, "major")
    assert parse_key("Cmaj") == Key(0, "major")
    assert parse_key("F#m") == Key(6, "minor")
    assert parse_key("Gbm") == Key(6, "minor")
    assert parse_key("Bb") == Key(10, "major")
    assert parse_key(" 8a ") == Key(9, "minor")


def test_unicode_accidentals():
    assert parse_key("F♯m") == parse_key("F#m")
    assert parse_key("B♭") == parse_key("Bb")


def test_rejects_nonsense():
    for bad in ("", "  ", "13A", "0A", "H minor", "8C", "banana", None):
        with pytest.raises(KeyParseError):
            parse_key(bad)


def test_distance_is_circular_and_capped_at_six():
    assert semitone_distance(0, 0) == 0
    assert semitone_distance(0, 11) == 1
    assert semitone_distance(0, 6) == 6
    assert semitone_distance(11, 1) == 2
    assert all(semitone_distance(a, b) <= 6
               for a in range(12) for b in range(12))


def test_signed_shift_takes_the_short_way():
    assert signed_shift(0, 2) == 2
    assert signed_shift(0, 11) == -1
    assert signed_shift(11, 0) == 1
    # A tritone resolves downward: gentler on loops with bass content.
    assert signed_shift(0, 6) == -6


def test_single_key_covers_a_tight_cluster():
    entries = [(f"l{i}", parse_key(k)) for i, k in
               enumerate(["Am", "Gm", "Bm", "Cm", "Am"])]
    plan = plan_sessions(entries, max_keys=3, max_semitones=3)
    assert plan.options[0].count == 5
    assert not plan.unreachable


def test_distant_keys_need_a_second_session_key():
    entries = [("a", parse_key("Am")), ("b", parse_key("Bm")),
               ("c", parse_key("Ebm")), ("d", parse_key("Em"))]
    plan = plan_sessions(entries, max_keys=3, max_semitones=1)
    assert len(plan.options) >= 2
    assert plan.covered_count + len(plan.unreachable) == 4


def test_modes_do_not_mix_unless_relative_is_allowed():
    entries = [("min", parse_key("Am")), ("maj", parse_key("Cmaj"))]

    strict = plan_sessions(entries, max_keys=1, max_semitones=3)
    assert strict.options[0].count == 1

    loose = plan_sessions(entries, max_keys=1, max_semitones=3,
                          allow_relative=True)
    assert loose.options[0].count == 2


def test_every_loop_is_accounted_for():
    """Covered plus unreachable must always equal the input. No silent drops."""
    import random
    random.seed(7)
    keys = [f"{random.randint(1, 12)}{random.choice('AB')}" for _ in range(60)]
    entries = [(f"l{i}", parse_key(k)) for i, k in enumerate(keys)]
    for limit in (0, 1, 2, 3, 6):
        plan = plan_sessions(entries, max_keys=3, max_semitones=limit)
        assert plan.covered_count + len(plan.unreachable) == 60


def test_shifts_never_exceed_the_limit():
    import random
    random.seed(11)
    entries = [(f"l{i}", parse_key(f"{random.randint(1, 12)}A"))
               for i in range(40)]
    plan = plan_sessions(entries, max_keys=3, max_semitones=2)
    for option in plan.options:
        for a in option.covered:
            assert abs(a.shift) <= 2


def test_first_pick_is_the_true_maximum():
    """Greedy's first choice should beat every alternative, exhaustively."""
    from loopcutter.keys import all_keys, _assign
    import random
    random.seed(3)
    entries = [(f"l{i}", parse_key(f"{random.randint(1, 12)}{random.choice('AB')}"))
               for i in range(50)]
    plan = plan_sessions(entries, max_keys=1, max_semitones=3)
    best = plan.options[0].count
    for session in all_keys():
        hits = sum(1 for lbl, k in entries
                   if _assign(lbl, k, session, 3, False) is not None)
        assert hits <= best


def test_max_keys_must_be_positive():
    with pytest.raises(ValueError):
        plan_sessions([("a", parse_key("Am"))], max_keys=0)


def test_sweep_is_monotonic_and_complete():
    """Coverage can only rise as the limit relaxes, and 0-6 are all present."""
    from loopcutter.keys import sweep
    import random
    random.seed(5)
    entries = [(f"l{i}", parse_key(f"{random.randint(1, 12)}{random.choice('AB')}"))
               for i in range(50)]
    rows = sweep(entries, max_keys=3)

    assert [r.semitones for r in rows] == list(range(7))
    covered = [r.covered for r in rows]
    assert covered == sorted(covered), "relaxing the limit cannot lose loops"
    assert rows[-1].covered == 50, "a tritone reaches everything in one mode span"


def test_sweep_keys_for_full_never_increases():
    from loopcutter.keys import sweep
    import random
    random.seed(9)
    entries = [(f"l{i}", parse_key(f"{random.randint(1, 12)}A"))
               for i in range(40)]
    rows = sweep(entries, max_keys=3)
    needed = [r.keys_for_full for r in rows if r.keys_for_full is not None]
    assert needed == sorted(needed, reverse=True), (
        "a looser limit cannot require more session keys")


def test_stretched_count_matches_the_comfort_threshold():
    from loopcutter.keys import sweep
    entries = [("a", parse_key("Am")), ("b", parse_key("Ebm"))]
    rows = {r.semitones: r for r in sweep(entries, max_keys=1, comfort=3)}
    # A minor to Eb minor is a tritone, so it only arrives at +/-6.
    assert rows[3].stretched == 0
    assert rows[6].stretched >= 1
