"""The arithmetic tests. If these fail, nothing else matters."""

import math

import pytest

from loopcutter.timing import (
    bar_start_seconds,
    loop_length_samples,
    parse_position,
    resolve_window,
    samples_per_beat,
)


def test_parse_position_forms():
    assert parse_position("12.5") == pytest.approx(12.5)
    assert parse_position("1:23.456") == pytest.approx(83.456)
    assert parse_position("0:00") == 0.0
    assert parse_position("1:02:03") == pytest.approx(3723.0)
    assert parse_position(7) == 7.0


def test_parse_position_rejects_nonsense():
    for bad in ("", "abc", "1:99", "1:60.0"):
        with pytest.raises(ValueError):
            parse_position(bad)


def test_four_bars_at_120_is_eight_seconds():
    # 120 BPM -> 2 beats/sec -> 16 beats -> 8.0 s -> 352800 samples at 44.1k
    assert loop_length_samples(4, 120, 44100) == 352800


def test_length_rounds_once():
    # 127 BPM is deliberately awkward: 4 bars is not a whole number of samples.
    exact = 16 * 44100 * 60 / 127
    assert loop_length_samples(4, 127, 44100) == round(exact)


def test_length_is_not_accumulated_per_beat():
    """Rounding per beat then multiplying drifts. Prove we do not do that."""
    naive = round(samples_per_beat(127, 44100)) * 16
    correct = loop_length_samples(4, 127, 44100)
    assert naive != correct
    assert abs(naive - correct) > 2


def test_lengths_are_additive():
    """Two 4-bar loops must equal one 8-bar loop, within a sample."""
    four = loop_length_samples(4, 131.7, 48000)
    eight = loop_length_samples(8, 131.7, 48000)
    assert abs(four * 2 - eight) <= 1


def test_bar_start_is_one_based():
    assert bar_start_seconds(10.0, 1, 120) == pytest.approx(10.0)
    assert bar_start_seconds(10.0, 2, 120) == pytest.approx(12.0)
    assert bar_start_seconds(10.0, 5, 120) == pytest.approx(18.0)
    with pytest.raises(ValueError):
        bar_start_seconds(10.0, 0, 120)


def test_three_four_time():
    assert loop_length_samples(4, 120, 44100, beats_per_bar=3) == 264600


def test_resolve_window_end():
    window = resolve_window(10.0, 4, 120, 44100)
    assert window.start_sample == 441000
    assert window.length_samples == 352800
    assert window.end_sample == 793800


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        loop_length_samples(0, 120, 44100)
    with pytest.raises(ValueError):
        loop_length_samples(4, 0, 44100)
    with pytest.raises(ValueError):
        resolve_window(-1, 4, 120, 44100)


from loopcutter.timing import rates_that_tile, tiling_error_ms


def test_rates_that_tile_depend_on_tempo():
    assert rates_that_tile(0.5, 128) == [48000]
    assert rates_that_tile(0.5, 135) == [44100]
    assert rates_that_tile(0.5, 125) == [44100, 48000]


def test_tiling_error_in_ms():
    assert tiling_error_ms(4, 1, 128, 44100) == pytest.approx(2 / 44100 * 1000)
