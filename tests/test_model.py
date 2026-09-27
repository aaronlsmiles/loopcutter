import pytest

from loopcutter.model import Grid, Marker, SnapResult


def test_grid_arithmetic():
    grid = Grid(period=0.5, phase=0.1, bar_phase=1)
    assert grid.bpm == 120.0
    assert grid.nearest_beat(1.33) == pytest.approx(1.1)
    assert grid.nearest_bar(1.33) == pytest.approx(0.6)       # bars on beats 1, 5, 9 -> 0.6, 2.6
    assert grid.is_bar_start(5) and not grid.is_bar_start(6)


def test_shifted_wraps_and_keeps_the_bar():
    grid = Grid(period=0.5, phase=0.49, bar_phase=0)
    moved = grid.shifted(0.02)
    assert moved.phase == pytest.approx(0.01) and moved.phase_offset == pytest.approx(0.02)
    assert moved.nearest_bar(0.51) == pytest.approx(0.51)
    back = moved.shifted(-0.02)
    assert back.phase == pytest.approx(0.49) and back.nearest_bar(0.49) == pytest.approx(0.49)


def test_marker_kind_and_snap_shift(tmp_path):
    assert Marker(tmp_path, 1.0, None, "", "serato").kind == "cue"
    assert Marker(tmp_path, 1.0, 3.0, "", "serato").kind == "loop"
    assert SnapResult(1.041, 1.0, 2, True).shift_ms == pytest.approx(-41.0)
