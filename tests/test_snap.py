import csv

import numpy as np
import pytest

from loopcutter.model import Marker, TrackRecord
from loopcutter.snap import (
    MANIFEST_FIELDS, SnapError, bars_from_length, markers_to_rows, snap_time, track_grid,
    write_manifest,
)

P125 = 60 / 125


def _rec(tmp_path, **kw):
    base = dict(track_id="Artist - Song", source=str(tmp_path / "Artist - Song.mp3"),
                master=str(tmp_path / "Artist - Song.aiff"), sample_rate=48000, duration=300.0,
                bpm=125.0, phase=0.1, bar_phase=0, key="8A", inlier_ratio=1.0)
    base.update(kw)
    return TrackRecord(**base)


def test_snap_moves_a_near_miss_onto_the_beat(tmp_path):
    result = snap_time(0.1 + 32 * P125 + 0.030, track_grid(_rec(tmp_path)))
    assert result.snapped == pytest.approx(0.1 + 32 * P125) and result.on_bar
    assert result.shift_ms == pytest.approx(-30.0, abs=0.01)


def test_refusal_names_the_nearby_beats_and_bar(tmp_path):
    with pytest.raises(SnapError, match="nearest bar"):
        snap_time(0.1 + 10.3 * P125, track_grid(_rec(tmp_path)))


def test_bar_mode_and_off_mode(tmp_path):
    grid = track_grid(_rec(tmp_path))
    assert snap_time(0.1 + 9 * P125, grid, mode="bar", max_shift_ms=600).snapped == pytest.approx(0.1 + 8 * P125)
    assert snap_time(3.21, grid, mode="off").snapped == 3.21


def test_your_overrides_move_the_grid(tmp_path):
    grid = track_grid(_rec(tmp_path, override_phase_ms=P125 * 500))     # half a beat later
    assert grid.nearest_beat(0.1 + P125 / 2) == pytest.approx(0.1 + P125 / 2)


def test_the_grid_uses_the_fitted_tempo(tmp_path):
    assert track_grid(_rec(tmp_path, bpm=125.0, bpm_fitted=124.995)).bpm == pytest.approx(124.995)


def test_bars_from_length_with_tolerance(tmp_path):
    grid = track_grid(_rec(tmp_path))
    assert bars_from_length(16 * P125 + 0.012, grid) == 4
    assert bars_from_length(2 * P125 - 0.01, grid) == 0.5
    assert bars_from_length(14 * P125, grid) is None                  # 3.5 bars: not a standard length


def test_app_offset_is_removed_before_snapping(tmp_path):
    rec = _rec(tmp_path, app="serato", app_offset_ms=41.0)
    beat = 0.1 + 64 * P125
    late = Marker(tmp_path / "Artist - Song.mp3", beat + 0.041 + 0.030, None, "A1", "serato")
    result = markers_to_rows([late], {rec.track_id: rec})
    assert float(result.rows[0]["start"]) == pytest.approx(beat, abs=1e-6)   # 71 ms late -> 30 ms after correction
    flagged = _rec(tmp_path, app="serato", app_offset_ms=41.0, flags="half-beat")
    assert markers_to_rows([late], {flagged.track_id: flagged}).problems      # no correction -> refused


def test_markers_become_resolved_rows(tmp_path):
    rec = _rec(tmp_path)
    start = 0.1 + 64 * P125
    markers = [
        Marker(tmp_path / "Artist - Song.mp3", start + 0.02, start + 0.02 + 16 * P125, "A1 bass 2,1", "serato"),
        Marker(tmp_path / "Artist - Song.mp3", start + 0.24, None, "", "serato"),          # half a beat out
        Marker(tmp_path / "Artist - Song.mp3", start, start + 14 * P125, "odd", "serato"),  # 3.5 bars
        Marker(tmp_path / "Unknown.mp3", 1.0, None, "", "serato"),
    ]
    result = markers_to_rows(markers, {rec.track_id: rec})
    row = result.rows[0]
    assert (row["label"], row["bars"], row["stem"], row["variations"], row["snap"]) == ("A1", "4", "bass", "2,1", "")
    assert row["source"] == rec.master and row["artist"] == "Artist" and row["track"] == "Song"
    assert float(row["start"]) == pytest.approx(start, abs=1e-6) and row["key"] == "8A" and row["bpm"] == "125"
    assert len(result.rows) == 1 and len(result.problems) == 3
    assert result.moves_ms == [pytest.approx(-20.0, abs=0.01)]


def test_local_grid_is_used_for_a_track_whose_tempo_moves(tmp_path):
    first = np.arange(150) * 60 / 120
    second = first[-1] + np.arange(1, 151) * 60 / 128
    beats = np.concatenate([first, second])
    rec = _rec(tmp_path, bpm=124.0, phase=0.0, inlier_ratio=0.5, flags="grid-fit")
    target = second[40]
    marker = Marker(tmp_path / "Artist - Song.mp3", target + 0.02, None, "", "serato")
    result = markers_to_rows([marker], {rec.track_id: rec}, beats_for=lambda r: (beats, beats[::4]))
    assert float(result.rows[0]["start"]) == pytest.approx(target, abs=0.002)
    assert result.rows[0]["bpm"] == "128"


def test_an_unsure_grid_keeps_rekordbox_marks_on_the_master_and_refuses_the_rest(tmp_path):
    rec = _rec(tmp_path, flags="phase")
    odd = 0.1 + 64 * P125 + 0.017
    kept = Marker(tmp_path / "Artist - Song.aiff", odd, None, "A1", "rekordbox")
    refused = Marker(tmp_path / "Artist - Song.mp3", odd, None, "A2", "serato")
    result = markers_to_rows([kept, refused], {rec.track_id: rec})
    assert [float(r["start"]) for r in result.rows] == [pytest.approx(odd)]
    assert len(result.problems) == 1 and "confident" in result.problems[0]


def test_write_manifest_refuses_to_overwrite(tmp_path):
    out = tmp_path / "m.csv"
    write_manifest([{"source": "a", "label": "A1", "bars": "4", "bpm": "125", "start": "1.0"}], out)
    with open(out) as fh:
        assert next(csv.reader(fh)) == MANIFEST_FIELDS
    with pytest.raises(FileExistsError):
        write_manifest([], out)


def _tempo_change():
    first = np.arange(150) * 60 / 120
    second = first[-1] + np.arange(1, 151) * 60 / 128
    beats = np.concatenate([first, second])
    return beats, beats[::4]


def test_a_local_grid_keeps_the_attack_correction_and_your_override(tmp_path):
    beats, downbeats = _tempo_change()
    target = beats[190]                                        # raw detector beat, 15 ms after the attack
    marker = Marker(tmp_path / "Artist - Song.mp3", target + 0.02, None, "", "serato")
    corrected = _rec(tmp_path, bpm=124.0, phase=0.0, inlier_ratio=0.5, flags="grid-fit", phase_offset_ms=-15.0)
    result = markers_to_rows([marker], {corrected.track_id: corrected}, beats_for=lambda r: (beats, downbeats))
    assert float(result.rows[0]["start"]) == pytest.approx(target - 0.015, abs=0.001)
    overridden = _rec(tmp_path, bpm=124.0, phase=0.0, inlier_ratio=0.5, flags="grid-fit", override_phase_ms=20.0)
    result = markers_to_rows([marker], {overridden.track_id: overridden}, beats_for=lambda r: (beats, downbeats))
    assert float(result.rows[0]["start"]) == pytest.approx(target + 0.020, abs=0.001)


def test_an_unanalysed_track_or_a_missing_beat_cache_is_reported_not_crashed(tmp_path):
    marker = Marker(tmp_path / "Artist - Song.mp3", 10.0, None, "", "serato")
    unscanned = _rec(tmp_path, bpm=0.0)
    assert "scan" in markers_to_rows([marker], {unscanned.track_id: unscanned}).problems[0]
    shaky = _rec(tmp_path, inlier_ratio=0.5, flags="grid-fit")
    result = markers_to_rows([marker], {shaky.track_id: shaky}, beats_for=lambda r: None)
    assert not result.rows and "beat cache" in result.problems[0]


def test_the_app_offset_only_applies_to_marks_on_the_file_it_was_measured_on(tmp_path):
    rec = _rec(tmp_path, app="serato", app_offset_ms=41.0)
    beat = 0.1 + 64 * P125
    on_master = Marker(tmp_path / "Artist - Song.aiff", beat + 0.010, None, "A1", "serato")
    result = markers_to_rows([on_master], {rec.track_id: rec}, max_shift_ms=25)   # corrected, it would be 31 ms out
    assert float(result.rows[0]["start"]) == pytest.approx(beat, abs=1e-6)
    assert result.moves_ms == [pytest.approx(-10.0, abs=0.01)]


def test_a_zero_phase_override_confirms_a_phase_flagged_grid(tmp_path):
    rec = _rec(tmp_path, flags="phase", override_phase_ms=0.0)
    beat = 0.1 + 64 * P125
    marker = Marker(tmp_path / "Artist - Song.mp3", beat + 0.012, None, "A1", "serato")
    result = markers_to_rows([marker], {rec.track_id: rec})
    assert float(result.rows[0]["start"]) == pytest.approx(beat, abs=1e-6)


def test_a_start_is_never_before_the_file(tmp_path):
    grid = track_grid(_rec(tmp_path, phase=P125 - 0.0012))           # beat 0 sits 1.2 ms before the file
    assert snap_time(0.03, grid).snapped == 0.0
    early = track_grid(_rec(tmp_path, phase=P125 - 0.030))
    with pytest.raises(SnapError, match="before the start"):
        snap_time(0.01, early)


def test_a_track_flagged_for_a_break_uses_its_whole_grid_where_that_still_fits(tmp_path):
    period = 60 / 124.37
    raw = np.round((0.3 + np.arange(400) * period) / 0.02) * 0.02              # detector beats, 50 fps
    junk = np.sort(np.random.default_rng(2).uniform(raw[100], raw[160], 60))   # a break the detector wanders in
    beats = np.sort(np.concatenate([raw[:100], junk, raw[160:]]))
    rec = _rec(tmp_path, bpm=124.37, bpm_fitted=124.37, phase=0.3 - 0.015, inlier_ratio=0.8,
               flags="grid-fit", phase_offset_ms=-15.0)
    marker = Marker(tmp_path / "Artist - Song.mp3", 0.3 + 300 * period + 0.02, None, "", "serato")
    result = markers_to_rows([marker], {rec.track_id: rec}, beats_for=lambda r: (beats, beats[::4]))
    assert result.rows[0]["bpm"] == "124.37"
    assert float(result.rows[0]["start"]) == pytest.approx(0.3 - 0.015 + 300 * period, abs=1e-6)


def test_junk_around_a_mark_does_not_trade_the_whole_grid_for_a_worse_local_one(tmp_path):
    period = 60 / 124.37
    raw = np.round((0.3 + np.arange(400) * period) / 0.02) * 0.02
    junk = np.sort(np.random.default_rng(3).uniform(raw[285], raw[315], 20))    # a busy fill right at the mark
    beats = np.sort(np.concatenate([raw, junk]))
    rec = _rec(tmp_path, bpm=124.37, bpm_fitted=124.37, phase=0.3 - 0.015, inlier_ratio=0.8,
               flags="grid-fit", phase_offset_ms=-15.0)
    marker = Marker(tmp_path / "Artist - Song.mp3", 0.3 + 300 * period + 0.02, None, "", "serato")
    result = markers_to_rows([marker], {rec.track_id: rec}, beats_for=lambda r: (beats, beats[::4]))
    assert result.rows[0]["bpm"] == "124.37"
    assert float(result.rows[0]["start"]) == pytest.approx(0.3 - 0.015 + 300 * period, abs=1e-6)
