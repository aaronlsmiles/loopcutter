import numpy as np
import pytest
import soundfile as sf

from loopcutter.grid import (
    GridError, analyse_grid, attack_phase, bar_phase, compare_grids, fit_grid, load_beats,
    local_grid, nominal_bpm, save_beats,
)
from loopcutter.model import Grid

P128 = 60 / 128


def _quantised(times):
    """beat_this reports beats on a 50 fps frame grid."""
    return np.round(np.asarray(times) / 0.02) * 0.02


def test_fit_survives_50fps_quantisation_and_a_breakdown():
    beats = _quantised(1.0 + np.arange(400) * P128)
    beats = np.delete(beats, np.arange(150, 182))           # 32 beats with no detections
    beats = np.append(beats, [beats[10] + 0.21])            # one stray detection
    grid = fit_grid(np.sort(beats))
    assert grid.bpm == pytest.approx(128.0, abs=0.01)
    assert grid.inlier_ratio > 0.95
    assert abs((1.0 - grid.phase) / grid.period - round((1.0 - grid.phase) / grid.period)) < 0.02


def test_fit_flags_a_tempo_change_with_low_inliers():
    first = np.arange(150) * 60 / 120
    second = first[-1] + np.arange(1, 151) * 60 / 128
    assert fit_grid(np.concatenate([first, second])).inlier_ratio < 0.9


def test_local_grid_fits_each_section_of_a_tempo_change():
    first = np.arange(150) * 60 / 120
    second = first[-1] + np.arange(1, 151) * 60 / 128
    beats = np.concatenate([first, second])
    downbeats = beats[::4]
    assert local_grid(beats, downbeats, t=30.0).bpm == pytest.approx(120.0, abs=0.01)
    assert local_grid(beats, downbeats, t=first[-1] + 40.0).bpm == pytest.approx(128.0, abs=0.01)


def test_fit_refuses_too_few_beats():
    with pytest.raises(GridError):
        fit_grid(np.arange(5) * P128)


def test_bar_phase_finds_the_one():
    assert bar_phase(Grid(period=P128, phase=0.0), np.arange(1, 200, 4) * P128) == (1, 1.0)


def test_nominal_bpm_snaps_near_integers_only():
    assert nominal_bpm(124.9994) == 125.0
    assert nominal_bpm(127.43) == pytest.approx(127.43)


def test_compare_grids_reports_offset_and_half_beat():
    grid = Grid(period=P128, phase=0.0)
    report = compare_grids(grid, np.arange(100) * P128 + 0.041)
    assert report["app_offset_ms"] == pytest.approx(41.0, abs=0.5)
    assert report["app_bpm"] == pytest.approx(128.0, abs=0.01) and not report["half_beat"]
    assert compare_grids(grid, np.arange(100) * P128 + P128 / 2)["half_beat"]


def test_attack_phase_tracks_a_known_shift(click_track):
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    base = Grid(period=P128, phase=0.0)
    off0, agreement = attack_phase(mono, sr, base)
    off10, _ = attack_phase(mono, sr, base.shifted(0.010))
    assert agreement > 0.9
    assert off10 - off0 == pytest.approx(-0.010, abs=0.0015)


def test_analyse_grid_with_an_injected_detector(click_track):
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    beats = _quantised(np.arange(64) * P128)
    grid, got_beats, _ = analyse_grid(audio.mean(axis=1), sr, detector=lambda m, s: (beats, beats[::4]))
    assert grid.bpm == pytest.approx(128.0, abs=0.02)
    assert grid.is_bar_start(grid.beat_index(0.0)) and grid.bar_agreement == 1.0
    assert grid.phase_agreement > 0.8
    assert np.array_equal(got_beats, beats)


def test_beat_cache_roundtrip(tmp_path):
    beats, downbeats = np.arange(10) * 0.5, np.arange(0, 10, 4) * 0.5
    save_beats(tmp_path / "t.npz", beats, downbeats)
    b, d = load_beats(tmp_path / "t.npz")
    assert np.array_equal(b, beats) and np.array_equal(d, downbeats)


@pytest.mark.slow
def test_real_detector_on_clicks(click_track):
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    grid, _, _ = analyse_grid(audio.mean(axis=1), sr)       # downloads beat_this weights once
    assert grid.bpm == pytest.approx(128.0, abs=0.05)


def test_compare_grids_reads_a_grid_stored_in_whole_milliseconds():
    grid = Grid(period=P128, phase=0.0)
    app = np.round((np.arange(400) * P128 + 0.041) * 1000) / 1000     # rekordbox keeps beat times in ms
    report = compare_grids(grid, app)
    assert report["app_bpm"] == pytest.approx(128.0, abs=0.005)
    assert report["app_offset_ms"] == pytest.approx(41.0, abs=0.5)
    gappy = np.delete(app, np.arange(100, 140))
    assert compare_grids(grid, gappy)["app_bpm"] == pytest.approx(128.0, abs=0.005)


P125 = 60 / 125


def test_fit_survives_a_section_the_detector_tracks_in_triplets():
    beats = 1.0 + np.arange(1000) * P125
    triplets = beats[400] + np.arange(60) * (2 * P125 / 3)      # 40 beats followed at two thirds of a beat
    grid = fit_grid(_quantised(np.sort(np.concatenate([beats[:400], triplets, beats[440:]]))))
    assert grid.bpm == pytest.approx(125.0, abs=0.01)
    assert grid.inlier_ratio > 0.9


def test_fit_survives_swing_and_dropouts():
    rng = np.random.default_rng(0)
    beats = 1.0 + np.arange(700) * 60 / 140
    kept = beats[rng.random(700) > 0.15]
    swung = beats[rng.random(700) < 0.3]
    detections = np.concatenate([kept, swung + rng.uniform(0.08, 0.2, swung.size)])
    grid = fit_grid(_quantised(np.sort(detections)))
    assert grid.bpm == pytest.approx(140.0, abs=0.01)
    assert grid.inlier_ratio > 0.7
