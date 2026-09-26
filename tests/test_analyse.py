import numpy as np
import pytest

from loopcutter.analyse import analyse_record, flags_for, scan_tracks
from loopcutter.grid import load_beats
from loopcutter.keydetect import KeyResult
from loopcutter.keys import parse_key
from loopcutter.model import Grid, TrackRecord
from loopcutter.trackdb import load_tracks, save_tracks

P128 = 60 / 128


def _detector(mono, sr):
    beats = np.arange(64) * P128
    return beats, beats[::4]


def _record(path, **kw):
    base = dict(track_id="clicks", source=str(path), master=str(path), sample_rate=0,
                duration=0.0, bpm=0.0, phase=0.0, bar_phase=0)
    base.update(kw)
    return TrackRecord(**base)


def test_analyse_record_fills_grid_key_and_the_beat_cache(click_track, tmp_path):
    record = analyse_record(_record(click_track["path"]), detector=_detector,
                            keyfinder=lambda p: parse_key("8A"), beats_dir=tmp_path)
    assert record.bpm == 128.0 and record.bpm_fitted == pytest.approx(128.0, abs=0.01)
    assert (record.sample_rate, round(record.duration)) == (44100, 30)
    assert (record.key, record.key_source, record.flags) == ("8A", "keyfinder", "")
    assert load_beats(tmp_path / "clicks.npz")[0].size == 64


def test_your_overrides_survive_a_rescan(click_track):
    record = analyse_record(_record(click_track["path"], override_phase_ms=12.5),
                            detector=_detector, keyfinder=lambda p: None)
    assert record.override_phase_ms == 12.5


def test_app_grid_comparison_flags_a_half_beat(click_track):
    record = analyse_record(_record(click_track["path"]), detector=_detector, keyfinder=lambda p: None,
                            app_beats=np.arange(60) * P128 + P128 / 2, app="serato")
    assert "half-beat" in record.flags and record.app == "serato"


def test_flags_for_each_problem():
    shaky = Grid(period=P128, phase=0.0, inlier_ratio=0.5, bar_agreement=0.6, phase_agreement=0.2)
    disagree = KeyResult(parse_key("8A"), "tag", parse_key("9A"))
    app = {"app_bpm": 127.0, "app_offset_ms": 30.0, "half_beat": False}
    assert flags_for(shaky, disagree, app, lossless=True) == ["grid-fit", "bar-phase", "phase", "key", "app-bpm", "app-offset"]
    mp3 = {"app_bpm": 128.0, "app_offset_ms": 41.0, "half_beat": False}
    assert "app-offset" not in flags_for(shaky, disagree, mp3, lossless=False)


def test_scan_tracks_updates_the_csv_and_skips_done_rows(click_track, tmp_path):
    csv_path = tmp_path / "tracks.csv"
    save_tracks(csv_path, {"clicks": _record(click_track["path"])})
    report = scan_tracks(csv_path, detector=_detector, keyfinder=lambda p: None)
    assert report.analysed == ["clicks"] and load_tracks(csv_path)["clicks"].bpm == 128.0
    assert scan_tracks(csv_path, detector=_detector, keyfinder=lambda p: None).skipped == ["clicks"]
    assert "1 analysed" in report.text()
