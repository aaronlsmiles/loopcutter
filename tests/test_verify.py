import numpy as np
import pytest
import soundfile as sf

from loopcutter.cut import cut_loop
from loopcutter.manifest import load_manifest
from loopcutter.verify import verify_cut

ON_BEAT = "15.0"            # beat 32 at 128 BPM: 32 * 60/128 = 15.0 s exactly


def _spec(tmp_path, source, bars=4, bpm=128, start=ON_BEAT):
    path = tmp_path / "m.csv"
    path.write_text(f"source,label,bars,bpm,start\n{source},A1,{bars},{bpm},{start}\n")
    return load_manifest(path)[0]


def _clicks(path, sr=44100, bpm=128.0, seconds=30.0, peak=0.6):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    audio = np.zeros(n)
    for beat in range(int(seconds * bpm / 60)):
        s = int(round(beat * sr * 60 / bpm)); e = min(n, s + 400)
        audio[s:e] += np.linspace(1, 0, e - s) * np.sin(2 * np.pi * 1000 * t[: e - s])
    audio *= peak / np.max(np.abs(audio))
    sf.write(str(path), np.column_stack([audio, audio]).astype("float32"), sr, subtype="FLOAT")
    return path


def _failed(report):
    return [c.name for c in report.failures]


def test_a_good_loop_passes_every_check(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"]), tmp_path / "o", fmt="wav")
    for report in (verify_cut(result, check_bpm=True, reference_bpm=128.004),
                   verify_cut(result, check_bpm=True)):
        assert report.ok, [(c.name, c.detail) for c in report.failures]
        assert not report.warnings, [(c.name, c.detail) for c in report.warnings]
        assert not [c.name for c in report.checks if "not judged" in c.detail]


def test_bpm_check_measures_end_error_against_the_analysis(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], bpm=134), tmp_path / "o", fmt="wav")
    assert "bpm_match" in _failed(verify_cut(result, check_bpm=True, reference_bpm=136.0))


def test_bpm_check_without_analysis_measures_the_audio(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], bpm=124), tmp_path / "o", fmt="wav")
    report = verify_cut(result, check_bpm=True)
    assert "bpm_match" in _failed(report)
    assert "measured 128.0" in next(c.detail for c in report.checks if c.name == "bpm_match")


def test_beat_alignment_catches_a_late_start(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], start="15.04"), tmp_path / "o", fmt="wav")
    assert "beat_alignment" in _failed(verify_cut(result))


def test_beat_alignment_catches_an_early_typed_start(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], start="14.85"), tmp_path / "o", fmt="wav")
    assert "beat_alignment" in _failed(verify_cut(result))


def test_beat_alignment_catches_a_wrong_tempo(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], bpm=134), tmp_path / "o", fmt="wav")
    assert "beat_alignment" in _failed(verify_cut(result))


def test_beat_alignment_catches_a_small_tempo_error(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], bpm=128.3), tmp_path / "o", fmt="wav")
    assert "beat_alignment" in _failed(verify_cut(result))     # 4 bars end 17.5 ms early


def test_a_start_10ms_early_warns_but_passes(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], start="14.99"), tmp_path / "o", fmt="wav")
    report = verify_cut(result)
    assert report.ok and [c.name for c in report.warnings] == ["beat_alignment"]


def test_full_scale_master_passes_the_peak_check(tmp_path):
    result = cut_loop(_spec(tmp_path, _clicks(tmp_path / "loud.wav", peak=1.0)), tmp_path / "o", fmt="wav")
    assert result.source_peak == pytest.approx(1.0, abs=1e-6)
    assert "peak_headroom" not in _failed(verify_cut(result))


def test_trim_lowers_the_peak_and_a_boost_that_clips_fails(tmp_path):
    spec = _spec(tmp_path, _clicks(tmp_path / "loud.wav", peak=1.0))
    trimmed = cut_loop(spec, tmp_path / "a", fmt="wav", trim_db=-6.0)
    assert trimmed.peak == pytest.approx(10 ** (-6 / 20), rel=0.01)
    assert "peak_headroom" not in _failed(verify_cut(trimmed))
    boosted = cut_loop(spec, tmp_path / "b", fmt="wav", trim_db=6.0)
    assert "peak_headroom" in _failed(verify_cut(boosted))
    assert np.max(np.abs(sf.read(str(boosted.output))[0])) <= 1.0    # clipped, never wrapped


def test_zero_crossing_search_never_moves_the_start_later(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"], start="15.003"), tmp_path / "o", fmt="wav")
    assert result.snap_offset_samples <= 0


def _names(report):
    return [c.name for c in report.failures]


def test_sample_count_and_sample_rate_catch_a_damaged_file(click_track, tmp_path):
    result = cut_loop(_spec(tmp_path, click_track["path"]), tmp_path / "o", fmt="wav")
    audio, sr = sf.read(str(result.output), dtype="float32")
    sf.write(str(result.output), audio[:-10], sr)                           # truncated
    assert "sample_count" in _names(verify_cut(result))
    sf.write(str(result.output), audio, 48000)                              # wrong rate
    assert "sample_rate" in _names(verify_cut(result))


def test_not_silent_and_dc_offset_can_fail(tmp_path):
    sr = 44100
    silent = tmp_path / "silent.wav"
    sf.write(str(silent), np.zeros((sr * 30, 2), dtype="float32"), sr)
    assert "not_silent" in _names(verify_cut(cut_loop(_spec(tmp_path, silent), tmp_path / "a", fmt="wav")))
    offset = tmp_path / "dc.wav"
    audio, _ = sf.read(str(_clicks(tmp_path / "c.wav", peak=0.5)), dtype="float32")
    sf.write(str(offset), audio + 0.05, sr, subtype="FLOAT")
    assert "dc_offset" in _names(verify_cut(cut_loop(_spec(tmp_path, offset), tmp_path / "b", fmt="wav")))
