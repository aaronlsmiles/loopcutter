import numpy as np
import pytest
import soundfile as sf

from loopcutter.onsets import attacks_near, detect_onsets

P128 = 60 / 128


def test_onsets_land_on_the_clicks_in_any_chunk_size(click_track):
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    whole, strength = detect_onsets(mono, sr)
    chunked, _ = detect_onsets(mono, sr, chunk_s=7.0)
    assert whole.size == chunked.size == strength.size >= 60
    assert np.max(np.abs(whole - chunked)) < 0.002
    found = attacks_near(whole, strength, np.arange(64) * P128, 0.060)
    assert abs(found.placement) < 0.0015 and found.agreement(64) > 0.9      # bias measured at +0.02 ms


def test_attacks_prefer_the_strong_onset_near_each_beat():
    rng = np.random.default_rng(0)
    kicks = np.arange(16) * P128 + 0.004
    hats = np.sort(rng.uniform(0, 16 * P128, 120))
    times = np.concatenate([kicks, hats])
    strengths = np.concatenate([np.full(16, 3.0), rng.uniform(0.3, 1.5, 120)])
    order = np.argsort(times)
    found = attacks_near(times[order], strengths[order], np.arange(16) * P128, 0.060)
    assert found.placement == pytest.approx(0.004, abs=1e-9)
    assert found.agreement(16) == 1.0 and abs(found.drift(16)) < 1e-6


def test_drift_measures_a_tempo_error():
    beats = np.arange(16) * P128
    found = attacks_near(beats + np.arange(16) * 0.001, np.ones(16), beats, 0.060)   # 1 ms slip per beat
    assert found.drift(16) == pytest.approx(0.016, abs=1e-6)


def test_a_track_an_exact_number_of_chunks_long_fills_every_frame():
    from loopcutter.onsets import _track_envelope, onset_envelope

    sr = 48000
    x = np.random.default_rng(3).normal(0.0, 0.1, 2 * 2 * sr).astype("float32")   # two 2 s chunks
    np.testing.assert_allclose(_track_envelope(x, sr, 2.0), onset_envelope(x, sr), rtol=1e-5, atol=1e-6)


def test_silence_has_no_onsets():
    times, strengths = detect_onsets(np.zeros(48000, dtype="float32"), 48000)
    assert times.size == strengths.size == 0
