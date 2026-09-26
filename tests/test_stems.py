import numpy as np
import pytest
import soundfile as sf

from loopcutter.stems import collect_stems, conform_stems, separate


def test_collect_reads_both_naming_styles(tmp_path):
    files = [tmp_path / "Song (Remix)_(Vocals)_m.wav", tmp_path / "Song (Remix)_(Bass)_m.wav",
             tmp_path / "drums.wav", tmp_path / "notes.txt"]
    assert set(collect_stems(files)) == {"vocals", "bass", "drums"}


def _runner(calls):
    def runner(source, out_dir, model):
        calls.append(model)
        paths = [out_dir / f"{source.stem}_({n})_{model}.wav" for n in ("Drums", "Bass")]
        for p in paths:
            p.write_bytes(b"")
        return paths
    return runner


def test_separate_runs_once_and_reuses(tmp_path):
    calls = []
    source = tmp_path / "Song.aiff"; source.write_bytes(b"")
    first = separate(source, tmp_path / "stems", model="m", runner=_runner(calls))
    assert set(first) == {"drums", "bass"}
    assert separate(source, tmp_path / "stems", model="m", runner=_runner(calls)) == first
    assert calls == ["m"]


def test_separate_caches_per_engine_and_model(tmp_path):
    calls = []
    source = tmp_path / "Song.aiff"; source.write_bytes(b"")
    for model in ("m", "n", "m"):
        separate(source, tmp_path / "stems", model=model, runner=_runner(calls))
    separate(source, tmp_path / "stems", engine="demucs", model="m", runner=_runner(calls))
    assert calls == ["m", "n", "m"]


def test_separate_reruns_over_output_it_never_finished(tmp_path):
    calls = []
    source = tmp_path / "Song.aiff"; source.write_bytes(b"")

    def crashes(source, out_dir, model):
        (out_dir / "Song_(Drums)_m.wav").write_bytes(b"")
        raise RuntimeError("killed")

    with pytest.raises(RuntimeError, match="killed"):
        separate(source, tmp_path / "stems", model="m", runner=crashes)
    assert set(separate(source, tmp_path / "stems", model="m", runner=_runner(calls))) == {"drums", "bass"}
    assert calls == ["m"]


def test_separate_reruns_when_the_source_changes(tmp_path):
    calls = []
    source = tmp_path / "Song.aiff"; source.write_bytes(b"")
    separate(source, tmp_path / "stems", model="m", runner=_runner(calls))
    source.write_bytes(b"new master")
    separate(source, tmp_path / "stems", model="m", runner=_runner(calls))
    separate(source, tmp_path / "stems", model="m", runner=_runner(calls))
    assert calls == ["m", "m"]


def test_separate_fails_loudly_when_nothing_comes_out(tmp_path):
    source = tmp_path / "Song.aiff"; source.write_bytes(b"")
    with pytest.raises(RuntimeError, match="no stems"):
        separate(source, tmp_path / "stems", runner=lambda s, o, m: [])


def _noise(path, sr, seed, seconds=6.0, shift=0):
    x = np.random.default_rng(seed).normal(0.0, 0.1, int(sr * seconds)).astype("float32")
    x = np.roll(x, shift)
    sf.write(str(path), np.column_stack([x, x]), sr)
    return x


def test_conform_resamples_to_the_master_rate_and_checks_alignment(tmp_path):
    import soxr

    a, b = _noise(tmp_path / "a.wav", 44100, 1), _noise(tmp_path / "b.wav", 44100, 2)
    mix = soxr.resample(np.column_stack([a + b, a + b]), 44100, 48000, quality="VHQ")
    master = tmp_path / "master.aiff"
    sf.write(str(master), mix, 48000, format="AIFF", subtype="PCM_24")
    out = conform_stems({"bass": tmp_path / "a.wav", "other": tmp_path / "b.wav"}, master, tmp_path / "out")
    assert set(out) == {"bass", "other"}
    assert all(sf.info(str(p)).samplerate == 48000 and p.parent == tmp_path / "out" for p in out.values())
    _noise(tmp_path / "a_late.wav", 44100, 1, shift=400)
    _noise(tmp_path / "b_late.wav", 44100, 2, shift=400)
    with pytest.raises(RuntimeError, match="line up"):
        conform_stems({"bass": tmp_path / "a_late.wav", "other": tmp_path / "b_late.wav"},
                      master, tmp_path / "late")


def _master(tmp_path, parts, sr=44100):
    mix = np.sum(parts, axis=0)
    master = tmp_path / "master.aiff"
    sf.write(str(master), np.column_stack([mix, mix]), sr, format="AIFF", subtype="PCM_24")
    return master


def test_conform_leaves_no_usable_stems_when_alignment_fails(tmp_path):
    a, b = _noise(tmp_path / "a.wav", 44100, 1), _noise(tmp_path / "b.wav", 44100, 2)
    master, out = _master(tmp_path, [a, b]), tmp_path / "out"
    conform_stems({"bass": tmp_path / "a.wav", "other": tmp_path / "b.wav"}, master, out)
    (out / "raw").mkdir()
    (out / "raw" / "cached.wav").write_bytes(b"")
    assert sorted(p.name for p in out.glob("*.aiff")) == ["bass.aiff", "other.aiff"]
    _noise(tmp_path / "a_late.wav", 44100, 1, shift=400)
    _noise(tmp_path / "b_late.wav", 44100, 2, shift=400)
    with pytest.raises(RuntimeError, match="line up"):
        conform_stems({"bass": tmp_path / "a_late.wav", "other": tmp_path / "b_late.wav"}, master, out)
    assert not list(out.glob("*.aiff")) and not list(tmp_path.glob("out.*"))
    assert (out / "raw" / "cached.wav").exists()


def test_conform_replaces_the_previous_set_and_keeps_the_raw_cache(tmp_path):
    a, b = _noise(tmp_path / "a.wav", 44100, 1), _noise(tmp_path / "b.wav", 44100, 2)
    master, out = _master(tmp_path, [a, b]), tmp_path / "out"
    out.mkdir()
    (out / "guitar.aiff").write_bytes(b"stale")
    (out / "raw").mkdir()
    (out / "raw" / "cached.wav").write_bytes(b"")
    conform_stems({"bass": tmp_path / "a.wav", "other": tmp_path / "b.wav"}, master, out)
    assert sorted(p.name for p in out.glob("*.aiff")) == ["bass.aiff", "other.aiff"]
    assert (out / "raw" / "cached.wav").exists() and not list(tmp_path.glob("out.*"))


def test_conform_rejects_stems_whose_level_doesnt_match_the_master(tmp_path):
    a, b = _noise(tmp_path / "a.wav", 44100, 1), _noise(tmp_path / "b.wav", 44100, 2)
    master = _master(tmp_path, [a / 0.7, b / 0.7])
    with pytest.raises(RuntimeError, match=r"level doesn't match the master: -3\.1 dB"):
        conform_stems({"bass": tmp_path / "a.wav", "other": tmp_path / "b.wav"}, master, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.aiff"))


def test_conform_reports_gain_and_clipped_samples(tmp_path):
    a = np.random.default_rng(1).normal(0.0, 0.1, 44100 * 6).astype("float32")
    b = np.random.default_rng(2).normal(0.0, 0.1, 44100 * 6).astype("float32")
    a[1000:1003] = 1.5
    for name, x in (("a", a), ("b", b)):
        sf.write(str(tmp_path / f"{name}.wav"), np.column_stack([x, x]), 44100, subtype="FLOAT")
    master = tmp_path / "master.wav"
    sf.write(str(master), np.column_stack([a + b, a + b]), 44100, subtype="FLOAT")
    out = conform_stems({"bass": tmp_path / "a.wav", "other": tmp_path / "b.wav"}, master, tmp_path / "out")
    assert out == {"bass": tmp_path / "out" / "bass.aiff", "other": tmp_path / "out" / "other.aiff"}
    assert abs(out.gain_db) < 0.01 and out.clipped == {"bass": 6, "other": 0}


def test_separators_keep_level_and_resolution(tmp_path, monkeypatch):
    import sys
    import types

    from loopcutter import stems

    made = {}

    class FakeSeparator:
        def __init__(self, **kwargs):
            made.update(kwargs)

        def load_model(self, model_filename):
            pass

        def separate(self, source):
            return []

    module = types.ModuleType("audio_separator.separator")
    module.Separator = FakeSeparator
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)
    stems._audio_separator(tmp_path / "Song.aiff", tmp_path, "m")
    assert made["normalization_threshold"] == 1.0 and made["use_soundfile"] is True

    commands = []
    monkeypatch.setattr(stems.subprocess, "run", lambda cmd, check: commands.append(cmd))
    stems._demucs(tmp_path / "Song.aiff", tmp_path, "m")
    assert "--float32" in commands[0] and commands[0][commands[0].index("--clip-mode") + 1] == "none"
