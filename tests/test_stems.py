import numpy as np
import pytest
import soundfile as sf

from loopcutter.stems import collect_stems, conform_stems, separate


def test_collect_reads_both_naming_styles(tmp_path):
    files = [tmp_path / "Song (Remix)_(Vocals)_m.wav", tmp_path / "Song (Remix)_(Bass)_m.wav",
             tmp_path / "drums.wav", tmp_path / "notes.txt"]
    assert set(collect_stems(files)) == {"vocals", "bass", "drums"}


def test_separate_runs_once_and_reuses(tmp_path):
    calls = []

    def runner(source, out_dir, model):
        calls.append(model)
        paths = [out_dir / f"{source.stem}_({n})_{model}.wav" for n in ("Drums", "Bass")]
        for p in paths:
            p.write_bytes(b"")
        return paths

    source = tmp_path / "Song.aiff"; source.write_bytes(b"")
    first = separate(source, tmp_path / "stems", model="m", runner=runner)
    assert set(first) == {"drums", "bass"}
    assert separate(source, tmp_path / "stems", model="m", runner=runner) == first and calls == ["m"]


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
