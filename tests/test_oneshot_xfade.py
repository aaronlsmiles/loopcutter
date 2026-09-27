import numpy as np
import pytest
import soundfile as sf

from loopcutter.cut import cut_loop, cut_oneshot
from loopcutter.manifest import ManifestError, load_manifest
from loopcutter.naming import build_filename, library_subdir
from loopcutter.verify import verify_cut


def test_oneshot_rows_need_start_and_end_not_bars(click_track, tmp_path):
    path = tmp_path / "o.csv"
    path.write_text(f"source,label,kind,start,end\n{click_track['path']},V1,oneshot,5.0,5.8\n")
    spec = load_manifest(path)[0]
    assert (spec.kind, spec.end_seconds, spec.bpm) == ("oneshot", 5.8, 0.0)
    assert build_filename(spec, "aiff").endswith("[V1][oneshot].aiff")
    assert str(library_subdir(spec)) == "full/oneshots"
    path.write_text(f"source,label,kind,start,end\n{click_track['path']},V1,oneshot,5.0,4.0\n")
    with pytest.raises(ManifestError, match="after"):
        load_manifest(path)


def test_oneshot_cut_is_the_marked_length_and_verifies(click_track, tmp_path):
    path = tmp_path / "o.csv"
    path.write_text(f"source,label,kind,start,end\n{click_track['path']},V1,oneshot,5.0,5.8\n")
    result = cut_oneshot(load_manifest(path)[0], tmp_path / "out", fmt="wav")
    assert abs(result.length_samples - round(0.8 * 44100)) <= 90
    assert verify_cut(result).ok


def test_crossfade_makes_the_wrap_continuous(tmp_path):
    sr = 44100
    t = np.arange(sr * 12) / sr
    tone = (0.5 * np.sin(2 * np.pi * 55.3 * t)).astype("float32")
    sf.write(str(tmp_path / "pad.wav"), np.column_stack([tone, tone]), sr)
    path = tmp_path / "p.csv"
    path.write_text(f"source,label,bars,bpm,start,xfade_ms\n{tmp_path / 'pad.wav'},P1,4,128,2.0,20\n")
    spec = load_manifest(path)[0]
    result = cut_loop(spec, tmp_path / "out", fmt="wav", xfade_ms=spec.xfade_ms)
    loop, _ = sf.read(str(result.output), dtype="float32")
    source, _ = sf.read(str(tmp_path / "pad.wav"), dtype="float32")
    assert loop[-1, 0] == pytest.approx(source[result.start_sample - 1, 0], abs=2e-3)
    assert loop[0, 0] == pytest.approx(source[result.start_sample, 0], abs=2e-3)
    assert len(loop) == result.length_samples
    assert verify_cut(result).ok


def test_a_oneshot_must_lie_inside_its_file(click_track, tmp_path):
    path = tmp_path / "o.csv"
    path.write_text(f"source,label,kind,start,end\n{click_track['path']},V1,oneshot,29.5,40.0\n")
    with pytest.raises(ValueError, match="row 2.*past the end"):
        cut_oneshot(load_manifest(path)[0], tmp_path / "out", fmt="wav")


def test_an_xfade_longer_than_the_loop_is_refused(click_track, tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(f"source,label,bars,bpm,start\n{click_track['path']},P1,0.25,128,15.0\n")
    with pytest.raises(ValueError, match="crossfade"):
        cut_loop(load_manifest(path)[0], tmp_path / "out", fmt="wav", xfade_ms=500)


def test_a_crossfade_with_no_audio_before_the_start_is_refused(click_track, tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(f"source,label,bars,bpm,start\n{click_track['path']},P1,4,128,0.005\n")
    with pytest.raises(ValueError, match="crossfade"):
        cut_loop(load_manifest(path)[0], tmp_path / "out", fmt="wav", snap_ms=0, xfade_ms=20)


def test_bad_oneshot_and_xfade_cells_name_the_row(click_track, tmp_path):
    path = tmp_path / "o.csv"
    path.write_text(f"source,label,kind,start,end\n{click_track['path']},V1,oneshot,-0.5,1.0\n")
    with pytest.raises(ManifestError, match="row 2"):
        load_manifest(path)
    path.write_text(f"source,label,bars,bpm,start,xfade_ms\n{click_track['path']},P1,4,128,15.0,abc\n")
    with pytest.raises(ManifestError, match="row 2.*xfade_ms"):
        load_manifest(path)


def test_stem_names_are_lowercased(click_track, tmp_path):
    path = tmp_path / "s.csv"
    path.write_text(f"source,label,bars,bpm,start,stem\n{click_track['path']},P1,4,128,15.0,Bass\n")
    assert load_manifest(path)[0].stem == "bass"
