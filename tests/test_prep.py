import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from loopcutter.audio_io import read_audio
import loopcutter.prep as prep
from loopcutter.prep import choose_sources, find_sources, master_path, prepare_master


def _tone(path, sr=44100, seconds=2.0, amp=0.5, fmt=None, subtype=None):
    t = np.arange(int(sr * seconds)) / sr
    x = (amp * np.sin(2 * np.pi * 440 * t)).astype("float32")
    sf.write(str(path), np.column_stack([x, x]), sr, format=fmt, subtype=subtype)
    return path


def test_master_is_48k_24bit_aiff(tmp_path):
    result = prepare_master(_tone(tmp_path / "in.wav"), tmp_path / "masters")
    info = sf.info(str(result.master))
    assert (info.samplerate, info.format, info.subtype) == (48000, "AIFF", "PCM_24")
    assert abs(info.frames - 96000) <= 1
    assert (result.source_rate, result.reused, result.clipped, result.stale) == (44100, False, 0, False)


def test_an_existing_master_is_never_overwritten(tmp_path):
    src = _tone(tmp_path / "in.wav")
    first = prepare_master(src, tmp_path / "m")
    stamp = first.master.stat().st_mtime_ns
    os.utime(src, ns=(stamp + 10**9, stamp + 10**9))               # the source changes later
    again = prepare_master(src, tmp_path / "m")
    assert again.reused and again.stale and again.master.stat().st_mtime_ns == stamp


def test_mp3_decodes_through_soundfile(tmp_path):
    audio, sr = read_audio(_tone(tmp_path / "in.mp3", fmt="MP3", subtype="MPEG_LAYER_III"))
    assert sr == 44100 and audio.shape[1] == 2


def test_overs_from_resampling_are_counted(tmp_path):
    sr = 44100
    square = np.sign(np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)).astype("float32")
    sf.write(str(tmp_path / "sq.wav"), np.column_stack([square, square]), sr, subtype="FLOAT")
    assert prepare_master(tmp_path / "sq.wav", tmp_path / "m").clipped > 0


def test_collisions_name_both_sources(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    one, two = _tone(tmp_path / "a" / "x.wav"), _tone(tmp_path / "b" / "x.wav")
    with pytest.raises(ValueError, match="x.wav"):
        choose_sources([one, two], tmp_path / "m")


def test_find_sources_walks_folders_and_skips_hidden(tmp_path):
    _tone(tmp_path / "a.wav"); (tmp_path / ".hidden.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in find_sources([tmp_path])] == ["a.wav"]
    assert master_path(tmp_path / "Artist - Song?.wav", tmp_path).name == "Artist - Song-.aiff"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_aac_decodes_through_ffmpeg(tmp_path):
    m4a = tmp_path / "in.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-ac", "2", "-c:a", "aac", str(m4a)], check=True)
    audio, sr = read_audio(m4a)
    assert audio.shape[1] == 2 and 0.9 * sr < len(audio) < 1.2 * sr


def test_a_lossless_file_always_wins_over_a_lossy_copy(tmp_path):
    (tmp_path / "lossless").mkdir(); (tmp_path / "lossy").mkdir()
    wav = _tone(tmp_path / "lossless" / "x.wav")
    mp3 = _tone(tmp_path / "lossy" / "x.mp3", fmt="MP3", subtype="MPEG_LAYER_III")
    other = _tone(tmp_path / "lossy" / "y.mp3", fmt="MP3", subtype="MPEG_LAYER_III")
    kept, skipped = choose_sources([mp3, other, wav], tmp_path / "m")
    assert kept == [other, wav] and skipped == [(mp3, wav)]
    with pytest.raises(ValueError, match="x.mp3"):                       # two lossy copies: no winner
        choose_sources([mp3, _tone(tmp_path / "x.mp3", fmt="MP3", subtype="MPEG_LAYER_III")], tmp_path / "m")


def test_headroom_keeps_overs_from_clipping(tmp_path):
    sr = 44100
    square = np.sign(np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)).astype("float32")
    sf.write(str(tmp_path / "sq.wav"), np.column_stack([square, square]), sr, subtype="FLOAT")
    result = prepare_master(tmp_path / "sq.wav", tmp_path / "m", gain_db=-3.0)
    master, _ = sf.read(str(result.master))
    assert result.clipped == 0 and result.gain_db == -3.0
    assert 0.85 < np.max(np.abs(master)) < 1.0          # the resampled square overshoots ~1.27; -3 dB fits it


def test_clashes_are_grouped_as_the_filesystem_sees_them(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    mp3 = _tone(tmp_path / "a" / "Song.mp3", fmt="MP3", subtype="MPEG_LAYER_III")
    wav = _tone(tmp_path / "b" / "song.wav")
    assert choose_sources([mp3, wav], tmp_path / "m") == ([wav], [(mp3, wav)])
    assert choose_sources([wav, wav, tmp_path / "b" / ".." / "b" / "song.wav"], tmp_path / "m") == ([wav], [])


def test_an_m4a_is_lossless_only_if_it_holds_alac(tmp_path, monkeypatch):
    m4a, mp3 = tmp_path / "T.m4a", _tone(tmp_path / "T.mp3", fmt="MP3", subtype="MPEG_LAYER_III")
    m4a.write_bytes(b"")
    monkeypatch.setattr(prep, "source_seconds", lambda path: 2.0)             # both copies 2 s long
    monkeypatch.setattr(prep, "_codec", lambda path: "alac")
    assert prep.is_lossless(m4a) is True
    assert choose_sources([mp3, m4a], tmp_path / "m") == ([m4a], [(mp3, m4a)])
    monkeypatch.setattr(prep, "_codec", lambda path: "aac")
    assert prep.is_lossless(m4a) is False
    monkeypatch.setattr(prep, "_codec", lambda path: None)
    assert prep.is_lossless(m4a) is None
    with pytest.raises(ValueError, match="T.aiff"):
        choose_sources([mp3, m4a], tmp_path / "m")


def test_a_lossless_file_of_a_different_length_does_not_silently_win(tmp_path):
    (tmp_path / "lossless").mkdir(); (tmp_path / "lossy").mkdir()
    wav = _tone(tmp_path / "lossless" / "x.wav", seconds=5.0)
    mp3 = _tone(tmp_path / "lossy" / "x.mp3", seconds=2.0, fmt="MP3", subtype="MPEG_LAYER_III")
    with pytest.raises(ValueError, match="x.mp3"):
        choose_sources([mp3, wav], tmp_path / "m")


def test_copy_all_tags_leaves_dj_app_cue_data_behind(tmp_path):
    from mutagen.aiff import AIFF
    from mutagen.id3 import GEOB, PRIV, TKEY, TLEN

    from loopcutter.tags import copy_all_tags, read_tag

    old = _tone(tmp_path / "old.aiff", fmt="AIFF")
    new = _tone(tmp_path / "new.aiff", fmt="AIFF")
    f = AIFF(str(old)); f.add_tags()
    f.tags.add(TKEY(encoding=3, text=["Am"]))
    f.tags.add(GEOB(encoding=0, mime="application/octet-stream", filename="", desc="Serato Markers2", data=b"x"))
    f.tags.add(PRIV(owner="TRAKTOR4", data=b"x"))
    f.tags.add(TLEN(encoding=3, text=["1000"]))
    f.tags.add(GEOB(encoding=0, mime="image/png", filename="art.png", desc="artwork", data=b"y"))
    f.save(v2_version=3)
    copy_all_tags(old, new)
    tags = AIFF(str(new)).tags
    assert read_tag(new, "TKEY") == "Am"
    assert [g.desc for g in tags.getall("GEOB")] == ["artwork"]            # only Serato's go
    assert not tags.getall("PRIV") and not tags.getall("TLEN")


def test_find_sources_skips_anything_inside_a_hidden_folder(tmp_path):
    _tone(tmp_path / "a.wav")
    (tmp_path / ".replaced").mkdir()
    _tone(tmp_path / ".replaced" / "old.wav")
    assert [p.name for p in find_sources([tmp_path])] == ["a.wav"]
