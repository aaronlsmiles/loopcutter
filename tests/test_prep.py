import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from loopcutter.audio_io import read_audio
from loopcutter.prep import check_collisions, choose_sources, find_sources, master_path, prepare_master


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
        check_collisions([one, two], tmp_path / "m")


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
