import shutil

import numpy as np
import pytest
import soundfile as sf

from loopcutter.keydetect import detect_key, key_from_keyfinder, key_from_tags
from loopcutter.keys import parse_key
from loopcutter.model import TrackRecord
from loopcutter.naming import bpm_band, library_subdir
from loopcutter.trackdb import find_record, load_tracks, save_tracks, track_id_for


def _rec(**kw):
    base = dict(track_id="Artist - Song", source="/s/a.mp3", master="/m/Artist - Song.aiff",
                sample_rate=48000, duration=300.0, bpm=125.0, phase=0.12, bar_phase=1)
    base.update(kw)
    return TrackRecord(**base)


def test_roundtrip_keeps_types_nones_and_overrides(tmp_path):
    path = tmp_path / "analysis" / "tracks.csv"
    rec = _rec(flags="grid-fit;key", override_phase_ms=-234.4)
    save_tracks(path, {rec.track_id: rec})
    back = load_tracks(path)["Artist - Song"]
    assert back == rec and isinstance(back.bar_phase, int) and back.app_offset_ms is None


def test_missing_file_is_empty(tmp_path):
    assert load_tracks(tmp_path / "none.csv") == {}


def test_find_record_by_source_or_master(tmp_path):
    rec = _rec(source=str(tmp_path / "a.mp3"), master=str(tmp_path / "A.aiff"))
    records = {rec.track_id: rec}
    assert find_record(records, tmp_path / "a.mp3") is rec
    assert find_record(records, tmp_path / "A.aiff") is rec
    assert find_record(records, tmp_path / "other.mp3") is None
    assert track_id_for(tmp_path / "A.aiff") == "A"


def test_library_layout():
    class Spec:
        stem, bpm = None, 128.0

    assert bpm_band(128) == "125-129" and bpm_band(124.99) == "120-124"
    assert str(library_subdir(Spec())) == "full/125-129"
    Spec.stem = "bass"
    assert str(library_subdir(Spec())) == "bass/125-129"


def _aiff(path):
    sf.write(str(path), np.zeros((4800, 2), dtype="float32"), 48000, format="AIFF")
    return path


def test_key_from_tag_and_mik_comment(tmp_path):
    from mutagen.aiff import AIFF
    from mutagen.id3 import COMM, TKEY

    path = _aiff(tmp_path / "k.aiff")
    f = AIFF(str(path)); f.add_tags()
    f.tags.add(COMM(encoding=3, lang="eng", desc="", text=["8A - Energy 6"])); f.save()
    assert key_from_tags(path) == (parse_key("8A"), "comment")
    f = AIFF(str(path)); f.tags.add(TKEY(encoding=3, text=["F#m"])); f.save()
    assert key_from_tags(path) == (parse_key("F#m"), "tag")


def test_detect_key_records_disagreement(tmp_path):
    from mutagen.aiff import AIFF
    from mutagen.id3 import TKEY

    path = _aiff(tmp_path / "k.aiff")
    f = AIFF(str(path)); f.add_tags(); f.tags.add(TKEY(encoding=3, text=["Em"])); f.save()
    result = detect_key(path, keyfinder=lambda p: parse_key("11A"))
    assert result.key == parse_key("Em") and result.alternative == parse_key("11A") and result.disagrees
    assert detect_key(path, keyfinder=lambda p: parse_key("9A")).alternative is None


@pytest.mark.skipif(shutil.which("keyfinder-cli") is None, reason="needs keyfinder-cli")
def test_key_from_keyfinder_reads_24bit_aiff(tmp_path):
    """keyfinder-cli can't resample 24-bit/float PCM itself; we must downsample to 16-bit first."""
    sr, seconds = 48000, 3
    t = np.arange(int(sr * seconds)) / sr
    a_minor = 0.2 * (np.sin(2 * np.pi * 220.0 * t)          # A3
                      + np.sin(2 * np.pi * 261.63 * t)      # C4
                      + np.sin(2 * np.pi * 329.63 * t))     # E4
    path = tmp_path / "chord.aiff"
    sf.write(str(path), np.column_stack([a_minor, a_minor]).astype("float32"), sr,
             format="AIFF", subtype="PCM_24")
    assert key_from_keyfinder(path) == parse_key("8A")
