import numpy as np
import soundfile as sf

from loopcutter.keys import parse_key
from loopcutter.tags import copy_basic_tags, id3_key, read_tag, write_loop_tags


def _silence(path, fmt):
    sf.write(str(path), np.zeros((4800, 2), dtype="float32"), 48000, format=fmt)
    return path


def test_id3_key_spelling():
    assert (id3_key(parse_key("8A")), id3_key(parse_key("F# minor")), id3_key(parse_key("8B"))) == ("Am", "F#m", "C")


def test_write_loop_tags_is_id3v23_and_leaves_audio_alone(tmp_path):
    from mutagen.aiff import AIFF

    path = _silence(tmp_path / "l.aiff", "AIFF")
    write_loop_tags(path, bpm=128.0, key=parse_key("8A"), stem="bass",
                    title="Track [A1][4bar]", artist="Artist", comment="A1")
    assert (read_tag(path, "TBPM"), read_tag(path, "TKEY"), read_tag(path, "TIT1")) == ("128", "Am", "bass")
    assert read_tag(path, "COMM") == "8A A1"
    assert AIFF(str(path)).tags.version[:2] == (2, 3)
    assert sf.info(str(path)).frames == 4800


def test_fractional_and_missing_bpm(tmp_path):
    path = _silence(tmp_path / "l.aiff", "AIFF")
    write_loop_tags(path, bpm=127.43)
    assert read_tag(path, "TBPM") == "127.43"
    other = _silence(tmp_path / "o.aiff", "AIFF")
    write_loop_tags(other, bpm=None, title="Vox [V1][oneshot]")
    assert read_tag(other, "TBPM") is None and read_tag(other, "TIT2") == "Vox [V1][oneshot]"


def test_copy_basic_tags_from_mp3_to_master(tmp_path):
    from mutagen.id3 import TIT2, TKEY, TPE1
    from mutagen.mp3 import MP3

    src = tmp_path / "s.mp3"
    sf.write(str(src), np.zeros((44100, 2), dtype="float32"), 44100, format="MP3", subtype="MPEG_LAYER_III")
    audio = MP3(str(src))
    if audio.tags is None:
        audio.add_tags()
    for frame in (TPE1(encoding=3, text=["Artist"]), TIT2(encoding=3, text=["Song"]), TKEY(encoding=3, text=["Gm"])):
        audio.tags.add(frame)
    audio.save()
    dest = _silence(tmp_path / "m.aiff", "AIFF")
    copy_basic_tags(src, dest)
    assert (read_tag(dest, "TPE1"), read_tag(dest, "TIT2"), read_tag(dest, "TKEY")) == ("Artist", "Song", "Gm")
