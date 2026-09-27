"""ID3v2.3 tags on AIFF and WAV loops, so Sononym, Live and DJ apps see BPM, key and stem.

Tag AIFF by default. On WAV the tags go in an 'id3 ' chunk that some hardware
samplers reject.
"""

from __future__ import annotations

from pathlib import Path

import mutagen
from mutagen.aiff import AIFF
from mutagen.id3 import COMM, TBPM, TIT1, TIT2, TKEY, TPE1
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

from .keys import PC_NAMES, Key


def id3_key(key: Key) -> str:
    return PC_NAMES[key.pitch_class] + ("m" if key.mode == "minor" else "")


def _open(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".aif", ".aiff"):
        return AIFF(str(path))
    if suffix == ".wav":
        return WAVE(str(path))
    raise ValueError(f"can't tag {suffix} files; loops are AIFF or WAV")


def _set(tags, frame, text: str) -> None:
    tags.setall(frame.__name__, [frame(encoding=3, text=[text])])


def write_loop_tags(path, *, bpm: float | None, key: Key | None = None, stem: str | None = None,
                    title: str | None = None, artist: str | None = None,
                    comment: str | None = None) -> None:
    audio = _open(path)
    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags
    if bpm:
        _set(tags, TBPM, str(int(round(bpm))) if abs(bpm - round(bpm)) < 1e-6 else f"{bpm:.2f}")
    if key is not None:
        _set(tags, TKEY, id3_key(key))
    for frame, value in ((TIT1, stem), (TIT2, title), (TPE1, artist)):
        if value:
            _set(tags, frame, value)
    notes = " ".join(n for n in (key.camelot if key else None, comment) if n)
    if notes:
        tags.setall("COMM", [COMM(encoding=3, lang="eng", desc="", text=[notes])])
    audio.save(v2_version=3)


def read_tag(path, frame: str) -> str | None:
    tags = _open(path).tags
    found = tags.getall(frame) if tags is not None else []
    return str(found[0].text[0]) if found and found[0].text else None


def _basic_tags(source) -> dict[str, str]:
    audio = mutagen.File(str(source))
    tags = getattr(audio, "tags", None)
    if not tags:
        return {}
    found: dict[str, str] = {}
    if hasattr(tags, "getall"):
        for frame, name in (("TPE1", "artist"), ("TIT2", "title"), ("TKEY", "key"), ("TBPM", "bpm")):
            values = tags.getall(frame)
            if values and values[0].text:
                found[name] = str(values[0].text[0])
    elif isinstance(audio, MP4):
        for atom, name in (("\xa9ART", "artist"), ("\xa9nam", "title")):
            if atom in tags:
                found[name] = str(tags[atom][0])
        if "tmpo" in tags:
            found["bpm"] = str(tags["tmpo"][0])
    else:
        for field, name in (("artist", "artist"), ("title", "title"), ("initialkey", "key"), ("bpm", "bpm")):
            if field in tags:
                found[name] = str(tags[field][0])
    return found


def copy_all_tags(source, dest) -> None:
    """Every ID3 frame of one AIFF or WAV onto another, over what's there. Used when a master
    is rebuilt, so tags other apps wrote to it (a Mixed In Key key, say) survive."""
    tags = _open(source).tags
    if not tags:
        return
    audio = _open(dest)
    if audio.tags is None:
        audio.add_tags()
    for frame in tags.values():
        audio.tags.add(frame)
    audio.save(v2_version=3)


def copy_basic_tags(source, dest) -> None:
    """Carry artist, title, key and BPM (never DJ-app cue data) onto a master."""
    found = _basic_tags(source)
    if not found:
        return
    audio = _open(dest)
    if audio.tags is None:
        audio.add_tags()
    for name, frame in (("artist", TPE1), ("title", TIT2), ("key", TKEY), ("bpm", TBPM)):
        if name in found:
            _set(audio.tags, frame, found[name])
    audio.save(v2_version=3)
