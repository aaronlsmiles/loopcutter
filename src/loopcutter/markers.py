"""Markers placed by hand in DJ software, read back as (file, start, end, name).

Times are in each app's own timeline. snap.py removes the app's measured offset
and resolves them against the cutter's grid; never cut at an app's timestamp.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .model import Marker, MarkerNote

STEMS = {"full", "drums", "bass", "other", "vocals", "guitar", "piano"}
SERATO_SUFFIXES = {".mp3", ".aif", ".aiff"}        # all serato-tools can open
_ROLLS = re.compile(r"^\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)*$")


def parse_note(text: str | None) -> MarkerNote:
    """'A1 bass 2,1' -> label A1, stem bass, roll lengths (2, 1).

    The first word that isn't a stem is the label, unless it contains a comma.
    A later number, or list of numbers, gives the rolls. So '16.2 bass' keeps
    16.2 as its label, and 'bass 2,1' has rolls but no label.
    """
    label = stem = None
    rolls: tuple[float, ...] = ()
    for token in (text or "").split():
        if token.lower() in STEMS and stem is None:
            stem = token.lower()
        elif label is None and "," not in token:
            label = token
        elif _ROLLS.match(token) and not rolls:
            rolls = tuple(float(v) for v in token.split(","))
    return MarkerNote(label, stem, rolls)


def _serato_entries(path: Path):
    from serato_tools.track_cues_v2 import TrackCuesV2

    return TrackCuesV2(str(path)).entries


def from_serato(paths: Iterable, reader: Callable | None = None) -> list[Marker]:
    reader = reader or _serato_entries
    markers = []
    for path in (Path(p) for p in paths):
        if path.suffix.lower() not in SERATO_SUFFIXES:
            continue
        for entry in reader(path):
            if hasattr(entry, "startposition"):
                markers.append(Marker(path, entry.startposition / 1000,
                                      entry.endposition / 1000, entry.name or "", "serato"))
            elif hasattr(entry, "position"):
                markers.append(Marker(path, entry.position / 1000, None, entry.name or "", "serato"))
    return markers


def _one(result):
    return result.one_or_none() if hasattr(result, "one_or_none") else result


def open_rekordbox():
    from pyrekordbox import Rekordbox6Database

    return Rekordbox6Database()


def from_rekordbox_db(playlist: str, db=None) -> list[Marker]:
    """Memory cues and memory loops of every track in a playlist. Hot cues are left alone."""
    db = db or open_rekordbox()
    found = _one(db.get_playlist(Name=playlist))
    if found is None:
        raise ValueError(f"rekordbox has no playlist called {playlist!r}")
    markers = []
    for content in db.get_playlist_contents(found):
        path = Path(content.FolderPath)
        for cue in db.get_cue(ContentID=content.ID):
            if cue.Kind != 0:
                continue
            out = cue.OutMsec if cue.OutMsec is not None and cue.OutMsec > cue.InMsec else None
            markers.append(Marker(path, cue.InMsec / 1000, None if out is None else out / 1000,
                                  cue.Comment or "", "rekordbox"))
    return sorted(markers, key=lambda m: (str(m.path), m.start))


def _xml_track_path(track) -> Path:
    """pyrekordbox's own Track.GETTERS already runs decode_path() on `Location`
    (rbxml.py), so track["Location"] never carries the "file:" prefix by the
    time we see it here - it's always pre-decoded. That decode strips exactly
    one leading "/" (its URL_PREFIX is "file://localhost/"), which round-trips
    fine for the Windows example in pyrekordbox's own docstring (localhost/ +
    C:/...) but not for a real Mac/Linux export, whose path already starts
    with "/": "file://localhost" + "/Users/..." decodes to "Users/..." with
    the leading slash silently eaten. Put it back unless this looks like a
    Windows drive path.
    """
    location = str(track["Location"])
    if location and location[0] != "/" and not re.match(r"^[A-Za-z]:", location):
        location = "/" + location
    return Path(location)


def from_rekordbox_xml(xml_path, playlist: str | None = None) -> list[Marker]:
    """Memory cues and loops (Num -1) from a rekordbox XML export."""
    from pyrekordbox.rbxml import RekordboxXml, decode_path

    xml = RekordboxXml(str(xml_path))
    if playlist:
        node = xml.get_playlist(*playlist.split("/"))
        if node.key_type == "TrackID":
            tracks = [xml.get_track(TrackID=key) for key in node.get_tracks()]
        else:
            tracks = [xml.get_track(Location=decode_path(key)) for key in node.get_tracks()]
    else:
        tracks = xml.get_tracks()
    markers = []
    for track in tracks:
        path = _xml_track_path(track)
        for mark in track.marks:
            if int(mark.get("Num", -1)) != -1:
                continue
            end = mark.get("End")
            markers.append(Marker(path, float(mark["Start"]),
                                  float(end) if end not in (None, "") else None,
                                  mark.get("Name") or "", "rekordbox-xml"))
    return markers


def serato_beats(path) -> np.ndarray:
    """Every beat of Serato's stored grid, in seconds of Serato's own timeline."""
    if Path(path).suffix.lower() not in SERATO_SUFFIXES:
        return np.array([])
    from serato_tools.track_beatgrid import TrackBeatgrid

    try:
        beats = TrackBeatgrid(str(path)).get_beats()
    except ValueError:
        return np.array([])
    return np.array([b.position_s for b in beats], dtype=float)


def rekordbox_beats(db, audio_path) -> np.ndarray:
    content = _one(db.get_content(FolderPath=str(audio_path)))
    if content is None:
        return np.array([])
    anlz = db.read_anlz_file(content, "DAT")
    if anlz is None:
        return np.array([])
    _beats, _bpms, times = anlz.get("beat_grid")
    return np.asarray(times, dtype=float)
