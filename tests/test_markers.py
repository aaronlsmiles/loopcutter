from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from loopcutter.markers import from_rekordbox_db, from_rekordbox_xml, from_serato, parse_note
from loopcutter.model import Marker


def test_parse_note_reads_label_stem_and_rolls():
    note = parse_note("A1 bass 2,1")
    assert (note.label, note.stem, note.variations) == ("A1", "bass", (2.0, 1.0))
    assert parse_note("") == parse_note(None)
    assert parse_note("drums").stem == "drums" and parse_note("drums").label is None
    assert parse_note("16.2 bass").label == "16.2"
    assert parse_note("bass 2,1").variations == (2.0, 1.0) and parse_note("bass 2,1").label is None


def test_serato_reader_turns_entries_into_markers(tmp_path):
    entries = [NS(position=12120, name="A1"),
               NS(startposition=40000, endposition=47500, name="B2 bass 2,1"),
               NS(color=b"\xff\xff\xff")]
    track = tmp_path / "t.mp3"
    assert from_serato([track, tmp_path / "t.flac"], reader=lambda p: entries) == [
        Marker(track, 12.12, None, "A1", "serato"),
        Marker(track, 40.0, 47.5, "B2 bass 2,1", "serato"),
    ]                                                   # .flac is skipped: serato-tools can't open it


class _Query(list):
    def one_or_none(self):
        return self[0] if self else None


class _FakeDB:
    def __init__(self, path):
        self.content = NS(ID="c1", FolderPath=str(path))

    def get_playlist(self, Name):
        return _Query([NS(ID="p1")] if Name == "Sources" else [])

    def get_playlist_contents(self, playlist):
        return [self.content]

    def get_cue(self, ContentID):
        return [NS(InMsec=1500, OutMsec=-1, Comment="", Kind=0),
                NS(InMsec=8000, OutMsec=15500, Comment="A1 bass", Kind=0),
                NS(InMsec=3000, OutMsec=-1, Comment="hot cue", Kind=1)]


def test_rekordbox_db_reader_takes_memory_cues_only(tmp_path):
    assert from_rekordbox_db("Sources", db=_FakeDB(tmp_path / "x.aiff")) == [
        Marker(tmp_path / "x.aiff", 1.5, None, "", "rekordbox"),
        Marker(tmp_path / "x.aiff", 8.0, 15.5, "A1 bass", "rekordbox"),
    ]
    with pytest.raises(ValueError, match="no playlist"):
        from_rekordbox_db("Missing", db=_FakeDB(tmp_path / "x.aiff"))


XML = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="7.0.0" Company="AlphaTheta"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="Song" Artist="Artist" Location="file://localhost{path}" AverageBpm="128.00" TotalTime="300">
      <TEMPO Inizio="0.120" Bpm="128.00" Metro="4/4" Battito="1"/>
      <POSITION_MARK Name="A1 bass 2,1" Type="4" Start="12.120" End="19.620" Num="-1"/>
      <POSITION_MARK Name="" Type="0" Start="40.120" Num="-1"/>
      <POSITION_MARK Name="hot" Type="0" Start="50.000" Num="0"/>
    </TRACK>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="1">
      <NODE Name="Sources" Type="1" KeyType="0" Entries="1"><TRACK Key="1"/></NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
"""


def test_rekordbox_xml_reader_takes_memory_cues_only(tmp_path):
    audio = tmp_path / "Song.aiff"
    (tmp_path / "rb.xml").write_text(XML.format(path=audio))
    markers = from_rekordbox_xml(tmp_path / "rb.xml", playlist="Sources")
    assert [(m.start, m.end, m.name) for m in markers] == [(12.12, 19.62, "A1 bass 2,1"), (40.12, None, "")]
    assert all(Path(m.path) == audio for m in markers)
