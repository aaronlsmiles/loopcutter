import csv
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from loopcutter.cli import main
from loopcutter.manifest import LoopSpec, load_manifest
from loopcutter.model import TrackRecord
from loopcutter.naming import build_filename, loop_code
from loopcutter.tags import read_tag
from loopcutter.trackdb import save_tracks
from loopcutter.workspace import init_workspace

ROW = ("source,track_id,label,artist,track,bars,bpm,start,stem,key\n"
       "{master},Artist - Clicks,A1,Artist,Clicks,4,128,15.0,,8A\n")


def _ws(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    master = ws.masters / "Artist - Clicks.aiff"
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    sf.write(str(master), audio, sr, format="AIFF", subtype="PCM_24")
    save_tracks(ws.tracks_csv, {"Artist - Clicks": TrackRecord(
        track_id="Artist - Clicks", source="/music/Artist - Clicks (Original Mix).flac", master=str(master),
        sample_rate=sr, duration=30.0, bpm=128.0, phase=0.0, bar_phase=0, bpm_fitted=128.0,
        inlier_ratio=1.0, bar_agreement=1.0, phase_agreement=1.0, key="8A")})
    (ws.manifests / "s.csv").write_text(ROW.format(master=master))
    monkeypatch.chdir(ws.root)
    return ws


def _index(ws, fmt="aiff"):
    with open(ws.loops / fmt / "provenance.csv") as f:
        return list(csv.DictReader(f))


def _spec(**kw):
    base = dict(source=Path("/m/a.aiff"), label="A1", bars=4, bpm=128.0, start_seconds=15.0,
                artist="Artist", track="Clicks", track_id="Artist - Clicks")
    base.update(kw)
    return LoopSpec(**base)


def test_loop_code_is_short_stable_and_keyed_on_what_the_loop_is():
    a = _spec()
    assert re.fullmatch(r"LC-[0-9a-f]{8}", loop_code(a))
    assert loop_code(a) == loop_code(replace(a, start_seconds=15.02))      # re-marking keeps the identity
    for other in (replace(a, label="A2"), replace(a, bars=2), replace(a, stem="bass"),
                  replace(a, track_id="Other - Track")):
        assert loop_code(other) != loop_code(a)


def test_the_code_is_the_last_bracket_of_the_name():
    spec = _spec(key="8A", stem="bass")
    assert build_filename(spec, "aiff", code="LC-0123abcd") == "Artist - Clicks [A1][128][4bar][8A][bass][LC-0123abcd].aiff"
    assert build_filename(spec, "aiff") == "Artist - Clicks [A1][128][4bar][8A][bass].aiff"


def test_cut_names_tags_and_indexes_every_loop(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    assert main(["cut", "manifests/s.csv"]) == 0
    spec = load_manifest(ws.manifests / "s.csv")[0]
    code = loop_code(spec)
    out = ws.loops / "aiff" / "full" / "125-129" / f"Artist - Clicks [A1][128][4bar][8A][{code}].aiff"
    assert out.exists()
    assert read_tag(out, "TXXX:PROVENANCE_CODE") == code
    source = read_tag(out, "TXXX:SOURCE")
    assert "/music/Artist - Clicks (Original Mix).flac" in source and "15.000000 s" in source and "4 bars" in source
    row = _index(ws)[0]
    assert (row["code"], row["file"], row["artist"], row["track"]) == (
        code, f"full/125-129/{out.name}", "Artist", "Clicks")
    assert (row["source_file"], row["start_seconds"], row["bars"], row["bpm"]) == (
        "/music/Artist - Clicks (Original Mix).flac", "15.000000", "4", "128")
    assert row["manifest"] == "s.csv" and row["row"] == "2"


def test_recut_replaces_the_old_uncoded_copy_and_keeps_one_index_row(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    assert main(["cut", "manifests/s.csv", "--no-code"]) == 0
    old = ws.loops / "aiff" / "full" / "125-129" / "Artist - Clicks [A1][128][4bar][8A].aiff"
    assert old.exists()
    assert main(["cut", "manifests/s.csv"]) == 0
    assert not old.exists()
    assert len(list((ws.loops / "aiff").rglob("*.aiff"))) == 1
    assert len(_index(ws)) == 1


def test_no_code_still_indexes_and_wav_is_indexed_without_tags(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    assert main(["cut", "manifests/s.csv", "--no-code"]) == 0
    assert _index(ws)[0]["file"].endswith("[A1][128][4bar][8A].aiff")
    assert main(["cut", "manifests/s.csv", "--format", "wav"]) == 0
    wav_row = _index(ws, "wav")[0]
    assert wav_row["file"].endswith(".wav") and wav_row["code"].startswith("LC-")


def test_provenance_backfills_loops_cut_before_codes(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    assert main(["cut", "manifests/s.csv", "--no-code"]) == 0
    (ws.loops / "aiff" / "provenance.csv").unlink()                         # as if cut by an older version
    assert main(["provenance", "manifests/s.csv"]) == 0
    files = list((ws.loops / "aiff").rglob("*.aiff"))
    assert len(files) == 1 and "[LC-" in files[0].name
    assert read_tag(files[0], "TXXX:PROVENANCE_CODE") == _index(ws)[0]["code"]


def test_provenance_can_keep_the_names(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    assert main(["cut", "manifests/s.csv", "--no-code"]) == 0
    assert main(["provenance", "manifests/s.csv", "--keep-names"]) == 0
    files = list((ws.loops / "aiff").rglob("*.aiff"))
    assert "[LC-" not in files[0].name
    assert read_tag(files[0], "TXXX:PROVENANCE_CODE") == _index(ws)[0]["code"]


def test_code_ignores_unicode_normalisation():
    import unicodedata

    nfc = _spec(track_id="Artist - Zürüx", track="Zürüx")
    nfd = _spec(track_id=unicodedata.normalize("NFD", "Artist - Zürüx"), track=unicodedata.normalize("NFD", "Zürüx"))
    assert loop_code(nfc) == loop_code(nfd)


def test_a_stale_code_is_corrected_not_duplicated(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    assert main(["cut", "manifests/s.csv"]) == 0
    [loop] = list((ws.loops / "aiff").rglob("*.aiff"))
    stale = loop.with_name(re.sub(r"\[LC-[0-9a-f]{8}\]", "[LC-00000000]", loop.name))
    loop.rename(stale)                                                      # as if coded from another spelling
    assert main(["provenance", "manifests/s.csv"]) == 0
    files = list((ws.loops / "aiff").rglob("*.aiff"))
    assert [f.name for f in files] == [loop.name]
    rows = _index(ws)
    assert len(rows) == 1 and rows[0]["code"] == loop_code(load_manifest(ws.manifests / "s.csv")[0])


def test_cut_replaces_a_copy_with_a_stale_code(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    assert main(["cut", "manifests/s.csv"]) == 0
    [loop] = list((ws.loops / "aiff").rglob("*.aiff"))
    loop.rename(loop.with_name(re.sub(r"\[LC-[0-9a-f]{8}\]", "[LC-00000000]", loop.name)))
    assert main(["cut", "manifests/s.csv"]) == 0
    assert [f.name for f in (ws.loops / "aiff").rglob("*.aiff")] == [loop.name]


def test_recut_keeps_a_loop_whose_file_name_is_stored_decomposed(tmp_path, click_track, monkeypatch):
    import unicodedata

    ws = _ws(tmp_path, click_track, monkeypatch)
    master = ws.masters / "Artist - Clicks.aiff"
    (ws.manifests / "s.csv").write_text("source,track_id,label,artist,track,bars,bpm,start,stem,key\n"
                                        f"{master},Artist - Clicks,A1,Artist,Zürüx,4,128,15.0,,8A\n")
    assert main(["cut", "manifests/s.csv"]) == 0
    [loop] = list((ws.loops / "aiff").rglob("*.aiff"))
    loop.rename(loop.with_name(unicodedata.normalize("NFD", loop.name)))    # how macOS may hand names back
    assert main(["cut", "manifests/s.csv"]) == 0
    assert len(list((ws.loops / "aiff").rglob("*.aiff"))) == 1


def test_index_replaces_the_row_of_a_stale_code_but_keeps_deleted_loops():
    from loopcutter.provenance import FIELDS, update_index
    import tempfile

    root = Path(tempfile.mkdtemp())
    blank = {k: "" for k in FIELDS}
    update_index(root, [dict(blank, code="LC-00000000", file="full/A [A1][128][4bar][LC-00000000].aiff"),
                        dict(blank, code="LC-11111111", file="full/Gone [A1][128][4bar][LC-11111111].aiff")])
    update_index(root, [dict(blank, code="LC-22222222", file="full/A [A1][128][4bar][LC-22222222].aiff")])
    with open(root / "provenance.csv") as f:
        codes = sorted(r["code"] for r in csv.DictReader(f))
    assert codes == ["LC-11111111", "LC-22222222"]
