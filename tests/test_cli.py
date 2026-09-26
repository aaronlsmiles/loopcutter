import numpy as np
import pytest
import soundfile as sf

from loopcutter.cli import main
from loopcutter.manifest import load_manifest
from loopcutter.model import TrackRecord
from loopcutter.tags import read_tag
from loopcutter.trackdb import load_tracks, save_tracks
from loopcutter.workspace import init_workspace

P128 = 60 / 128


def _ws(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    master = ws.masters / "Artist - Clicks.aiff"
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    sf.write(str(master), audio, sr, format="AIFF", subtype="PCM_24")
    save_tracks(ws.tracks_csv, {"Artist - Clicks": TrackRecord(
        track_id="Artist - Clicks", source=str(click_track["path"]), master=str(master),
        sample_rate=sr, duration=30.0, bpm=128.0, phase=0.0, bar_phase=0, bpm_fitted=128.0,
        inlier_ratio=1.0, bar_agreement=1.0, phase_agreement=1.0, key="8A")})
    monkeypatch.chdir(ws.root)
    return ws


def test_init_builds_the_scaffolding(tmp_path):
    assert main(["init", str(tmp_path / "ws")]) == 0
    assert (tmp_path / "ws" / "loops" / "aiff").is_dir()


def test_unresolved_rows_are_refused_by_cut(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.manifests / "s.csv").write_text("track_id,label,bars,start,snap\nArtist - Clicks,A1,4,15.03,beat\n")
    assert main(["cut", "manifests/s.csv"]) == 2
    assert "resolve" in capsys.readouterr().err


def test_resolve_snaps_fills_and_backs_up(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    path = ws.manifests / "s.csv"
    path.write_text("track_id,label,bars,start,snap\nArtist - Clicks,A1,4,15.03,beat\n")
    assert main(["resolve", "manifests/s.csv"]) == 0
    spec = load_manifest(path)[0]
    assert spec.start_seconds == pytest.approx(15.0) and spec.bpm == 128.0 and spec.key == "8A"
    assert "-30.0 ms" in capsys.readouterr().out
    assert any(ws.reports.glob("s.*.bak.csv"))


def test_resolve_refuses_far_starts_and_names_the_row(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.manifests / "s.csv").write_text("track_id,label,bars,start,snap\nArtist - Clicks,A1,4,15.2,beat\n")
    assert main(["resolve", "manifests/s.csv"]) == 1
    assert "row 2" in capsys.readouterr().err


def test_cut_files_tags_and_checks_in_a_workspace(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.manifests / "s.csv").write_text(
        "source,track_id,label,artist,track,bars,bpm,start,stem,key\n"
        f"{ws.masters / 'Artist - Clicks.aiff'},Artist - Clicks,A1,Artist,Clicks,4,128,15.0,bass,8A\n")
    assert main(["cut", "manifests/s.csv"]) == 0
    out = ws.loops / "aiff" / "bass" / "125-129" / "Artist - Clicks [A1][128][4bar][8A][bass].aiff"
    assert read_tag(out, "TBPM") == "128" and read_tag(out, "TIT1") == "bass"
    assert main(["cut", "manifests/s.csv", "--format", "wav"]) == 0
    wav = ws.loops / "wav" / "bass" / "125-129" / "Artist - Clicks [A1][128][4bar][8A][bass].wav"
    assert wav.exists() and read_tag(wav, "TBPM") is None                # WAV untagged by default


def test_lenient_turns_alignment_failures_into_warnings(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.manifests / "s.csv").write_text(
        f"source,label,bars,bpm,start\n{ws.masters / 'Artist - Clicks.aiff'},A1,4,128,15.04\n")
    assert main(["cut", "manifests/s.csv"]) == 1                    # 40 ms late
    assert main(["cut", "manifests/s.csv", "--lenient"]) == 0
    assert "warn beat_alignment" in capsys.readouterr().out


def test_same_window_two_stems_do_not_collide(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    rows = "".join(f"{click_track['path']},A1,4,128,15.0,{s}\n" for s in ("bass", "drums"))
    (ws.manifests / "s.csv").write_text("source,label,bars,bpm,start,stem\n" + rows)
    assert len(load_manifest(ws.manifests / "s.csv")) == 2


def test_live_note_and_honest_tiling_note(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.manifests / "s.csv").write_text(
        "source,label,bars,bpm,start,variations\n"
        f"{click_track['path']},A1,4,128,15.0,\"0.5\"\n"
        f"{click_track['path']},B1,4,135,15.0,\"0.5\"\n")
    assert main(["cut", "manifests/s.csv", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Live infers tempo only" in out
    assert "[A1][0.5bar]: drifts" in out and "Exact at 48000 Hz" in out
    assert "[B1][0.5bar]: drifts" not in out                # 135 BPM half bars tile exactly at 44.1 kHz
    assert "48 kHz sources avoid this" not in out


def test_import_from_serato_writes_resolved_rows(tmp_path, click_track, monkeypatch):
    from types import SimpleNamespace as NS

    import loopcutter.markers as markers

    ws = _ws(tmp_path, click_track, monkeypatch)
    monkeypatch.setattr(markers, "SERATO_SUFFIXES", {".wav", ".mp3", ".aiff"})
    monkeypatch.setattr(markers, "_serato_entries",
                        lambda p: [NS(position=int((32 * P128 + 0.020) * 1000), name="A1 drums")])
    assert main(["import", "--from", "serato", "--out", "manifests/imp.csv"]) == 0
    spec = load_manifest(ws.manifests / "imp.csv")[0]
    assert spec.start_seconds == pytest.approx(15.0) and spec.stem == "drums"


def test_prep_and_scan_commands(tmp_path, click_track, monkeypatch):
    import loopcutter.grid as grid

    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    beats = np.arange(64) * P128
    monkeypatch.setattr(grid, "detect", lambda m, s: (beats, beats[::4]))
    monkeypatch.setattr("loopcutter.keydetect.shutil.which", lambda name: None)
    assert main(["prep", str(click_track["path"])]) == 0
    assert main(["scan"]) == 0
    record = next(iter(load_tracks(ws.tracks_csv).values()))
    assert record.bpm == 128.0 and record.sample_rate == 48000
    assert any(ws.beats_dir.glob("*.npz"))


def test_prep_warns_when_a_master_was_built_from_another_source(tmp_path, click_track, monkeypatch, capsys):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    (tmp_path / "lossy").mkdir(); (tmp_path / "lossless").mkdir()
    sf.write(str(tmp_path / "lossy" / "Song.wav"), audio, sr)
    sf.write(str(tmp_path / "lossless" / "Song.flac"), audio, sr)          # a better copy, same name
    assert main(["prep", str(tmp_path / "lossy" / "Song.wav")]) == 0
    assert main(["prep", str(tmp_path / "lossless" / "Song.flac")]) == 0
    out = capsys.readouterr().out
    assert "CHANGED Song.aiff" in out and "lossy" in out
