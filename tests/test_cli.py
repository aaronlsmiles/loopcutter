import pathlib
import numpy as np
import pytest
import soundfile as sf

from loopcutter.cli import main
from loopcutter.manifest import ManifestError, load_manifest
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
    (ws.stems / "Artist - Clicks").mkdir(parents=True)                  # stem rows are cut from the stem
    (ws.stems / "Artist - Clicks" / "bass.aiff").write_bytes((ws.masters / "Artist - Clicks.aiff").read_bytes())
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
    assert not list(ws.loops.rglob("*.aiff"))                        # a failed loop never stays in the library
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


def _flag(ws, **kw):
    from dataclasses import replace

    records = load_tracks(ws.tracks_csv)
    records["Artist - Clicks"] = replace(records["Artist - Clicks"], **kw)
    save_tracks(ws.tracks_csv, records)


def test_cut_measures_the_tempo_of_a_track_whose_tempo_moves(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    _flag(ws, bpm=124.0, bpm_fitted=124.0, inlier_ratio=0.5, flags="grid-fit")   # a whole-track average
    (ws.manifests / "s.csv").write_text(
        "source,track_id,label,bars,bpm,start\n"
        f"{ws.masters / 'Artist - Clicks.aiff'},Artist - Clicks,A1,4,128,15.0\n")          # local tempo 128
    assert main(["cut", "manifests/s.csv"]) == 0


def test_resolve_reports_an_unanalysed_track_and_rows_it_cannot_read(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    _flag(ws, bpm=0.0, bpm_fitted=0.0)
    path = ws.manifests / "s.csv"
    path.write_text("track_id,label,bars,start,snap\nArtist - Clicks,A1,4,15.03,beat \n"
                    "Artist - Clicks,A2,4,,beat\n")
    assert main(["resolve", "manifests/s.csv"]) == 1
    err = capsys.readouterr().err
    assert "row 2" in err and "scan" in err and "row 3" in err


def test_resolve_never_truncates_a_manifest_with_a_stray_comma(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    path = ws.manifests / "s.csv"
    text = ("track_id,label,bars,start,snap,notes\nArtist - Clicks,A1,4,15.03,beat,fine\n"
            "Artist - Clicks,A2,4,15.03,beat,stray, comma\n")
    path.write_text(text)
    assert main(["resolve", "manifests/s.csv"]) == 2
    assert path.read_text() == text and "row 3" in capsys.readouterr().err


def test_resolve_strips_snap_and_refuses_an_unsure_phase(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    path = ws.manifests / "s.csv"
    path.write_text("track_id,label,bars,start,snap\nArtist - Clicks,A1,4,15.03, Beat \n")
    assert main(["resolve", "manifests/s.csv"]) == 0
    assert load_manifest(path)[0].start_seconds == pytest.approx(15.0)
    _flag(ws, flags="phase", phase_agreement=0.1)
    path.write_text("track_id,label,bars,start,snap\nArtist - Clicks,A1,4,15.03,beat\n")
    assert main(["resolve", "manifests/s.csv"]) == 1
    assert "confident" in capsys.readouterr().err


def test_prep_stores_absolute_source_paths(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    (tmp_path / "src").mkdir()
    sf.write(str(tmp_path / "src" / "Song.wav"), audio, sr)
    assert main(["prep", "../src/Song.wav"]) == 0
    record = load_tracks(ws.tracks_csv)["Song"]
    assert record.source == str((tmp_path / "src" / "Song.wav").resolve())


def test_cut_writes_oneshots_and_crossfaded_loops(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    master = ws.masters / "Artist - Clicks.aiff"
    (ws.manifests / "s.csv").write_text("source,label,kind,bars,bpm,start,end\n"
                                        f"{master},V1,oneshot,,,5.0,5.8\n"
                                        f"{master},A1,loop,4,128,15.0,\n")
    assert main(["cut", "manifests/s.csv", "--xfade-ms", "10"]) == 0
    one = ws.loops / "aiff" / "full" / "oneshots" / "Artist - Clicks [V1][oneshot].aiff"
    assert read_tag(one, "TBPM") is None and read_tag(one, "TIT2").endswith("[V1][oneshot]")
    assert (ws.loops / "aiff" / "full" / "125-129" / "Artist - Clicks [A1][128][4bar].aiff").exists()
    assert "Live infers tempo" not in capsys.readouterr().out


def test_stems_command_separates_and_conforms_each_track(tmp_path, click_track, monkeypatch, capsys):
    import loopcutter.stems as stems

    _ws(tmp_path, click_track, monkeypatch)
    calls = []
    monkeypatch.setattr(stems, "separate", lambda src, out, engine, model: calls.append((engine, model)) or {"bass": src})
    monkeypatch.setattr(stems, "conform_stems",
                        lambda raw, master, out: stems.ConformedStems({"bass": out / "bass.aiff"}, -0.12, {"bass": 3}))
    assert main(["stems", "Artist - Clicks"]) == 0
    out = capsys.readouterr().out
    assert calls == [("audio-separator", "htdemucs_6s.yaml")] and "bass" in out
    assert "-0.12 dB" in out and "3 sample(s) clipped" in out
    assert main(["stems", "Unknown"]) == 1


def test_stem_rows_are_cut_from_the_stem_and_timed_on_the_full_mix(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    master = ws.masters / "Artist - Clicks.aiff"
    (ws.manifests / "s.csv").write_text("source,track_id,label,bars,bpm,start,stem\n"
                                        f"{master},Artist - Clicks,A1,4,128,15.0,bass\n")
    assert main(["cut", "manifests/s.csv"]) == 1
    assert "loopcutter stems" in capsys.readouterr().err
    audio, sr = sf.read(str(master), dtype="float32", always_2d=True)
    (ws.stems / "Artist - Clicks").mkdir(parents=True)
    sf.write(str(ws.stems / "Artist - Clicks" / "bass.aiff"), audio * 0.5, sr, format="AIFF", subtype="PCM_24")
    assert main(["cut", "manifests/s.csv"]) == 0
    loop, _ = sf.read(str(ws.loops / "aiff" / "bass" / "125-129" / "Artist - Clicks [A1][128][4bar][bass].aiff"))
    assert np.max(np.abs(loop)) == pytest.approx(0.5 * np.max(np.abs(audio)), rel=0.05)


def test_a_loop_and_a_oneshot_at_one_spot_do_not_break_the_tiling_note(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    master = ws.masters / "Artist - Clicks.aiff"
    (ws.manifests / "s.csv").write_text("source,label,kind,bars,bpm,start,end\n"
                                        f"{master},A1,loop,4,128,15.0,\n{master},A1,oneshot,,,15.0,15.5\n")
    assert main(["cut", "manifests/s.csv", "--dry-run"]) == 0


def test_a_stem_row_without_a_track_is_cut_from_its_source_with_a_note(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.manifests / "s.csv").write_text("source,label,bars,bpm,start,stem\n"
                                        f"{ws.masters / 'Artist - Clicks.aiff'},A1,4,128,15.0,drums\n")
    assert main(["cut", "manifests/s.csv"]) == 0
    assert "cut from its source as given" in capsys.readouterr().out


def test_a_row_xfade_of_zero_beats_the_command_line(tmp_path, click_track, monkeypatch):
    from loopcutter.manifest import load_manifest as load

    ws = _ws(tmp_path, click_track, monkeypatch)
    path = ws.manifests / "s.csv"
    path.write_text(f"source,label,bars,bpm,start,xfade_ms\n{ws.masters / 'Artist - Clicks.aiff'},A1,4,128,15.0,0\n")
    assert load(path)[0].xfade_ms == 0.0
    path.write_text(f"source,label,bars,bpm,start,xfade_ms\n{ws.masters / 'Artist - Clicks.aiff'},A1,4,128,15.0,-5\n")
    with pytest.raises(ManifestError, match="xfade_ms"):
        load(path)


def test_resolve_refuses_a_downbeat_row_on_a_moving_tempo_and_leaves_complete_rows_alone(
        tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    _flag(ws, inlier_ratio=0.5, flags="grid-fit")
    path = ws.manifests / "s.csv"
    path.write_text("track_id,label,bars,downbeat,start_bar\nArtist - Clicks,A1,4,0.0,9\n")
    assert main(["resolve", "manifests/s.csv"]) == 1
    assert "give a start" in capsys.readouterr().err
    master = ws.masters / "Artist - Clicks.aiff"
    path.write_text(f"source,track_id,label,bars,bpm,start,key\n{master},Artist - Clicks,A1,4,128,15.0,8A\n")
    assert main(["resolve", "manifests/s.csv"]) == 0                      # no beat cache needed for a done row


def test_cut_refuses_a_double_time_tempo_on_a_moving_tempo_track(tmp_path, click_track, monkeypatch, capsys):
    ws = _ws(tmp_path, click_track, monkeypatch)
    _flag(ws, inlier_ratio=0.5, flags="grid-fit")
    (ws.manifests / "s.csv").write_text(
        "source,track_id,label,bars,bpm,start\n"
        f"{ws.masters / 'Artist - Clicks.aiff'},Artist - Clicks,A1,4,256,15.0\n")
    assert main(["cut", "manifests/s.csv"]) == 1
    assert "double" in capsys.readouterr().err


def test_a_long_loop_at_the_analysed_whole_tempo_passes_the_tempo_check(tmp_path, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    sr, n = 44100, 44100 * 70
    audio = np.zeros(n)
    for beat in range(int(70 * 128 / 60)):
        s = int(round(beat * sr * 60 / 128)); e = min(n, s + 400)
        audio[s:e] += 0.6 * np.linspace(1, 0, e - s) * np.sin(2 * np.pi * 1000 * np.arange(e - s) / sr)
    master = ws.masters / "Artist - Long.aiff"
    sf.write(str(master), np.column_stack([audio, audio]).astype("float32"), sr, format="AIFF", subtype="PCM_24")
    save_tracks(ws.tracks_csv, {"Artist - Long": TrackRecord(
        track_id="Artist - Long", source=str(master), master=str(master), sample_rate=sr, duration=70.0,
        bpm=128.0, bpm_fitted=128.009, phase=0.0, bar_phase=0, inlier_ratio=1.0, bar_agreement=1.0,
        phase_agreement=1.0)})                                     # scan rounded 128.009 to 128
    monkeypatch.chdir(ws.root)
    (ws.manifests / "s.csv").write_text(f"source,track_id,label,bars,bpm,start\n{master},Artist - Long,A1,16,128,15.0\n")
    assert main(["cut", "manifests/s.csv"]) == 0


def test_import_defaults_to_the_workspace_marking_app(tmp_path, click_track, monkeypatch):
    from types import SimpleNamespace as NS

    import loopcutter.markers as markers

    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.root / "loopcutter.toml").write_text((ws.root / "loopcutter.toml").read_text()
                                             .replace('app = "rekordbox"', 'app = "serato"'))
    monkeypatch.setattr(markers, "SERATO_SUFFIXES", {".wav", ".mp3", ".aiff"})
    monkeypatch.setattr(markers, "_serato_entries", lambda p: [NS(position=int((32 * P128 + 0.02) * 1000), name="A1")])
    assert main(["import", "--out", "manifests/imp.csv"]) == 0
    assert load_manifest(ws.manifests / "imp.csv")[0].start_seconds == pytest.approx(15.0)


def test_a_zero_shift_limit_means_zero(tmp_path, click_track, monkeypatch):
    ws = _ws(tmp_path, click_track, monkeypatch)
    (ws.manifests / "s.csv").write_text("track_id,label,bars,start,snap\nArtist - Clicks,A1,4,15.03,beat\n")
    assert main(["resolve", "manifests/s.csv", "--max-shift-ms", "0"]) == 1


def _two_copies(tmp_path, click_track, seconds=None, lossless_name="Song.wav"):
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    (tmp_path / "lossy").mkdir(exist_ok=True); (tmp_path / "lossless").mkdir(exist_ok=True)
    sf.write(str(tmp_path / "lossy" / "Song.mp3"), audio, sr, format="MP3", subtype="MPEG_LAYER_III")
    if seconds is None:                                        # the same track, 40 ms longer (encoder padding)
        other = np.concatenate([audio, np.zeros((int(0.04 * sr), audio.shape[1]), dtype=audio.dtype)])
    else:
        other = audio[: int(seconds * sr)]
    sf.write(str(tmp_path / "lossless" / lossless_name), other, sr)
    return tmp_path / "lossy" / "Song.mp3", tmp_path / "lossless" / lossless_name


def _tag_key(master, key="Am"):
    from mutagen.aiff import AIFF
    from mutagen.id3 import TKEY

    tagged = AIFF(str(master))
    if tagged.tags is None:
        tagged.add_tags()
    tagged.tags.add(TKEY(encoding=3, text=[key])); tagged.save(v2_version=3)


def test_prep_rebuilds_a_lossy_master_from_a_lossless_copy_and_keeps_what_is_yours(
        tmp_path, click_track, monkeypatch, capsys):
    from dataclasses import replace
    from mutagen.aiff import AIFF
    from mutagen.id3 import TKEY

    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track)
    assert main(["prep", str(mp3)]) == 0
    _tag_key(ws.masters / "Song.aiff")                                       # as a key app would
    records = load_tracks(ws.tracks_csv)
    save_tracks(ws.tracks_csv, {"Song": replace(records["Song"], bpm=128.0, override_bpm=127.5,
                                                override_phase_ms=4.0)})
    capsys.readouterr()
    assert main(["prep", str(wav.parent), str(mp3.parent)]) == 0
    out = capsys.readouterr().out
    assert "SKIP Song.mp3" in out and "REBUILT Song.aiff" in out and "1 master(s) written" in out
    assert "DJ app" in out
    record = load_tracks(ws.tracks_csv)["Song"]
    assert record.source.endswith("Song.wav") and record.bpm == 0.0 and record.override_bpm == 127.5
    assert record.override_phase_ms == 4.0 and "check override_phase_ms" in out
    assert read_tag(ws.masters / "Song.aiff", "TKEY") == "Am"                 # your key tag survives
    assert len(list((ws.masters / ".replaced").glob("Song.aiff_*.aiff"))) == 1   # the old master is kept
    assert not list(ws.masters.glob(".building*"))
    assert record.duration == pytest.approx(sf.info(str(ws.masters / "Song.aiff")).duration)


def test_prep_never_rebuilds_a_different_track_that_shares_a_name(tmp_path, click_track, monkeypatch, capsys):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track, seconds=10.0)              # same name, different length
    assert main(["prep", str(mp3)]) == 0
    before = (ws.masters / "Song.aiff").read_bytes()
    assert main(["prep", str(wav)]) == 0
    out = capsys.readouterr().out
    assert "REBUILT" not in out and "CHANGED Song.aiff" in out
    assert (ws.masters / "Song.aiff").read_bytes() == before


def test_prep_upgrades_the_row_when_the_lossy_master_is_gone(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track)
    assert main(["prep", str(mp3)]) == 0
    (ws.masters / "Song.aiff").unlink()
    assert main(["prep", str(wav)]) == 0
    assert load_tracks(ws.tracks_csv)["Song"].source.endswith("Song.wav")


def test_prep_keeps_a_lossless_master_when_a_lossy_copy_turns_up(tmp_path, click_track, monkeypatch, capsys):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track)
    assert main(["prep", str(wav)]) == 0
    assert main(["prep", str(mp3)]) == 0
    out = capsys.readouterr().out
    assert "KEPT Song.aiff" in out and "delete the master" not in out


def test_prep_refuses_a_negative_or_unreadable_headroom(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    config = (ws.root / "loopcutter.toml").read_text()
    for bad in ("-6.0", '"three"'):
        (ws.root / "loopcutter.toml").write_text(config.replace("headroom_db = 3.0", f"headroom_db = {bad}"))
        assert main(["prep", str(click_track["path"])]) == 2


def test_prep_takes_the_workspace_headroom_off_every_master(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    config = (ws.root / "loopcutter.toml").read_text()
    (ws.root / "loopcutter.toml").write_text(config.replace("headroom_db = 3.0", "headroom_db = 6.0"))
    assert main(["prep", str(click_track["path"])]) == 0
    source, _ = sf.read(str(click_track["path"]))
    master, _ = sf.read(str(ws.masters / "click.aiff"))
    assert np.max(np.abs(master)) == pytest.approx(0.5 * np.max(np.abs(source)), rel=0.02)


def test_a_rebuild_keeps_the_master_s_name_whatever_the_case_of_the_new_file(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track, lossless_name="song.wav")
    assert main(["prep", str(mp3)]) == 0
    assert main(["prep", str(wav)]) == 0
    assert [p.name for p in ws.masters.glob("*.aiff")] == ["Song.aiff"]     # rekordbox still finds it
    records = load_tracks(ws.tracks_csv)
    assert list(records) == ["Song"] and records["Song"].source.endswith("song.wav")


def test_a_rebuild_that_fails_leaves_the_old_master_and_its_tags(tmp_path, click_track, monkeypatch):
    import loopcutter.prep as prep

    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track)
    assert main(["prep", str(mp3)]) == 0
    _tag_key(ws.masters / "Song.aiff")

    def broken(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(prep, "prepare_master", broken)
    with pytest.raises(KeyboardInterrupt):
        main(["prep", str(wav)])
    assert read_tag(ws.masters / "Song.aiff", "TKEY") == "Am"
    assert not list((ws.masters / ".replaced").glob("*")) if (ws.masters / ".replaced").exists() else True


def test_a_different_track_is_never_built_under_the_old_row(tmp_path, click_track, monkeypatch, capsys):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track, seconds=10.0)
    assert main(["prep", str(mp3)]) == 0
    (ws.masters / "Song.aiff").unlink()
    assert main(["prep", str(wav)]) == 0
    assert "CHANGED" in capsys.readouterr().out and not (ws.masters / "Song.aiff").exists()


def test_kept_says_so_when_the_lossless_master_is_missing(tmp_path, click_track, monkeypatch, capsys):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track)
    assert main(["prep", str(wav)]) == 0
    (ws.masters / "Song.aiff").unlink()
    assert main(["prep", str(mp3)]) == 0
    assert "missing" in capsys.readouterr().out


def test_a_non_finite_headroom_is_refused(tmp_path, click_track, monkeypatch):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    config = (ws.root / "loopcutter.toml").read_text()
    for bad in ("nan", "inf"):
        (ws.root / "loopcutter.toml").write_text(config.replace("headroom_db = 3.0", f"headroom_db = {bad}"))
        assert main(["prep", str(click_track["path"])]) == 2


def test_prep_never_takes_the_workspace_s_own_masters_as_sources(tmp_path, click_track, monkeypatch, capsys):
    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track)
    assert main(["prep", str(mp3)]) == 0
    before = (ws.masters / "Song.aiff").read_bytes()
    assert main(["prep", str(ws.root)]) == 2                       # nothing but the workspace itself
    assert (ws.masters / "Song.aiff").read_bytes() == before
    assert load_tracks(ws.tracks_csv)["Song"].source.endswith("Song.mp3")


def test_a_rebuild_is_saved_even_if_a_later_track_stops_the_run(tmp_path, click_track, monkeypatch):
    import loopcutter.prep as prep

    ws = init_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    mp3, wav = _two_copies(tmp_path, click_track)
    assert main(["prep", str(mp3)]) == 0
    audio, sr = sf.read(str(click_track["path"]), dtype="float32", always_2d=True)
    sf.write(str(wav.parent / "Zed.wav"), audio, sr)
    real = prep.prepare_master

    def stop_on_zed(source, *args, **kwargs):
        if pathlib.Path(source).name == "Zed.wav":
            raise KeyboardInterrupt
        return real(source, *args, **kwargs)
    monkeypatch.setattr(prep, "prepare_master", stop_on_zed)
    with pytest.raises(KeyboardInterrupt):
        main(["prep", str(wav.parent)])
    record = load_tracks(ws.tracks_csv)["Song"]
    assert record.source.endswith("Song.wav") and record.bpm == 0.0
