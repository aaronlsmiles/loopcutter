import csv

import pytest
import soundfile as sf

from loopcutter.cut import cut_loop
from loopcutter.manifest import ManifestError, load_manifest
from loopcutter.verify import verify_cut


def _manifest(tmp_path, rows, fieldnames):
    path = tmp_path / "m.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_cut_and_verify_roundtrip(click_track, tmp_path):
    fields = ["source", "label", "bars", "bpm", "start", "artist", "track"]
    manifest = _manifest(
        tmp_path,
        [{
            "source": str(click_track["path"]),
            "label": "A1",
            "bars": "4",
            "bpm": str(click_track["bpm"]),
            "start": "5.15625",          # beat 11 at 128 BPM; loops must start on a beat
            "artist": "Test",
            "track": "Clicks",
        }],
        fields,
    )

    specs = load_manifest(manifest)
    assert len(specs) == 1

    result = cut_loop(specs[0], tmp_path / "out", fmt="wav")
    report = verify_cut(result)

    assert report.ok, [(c.name, c.detail) for c in report.failures]
    assert result.length_samples == 4 * 4 * 44100 * 60 / 128


def test_bar_addressing_matches_timecode(click_track, tmp_path):
    fields = ["source", "label", "bars", "bpm", "start", "downbeat", "start_bar"]
    by_time = _manifest(
        tmp_path,
        [{"source": str(click_track["path"]), "label": "t", "bars": "2",
          "bpm": "128", "start": "7.5", "downbeat": "", "start_bar": ""}],
        fields,
    )
    # 128 BPM: one bar is 1.875 s. Downbeat 0, bar 5 -> 7.5 s.
    by_bar = tmp_path / "bar.csv"
    by_bar.write_text(
        "source,label,bars,bpm,start,downbeat,start_bar\n"
        f"{click_track['path']},b,2,128,,0.0,5\n",
        encoding="utf-8",
    )

    a = load_manifest(by_time)[0]
    b = load_manifest(by_bar)[0]
    assert a.start_seconds == pytest.approx(b.start_seconds)


def test_manifest_rejects_double_start(click_track, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "source,label,bars,bpm,start,downbeat,start_bar\n"
        f"{click_track['path']},x,4,128,5.0,0.0,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="not both"):
        load_manifest(path)


def test_manifest_rejects_colliding_names(click_track, tmp_path):
    path = tmp_path / "dupe.csv"
    row = f"{click_track['path']},A1,4,128,5.0,Artist,Track\n"
    path.write_text(
        "source,label,bars,bpm,start,artist,track\n" + row + row, encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="collide"):
        load_manifest(path)


def test_overrun_is_an_error_not_a_short_file(click_track, tmp_path):
    path = tmp_path / "over.csv"
    path.write_text(
        "source,label,bars,bpm,start\n"
        f"{click_track['path']},x,8,128,29.0\n",
        encoding="utf-8",
    )
    spec = load_manifest(path)[0]
    with pytest.raises(ValueError, match="too short"):
        cut_loop(spec, tmp_path / "out", fmt="wav")


def test_output_is_exactly_the_declared_length(click_track, tmp_path):
    path = tmp_path / "len.csv"
    path.write_text(
        "source,label,bars,bpm,start\n"
        f"{click_track['path']},x,4,127,5.0\n",
        encoding="utf-8",
    )
    result = cut_loop(load_manifest(path)[0], tmp_path / "out", fmt="wav")
    info = sf.info(str(result.output))
    assert info.frames == round(16 * 44100 * 60 / 127)


def test_variations_expand_one_row_into_many(click_track, tmp_path):
    path = tmp_path / "var.csv"
    path.write_text(
        "source,label,artist,track,bars,bpm,start,variations\n"
        f"{click_track['path']},A1,Test,Clicks,4,128,5.0,\"2,1,0.5\"\n",
        encoding="utf-8",
    )
    specs = load_manifest(path)
    assert [s.bars for s in specs] == [4.0, 2.0, 1.0, 0.5]
    assert len({s.start_seconds for s in specs}) == 1
    assert len({s.slug for s in specs}) == 4


def test_variations_deduplicate_the_base_length(click_track, tmp_path):
    path = tmp_path / "dupe-var.csv"
    path.write_text(
        "source,label,bars,bpm,start,variations\n"
        f"{click_track['path']},A1,4,128,5.0,\"4,2\"\n",
        encoding="utf-8",
    )
    assert [s.bars for s in load_manifest(path)] == [4.0, 2.0]


def test_variations_tile_within_a_sample(click_track, tmp_path):
    """Variations must tile, allowing for unavoidable integer rounding.

    At 128 BPM / 44.1 kHz a bar is 82687.5 samples, so a 1-bar loop cannot
    divide a 4-bar one exactly. One sample is the most that rounding can cost.
    """
    path = tmp_path / "halves.csv"
    path.write_text(
        "source,label,bars,bpm,start,variations\n"
        f"{click_track['path']},A1,4,128,5.0,\"2,1\"\n",
        encoding="utf-8",
    )
    results = [cut_loop(s, tmp_path / "out", fmt="wav") for s in load_manifest(path)]
    four, two, one = (r.length_samples for r in results)
    assert four == two * 2
    assert abs(one * 4 - four) <= 2


def test_tiling_is_exact_at_48k():
    """48 kHz is the reason to prefer it: 128 BPM gives whole samples per bar."""
    from loopcutter.timing import loop_length_samples, tiling_error_samples

    assert loop_length_samples(1, 128, 48000) == 90000
    assert tiling_error_samples(4, 1, 128, 48000) == 0
    assert tiling_error_samples(4, 0.5, 128, 48000) == 0
    assert tiling_error_samples(4, 1, 128, 44100) > 0


def test_variations_reject_nonsense(click_track, tmp_path):
    path = tmp_path / "bad-var.csv"
    path.write_text(
        "source,label,bars,bpm,start,variations\n"
        f"{click_track['path']},A1,4,128,5.0,\"2,banana\"\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="bar numbers"):
        load_manifest(path)


def test_find_zero_crossing_direction_picks_which_side_wins():
    """test_verify.py's roundtrip test can't tell which way the search looks, because
    15.003 s already sits on a crossing. Pin the direction directly: with a crossing on
    each side of centre, "backward" must return the earlier one and "both" the later one.
    """
    import numpy as np

    from loopcutter.cut import _find_zero_crossing

    centre = 10
    mono = np.ones(21)
    mono[centre - 5] = -1.0    # sign change lands the backward crossing at centre - 4
    mono[centre + 2] = -1.0    # sign change lands the forward crossing at centre + 2

    assert _find_zero_crossing(mono, centre, radius=5, direction="backward") == centre - 4
    assert _find_zero_crossing(mono, centre, radius=5, direction="both") == centre + 2
