import subprocess

import pytest

from loopcutter.workspace import MARKER, WorkspaceError, find_workspace, init_workspace

DIRS = ("masters", "analysis", "manifests", "reports", "loops/aiff", "loops/wav", "stems")


def test_init_creates_scaffolding(tmp_path):
    ws = init_workspace(tmp_path / "ws")
    for sub in DIRS:
        assert (ws.root / sub).is_dir()
    assert (ws.root / MARKER).is_file() and "masters/" in (ws.root / ".gitignore").read_text()
    assert ws.setting("audio", "sample_rate") == 48000
    assert ws.tracks_csv == ws.root / "analysis" / "tracks.csv"
    assert ws.beats_dir == ws.root / "analysis" / "beats"


def test_init_is_idempotent_and_keeps_your_edits(tmp_path):
    ws = init_workspace(tmp_path)
    (ws.root / MARKER).write_text("[audio]\nsample_rate = 44100\n")
    (ws.root / ".gitignore").write_text("custom\n")
    again = init_workspace(tmp_path)
    assert again.setting("audio", "sample_rate") == 44100
    assert (again.root / ".gitignore").read_text() == "custom\n"


def test_init_refuses_inside_a_repo_with_a_remote(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "remote", "add", "origin", "https://example.com/x.git"], check=True)
    with pytest.raises(WorkspaceError, match="remote"):
        init_workspace(tmp_path / "ws")


def test_init_allows_a_repo_without_a_remote(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert init_workspace(tmp_path).root == tmp_path.resolve()


def test_find_walks_up_and_returns_none_outside(tmp_path):
    init_workspace(tmp_path / "ws")
    deep = tmp_path / "ws" / "manifests" / "nested"
    deep.mkdir(parents=True)
    assert find_workspace(deep).root == (tmp_path / "ws").resolve()
    assert find_workspace(tmp_path) is None
