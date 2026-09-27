"""Workspaces: private working data kept outside the tool's repository.

A workspace is any folder containing loopcutter.toml. Commands find it the way
git finds a repository, by walking up from the current directory. A workspace
may never sit inside a repository that has a remote: it holds private data.
"""

from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "loopcutter.toml"
DIRS = ("masters", "analysis", "manifests", "reports", "loops/aiff", "loops/wav", "stems")

DEFAULT_CONFIG = """\
# loopcutter workspace. Everything in this folder is private working data.

[audio]
sample_rate = 48000     # rate of the working masters; a cut never resamples
subtype = "PCM_24"
headroom_db = 3.0       # taken off every master, so overs from decoding and resampling don't clip

[sources]
paths = []              # folders or files that `loopcutter prep` reads by default

[prep]
replaced_dir = "masters/.replaced"   # where a lossy-built master goes when a lossless copy replaces it

[snap]
max_shift_ms = 60       # refuse to move a start further than this

[marking]
app = "rekordbox"       # rekordbox | rekordbox-xml | serato
playlist = ""           # the playlist that holds your masters
"""

GITIGNORE = ("masters/\nloops/\nstems/\nanalysis/beats/\n"
             "*.wav\n*.aif\n*.aiff\n*.flac\n*.mp3\n*.m4a\n.DS_Store\n")


class WorkspaceError(RuntimeError):
    """Raised when a workspace can't be created or found safely."""


@dataclass(frozen=True)
class Workspace:
    root: Path
    config: dict = field(default_factory=dict)

    @property
    def masters(self) -> Path:
        return self.root / "masters"

    @property
    def tracks_csv(self) -> Path:
        return self.root / "analysis" / "tracks.csv"

    @property
    def beats_dir(self) -> Path:
        return self.root / "analysis" / "beats"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def loops(self) -> Path:
        return self.root / "loops"

    @property
    def stems(self) -> Path:
        return self.root / "stems"

    def setting(self, section: str, key: str, default=None):
        return self.config.get(section, {}).get(key, default)


def load_workspace(root: str | Path) -> Workspace:
    root = Path(root).expanduser().resolve()
    with (root / MARKER).open("rb") as handle:
        return Workspace(root=root, config=tomllib.load(handle))


def find_workspace(start: str | Path | None = None) -> Workspace | None:
    here = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (here, *here.parents):
        if (candidate / MARKER).is_file():
            return load_workspace(candidate)
    return None


def _repo_with_remote(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            remotes = subprocess.run(["git", "-C", str(candidate), "remote"],
                                     capture_output=True, text=True).stdout.strip()
            return candidate if remotes else None
    return None


def init_workspace(root: str | Path) -> Workspace:
    root = Path(root).expanduser().resolve()
    repo = _repo_with_remote(root)
    if repo is not None:
        raise WorkspaceError(f"{root} is inside {repo}, a git repository with a remote; "
                             "a workspace holds private data, so put it somewhere else")
    for sub in DIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name, text in ((MARKER, DEFAULT_CONFIG), (".gitignore", GITIGNORE)):
        if not (root / name).exists():
            (root / name).write_text(text, encoding="utf-8")
    return load_workspace(root)
