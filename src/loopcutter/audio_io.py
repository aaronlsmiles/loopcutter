"""Read any audio file as float32 frames x channels.

soundfile handles WAV, AIFF, FLAC and MP3 (gapless-correct). AAC, M4A and the
rest go through ffmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioReadError(RuntimeError):
    """Raised when neither soundfile nor ffmpeg can read a file."""


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return audio, int(sr)
    except sf.LibsndfileError:
        return _read_with_ffmpeg(path)


def _read_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise AudioReadError(f"{path.name}: soundfile can't read it and ffmpeg isn't installed")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise AudioReadError(f"{path.name}: no audio stream")
    sr, channels = int(streams[0]["sample_rate"]), int(streams[0]["channels"])
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(channels), "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, channels).copy(), sr
