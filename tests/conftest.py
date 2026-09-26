import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def click_track(tmp_path):
    """30 s of 128 BPM clicks over a quiet sine. Deterministic, loopable."""
    sr, bpm, seconds = 44100, 128.0, 30.0
    n = int(sr * seconds)
    t = np.arange(n) / sr

    audio = 0.05 * np.sin(2 * np.pi * 55.0 * t)
    samples_per_beat = sr * 60.0 / bpm
    for beat in range(int(seconds * bpm / 60)):
        start = int(round(beat * samples_per_beat))
        end = min(n, start + 400)
        env = np.linspace(1.0, 0.0, end - start)
        audio[start:end] += 0.6 * env * np.sin(
            2 * np.pi * 1000.0 * t[: end - start]
        )

    path = tmp_path / "click.wav"
    sf.write(str(path), np.column_stack([audio, audio]).astype("float32"), sr)
    return {"path": path, "sr": sr, "bpm": bpm}
