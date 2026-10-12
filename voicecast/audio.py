"""Audio helpers: trim silence, join pieces, level, write files atomically."""
import os
from pathlib import Path

import numpy as np


def trim(wave, rate, floor=0.04):
    """Drop silence / low noise at the start and end."""
    frame = max(rate // 100, 1)
    n = len(wave) // frame
    if n < 3:
        return wave
    rms = np.sqrt((wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.where(rms > max(rms.max() * floor, 1e-4))[0]
    if not len(loud):
        return wave[:0]
    return wave[max(loud[0] - 5, 0) * frame: min(loud[-1] + 8, n) * frame]


def join(waves, rate, gap=0.25):
    pad = np.zeros(int(gap * rate), dtype=np.float32)
    parts = []
    for i, w in enumerate(waves):
        if i:
            parts.append(pad)
        parts.append(np.asarray(w, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def level(wave, target_rms):
    """Scale to a target RMS (0 = unchanged), never past full scale."""
    if not target_rms or not len(wave):
        return wave
    rms = float(np.sqrt(np.mean(wave ** 2)))
    if rms < 1e-6:
        return wave
    out = wave * (target_rms / rms)
    peak = float(np.abs(out).max())
    return out / peak * 0.98 if peak > 0.98 else out


def write(path, wave, rate, fmt="ogg"):
    """Write mono audio; the file appears only when complete. OGG is written in
    small blocks (libsndfile's Vorbis encoder can overflow on long clips)."""
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    sub = {"ogg": ("OGG", "VORBIS"), "wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16")}[fmt]
    with sf.SoundFile(tmp, "w", samplerate=rate, channels=1, format=sub[0], subtype=sub[1]) as f:
        wave = np.asarray(wave, dtype=np.float32)
        for i in range(0, len(wave), 8192):
            f.write(wave[i:i + 8192])
    os.replace(tmp, path)
