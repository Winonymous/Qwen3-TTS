# coding=utf-8
"""Loudness helpers shared by data preparation and inference.

The model learns the loudness of the clips it is trained on, so a training set with quiet and loud clips gives quiet
and loud speech. `normalize_loudness` brings every clip to the same speech level.
"""
import numpy as np


def speech_rms_db(y, sr, frame_ms=25, gate_db=40.0):
    """RMS level in dBFS of the active speech, or None for a silent clip.

    Only frames within `gate_db` of the loudest frame count, so pauses and background hiss do not pull the level
    down (a clip with long pauses is not treated as quiet).
    """
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = len(y) // frame
    if n_frames == 0:
        rms = np.array([np.sqrt(np.mean(np.square(y, dtype=np.float64)))]) if len(y) else np.zeros(1)
    else:
        frames = y[: n_frames * frame].astype(np.float64).reshape(n_frames, frame)
        rms = np.sqrt((frames ** 2).mean(axis=1))
    loudest = rms.max()
    if loudest <= 1e-8:
        return None
    active = rms[rms >= loudest * 10 ** (-gate_db / 20)]
    return float(20 * np.log10(np.sqrt((active ** 2).mean()) + 1e-12))


def normalize_loudness(y, sr, target_db=-23.0, peak_limit=0.99, max_gain_db=25.0):
    """Scale `y` so that its speech level is `target_db` dBFS.

    The gain is limited to +-`max_gain_db` (a near-silent clip is not blown up into noise), and the clip is scaled
    down if it would peak above `peak_limit`, so a very peaky clip can end up a little quieter than the target.
    """
    level = speech_rms_db(y, sr)
    if level is None:
        return y
    gain_db = float(np.clip(target_db - level, -max_gain_db, max_gain_db))
    out = y.astype(np.float32) * np.float32(10 ** (gain_db / 20))
    peak = float(np.abs(out).max())
    if peak > peak_limit:
        out = out * np.float32(peak_limit / peak)
    return out.astype(np.float32)
