import librosa
import numpy as np
from typing import Optional


def load_audio(path: str, sr: int = 22050, mono: bool = True, offset: float = 0.0, duration: Optional[float] = None):
    y, sr = librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=0)
    return y, sr
