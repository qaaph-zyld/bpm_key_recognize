import numpy as np
import librosa


def _octave_correction(bpm: float, min_bpm: float, max_bpm: float) -> float:
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0
    return bpm


def estimate_bpm(y: np.ndarray, sr: int, min_bpm: float = 60.0, max_bpm: float = 200.0, hop_length: int = 512):
    y_harm, y_perc = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length, aggregate=np.mean)
    if onset_env.size < 4:
        tempo = float(librosa.beat.tempo(sr=sr, onset_envelope=onset_env, hop_length=hop_length))
        tempo = _octave_correction(float(tempo), min_bpm, max_bpm)
        return {"bpm": float(tempo), "confidence": 0.0, "candidates": [float(tempo)]}

    ac = librosa.autocorrelate(onset_env, max_size=onset_env.size // 2)
    ac[0] = 0.0

    peaks = librosa.util.peak_pick(ac, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=float(np.mean(ac) * 0.1), wait=3)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(ac))])

    lags = peaks
    lags = lags[lags > 0]
    if lags.size == 0:
        tempo = float(librosa.beat.tempo(sr=sr, onset_envelope=onset_env, hop_length=hop_length))
        tempo = _octave_correction(float(tempo), min_bpm, max_bpm)
        return {"bpm": float(tempo), "confidence": 0.0, "candidates": [float(tempo)]}

    tempos = 60.0 * sr / (lags * hop_length)
    strengths = ac[peaks[:len(lags)]]

    corrected = np.array([_octave_correction(float(t), min_bpm, max_bpm) for t in tempos])
    order = np.argsort(strengths)[::-1]
    corrected = corrected[order]
    strengths = strengths[order]

    if corrected.size > 1:
        conf = float(strengths[0] / (strengths[1] + 1e-9))
    else:
        conf = 1.0

    bpm = float(corrected[0])
    candidates = [float(x) for x in corrected[:5]]
    return {"bpm": bpm, "confidence": conf, "candidates": candidates}
