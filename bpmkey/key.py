import numpy as np
import librosa

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _profile_correlation(chroma_vec: np.ndarray, profile: np.ndarray) -> float:
    a = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-9)
    b = profile / (np.linalg.norm(profile) + 1e-9)
    return float(np.dot(a, b))


def estimate_key(y: np.ndarray, sr: int):
    y_harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_vec = chroma.mean(axis=1)

    scores = []
    for shift in range(12):
        rotated = np.roll(chroma_vec, -shift)
        maj = _profile_correlation(rotated, KRUMHANSL_MAJOR)
        minr = _profile_correlation(rotated, KRUMHANSL_MINOR)
        scores.append((maj, "major", shift))
        scores.append((minr, "minor", shift))

    scores.sort(key=lambda x: x[0], reverse=True)
    best, second = scores[0], scores[1]

    key_name = f"{PITCH_CLASSES[best[2]]} {best[1]}"
    confidence = float((best[0] - second[0]) / (abs(best[0]) + 1e-9))

    return {"key": key_name, "confidence": confidence, "top2": [
        {"key": f"{PITCH_CLASSES[best[2]]} {best[1]}", "score": float(best[0])},
        {"key": f"{PITCH_CLASSES[second[2]]} {second[1]}", "score": float(second[0])},
    ]}
