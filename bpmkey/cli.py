import os
import json
import click
from .audio import load_audio
from .bpm import estimate_bpm
from .key import estimate_key


@click.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--min-bpm", type=float, default=60.0, show_default=True)
@click.option("--max-bpm", type=float, default=200.0, show_default=True)
@click.option("--sr", type=int, default=22050, show_default=True, help="Target sampling rate")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
def main(audio_path: str, min_bpm: float, max_bpm: float, sr: int, json_output: bool):
    y, sr = load_audio(audio_path, sr=sr, mono=True)
    bpm_res = estimate_bpm(y, sr, min_bpm=min_bpm, max_bpm=max_bpm)
    key_res = estimate_key(y, sr)

    result = {
        "file": os.path.basename(audio_path),
        "bpm": float(bpm_res.get("bpm")) if bpm_res.get("bpm") == bpm_res.get("bpm") else None,
        "bpm_confidence": float(bpm_res.get("confidence", 0.0)),
        "key": key_res.get("key"),
        "key_confidence": float(key_res.get("confidence", 0.0)),
    }

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"File: {result['file']}")
        print(f"BPM: {result['bpm']} (conf {result['bpm_confidence']:.2f})")
        print(f"Key: {result['key']} (conf {result['key_confidence']:.2f})")

    return result


if __name__ == "__main__":
    main()
