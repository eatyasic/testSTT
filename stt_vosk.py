
from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Iterable, List, Tuple

import vosk  # pip install vosk

VOICE_DIR = Path("data\\voice")
RESULT_DIR = Path("data\\result_vosk")
MODEL_PATH = Path("models\\vosk-model-small-ko-0.22")  # change to your installed model
START_ID = 17388
TARGET_COUNT = 20
FFMPEG_BIN = "ffmpeg"
FFMPEG_ARGS = ["-ac", "1", "-ar", "16000"]  # mono, 16 kHz for Vosk


def parse_clip_id(path: Path) -> int | None: 
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def select_targets(start_id: int, limit: int) -> List[Path]:
    numbered: List[Tuple[int, Path]] = []
    for mp4 in VOICE_DIR.glob("MYR_*.mp4"):
        clip_id = parse_clip_id(mp4)
        if clip_id is None or clip_id < start_id:
            continue
        numbered.append((clip_id, mp4))
    numbered.sort(key=lambda item: item[0])
    return [path for _, path in numbered[:limit]]


def convert_to_wav(src: Path, dst: Path) -> None:
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(src),
        *FFMPEG_ARGS,
        str(dst),
    ]
    print(f"--> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe(audio_wav: Path, recognizer: vosk.Model) -> str:
    rec = vosk.KaldiRecognizer(recognizer, 16000)
    pieces: List[str] = []
    with wave.open(str(audio_wav), "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                pieces.append(json.loads(rec.Result()).get("text", ""))
    pieces.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(filter(None, pieces)).strip()


def run_vosk(clip: Path, model: vosk.Model) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / f"{clip.stem}.wav"
        convert_to_wav(clip, wav_path)
        text = transcribe(wav_path, model)

    output_file = RESULT_DIR / f"{clip.stem}.txt"
    output_file.write_text(text, encoding="utf-8")
    print(f"Saved transcript -> {output_file}")


def main() -> None:
    if not VOICE_DIR.exists():
        raise SystemExit(f"Voice directory not found: {VOICE_DIR}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Vosk model not found: {MODEL_PATH}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model = vosk.Model(str(MODEL_PATH))
    targets = select_targets(START_ID, TARGET_COUNT)
    if not targets:
        raise SystemExit("No MP4 files matched the requested start ID.")

    for idx, clip in enumerate(targets, start=1):
        print(f"[{idx}\\{len(targets)}] Transcribing {clip.name}")
        run_vosk(clip, model)

    if len(targets) < TARGET_COUNT:
        print(
            f"Only {len(targets)} clips were available from MYR_{START_ID} upward; "
            "all available clips were transcribed."
        )


if __name__ == "__main__":
    main()
