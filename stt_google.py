from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from google.cloud import speech

VOICE_DIR = Path("data\\voice")
RESULT_DIR = Path("data\\result_google")
START_ID = 17388
TARGET_COUNT = 20
SAMPLE_RATE = 16000
LANGUAGE_CODE = "ko-KR"
FFMPEG_BIN = "ffmpeg"
ENABLE_PUNCTUATION = True
USE_LONG_RUNNING = True  # False -> synchronous recognize for very short clips


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


def extract_linear16_audio(src: Path, sample_rate: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / f"{src.stem}.wav"
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(tmp_wav),
        ]
        print(f"--> {' '.join(cmd)}")
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                f"ffmpeg executable not found: {FFMPEG_BIN}. "
                "Install ffmpeg and ensure it is on PATH."
            ) from exc

        # ✅ 임시 디렉터리가 삭제되기 전에 바로 읽어서 리턴
        return tmp_wav.read_bytes()



def run_google_speech(client: speech.SpeechClient, clip: Path) -> None:
    wav_content = extract_linear16_audio(clip, SAMPLE_RATE)
    audio = speech.RecognitionAudio(content=wav_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
        enable_automatic_punctuation=ENABLE_PUNCTUATION,
        max_alternatives=1,
    )

    if USE_LONG_RUNNING:
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=900)
    else:
        response = client.recognize(config=config, audio=audio)

    lines: List[str] = []
    for result in response.results:
        alternative = result.alternatives[0]
        if alternative.transcript.strip():
            lines.append(alternative.transcript.strip())

    transcript = "\n".join(lines).strip()
    if not transcript:
        transcript = "[No transcript returned]"

    output_file = RESULT_DIR / f"{clip.stem}.txt"
    output_file.write_text(transcript, encoding="utf-8")
    print(f"Saved transcript -> {output_file}")


def main() -> None:
    if not VOICE_DIR.exists():
        raise SystemExit(f"Voice directory not found: {VOICE_DIR}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    client = speech.SpeechClient()

    targets = select_targets(START_ID, TARGET_COUNT)
    if not targets:
        raise SystemExit("No MP4 files matched the requested start ID.")

    for idx, clip in enumerate(targets, start=1):
        print(f"[{idx}\\{len(targets)}] Transcribing {clip.name}")
        run_google_speech(client, clip)

    if len(targets) < TARGET_COUNT:
        print(
            f"Only {len(targets)} clips were available from MYR_{START_ID} upward; "
            "all available clips were transcribed."
        )


if __name__ == "__main__":
    main()
