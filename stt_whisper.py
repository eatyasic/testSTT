from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

VOICE_DIR = Path("data\\voice")
RESULT_DIR = Path("data\\result_whisper")
WHISPER_BIN = "whisper"
LANGUAGE = "Korean"
MODEL = "large"
OUTPUT_FORMAT = "txt"
START_ID = 17388        # adjust if needed
TARGET_COUNT = 20
EXTRA_ARGS: List[str] = []  # e.g. ["--device", "cuda"]


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


def run_whisper(target: Path) -> None:
    cmd = [
        WHISPER_BIN,
        str(target),
        "--language",
        LANGUAGE,
        "--model",
        MODEL,
        "--output_dir",
        str(RESULT_DIR),
        "--output_format",
        OUTPUT_FORMAT,
        *EXTRA_ARGS,
    ]
    print(f"--> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    if not VOICE_DIR.exists():
        raise SystemExit(f"Voice directory not found: {VOICE_DIR}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    targets = select_targets(START_ID, TARGET_COUNT)
    if not targets:
        raise SystemExit("No MP4 files matched the requested start ID.")

    for idx, clip in enumerate(targets, start=1):
        print(f"[{idx}\\{len(targets)}] Transcribing {clip.name}")
        run_whisper(clip)

    if len(targets) < TARGET_COUNT:
        print(
            f"Only {len(targets)} clips were available from MYR_{START_ID} upward; "
            "all available clips were transcribed."
        )


if __name__ == "__main__":
    main()
