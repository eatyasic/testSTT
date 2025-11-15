from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

VOICE_DIR = Path("data\\voice")
RESULT_DIR = Path("data\\result_clova")
CONFIG_PATH = Path("clova_config.json")
LANGUAGE = "ko-KR"
CLOVA_COMPLETION = "sync"  # sync: wait for response immediately
START_ID = 17388
TARGET_COUNT = 20


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


def load_config(config_path: Path) -> Tuple[str, str, Dict[str, Any]]:
    if not config_path.exists():
        raise SystemExit(
            f"Clova config not found: {config_path}. "
            "Copy clova_config.example.json to clova_config.json and fill in your credentials."
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        invoke_url = data["invoke_url"].rstrip("/")
        secret = data["secret"]
    except KeyError as exc:  # pragma: no cover - configuration validation
        raise SystemExit(f"Missing field in config: {exc.args[0]}") from exc

    if not invoke_url or not secret:
        raise SystemExit("invoke_url and secret must be provided in clova_config.json")

    request_options = data.get("request_options") or {}
    if not isinstance(request_options, dict):
        raise SystemExit("request_options must be a JSON object if provided.")
    cleaned_options = {k: v for k, v in request_options.items() if v is not None}

    return invoke_url, secret, cleaned_options


class ClovaSpeechClient:
    def __init__(self, invoke_url: str, secret: str, default_params: Dict[str, Any] | None = None) -> None:
        self.invoke_url = invoke_url
        self.secret = secret
        self.default_params = dict(default_params) if default_params else {}

    def req_upload(
        self,
        file: Path,
        completion: str,
        callback: str | None = None,
        userdata: str | None = None,
        forbiddens: List[str] | None = None,
        boostings: List[Dict[str, Any]] | None = None,
        word_alignment: bool = True,
        full_text: bool = True,
        diarization: Dict[str, Any] | None = None,
        sed: Dict[str, Any] | None = None,
    ) -> requests.Response:
        request_body: Dict[str, Any] = {
            "language": LANGUAGE,
            "completion": completion,
            **self.default_params,
        }
        if "wordAlignment" not in request_body:
            request_body["wordAlignment"] = word_alignment
        if "fullText" not in request_body:
            request_body["fullText"] = full_text

        if callback is not None:
            request_body["callback"] = callback
        if userdata is not None:
            request_body["userdata"] = userdata
        if forbiddens:
            request_body["forbiddens"] = forbiddens
        if boostings:
            request_body["boostings"] = boostings
        if diarization is not None:
            request_body["diarization"] = diarization
        if sed is not None:
            request_body["sed"] = sed
        headers = {
            "Accept": "application/json;UTF-8",
            "X-CLOVASPEECH-API-KEY": self.secret,
        }
        url = f"{self.invoke_url}/recognizer/upload"
        print(f"--> POST {url} ({file.name})")
        with open(file, "rb") as media:
            files = {
                "media": media,
                "params": (
                    None,
                    json.dumps(request_body, ensure_ascii=False).encode("UTF-8"),
                    "application/json",
                ),
            }
            response = requests.post(headers=headers, url=url, files=files, timeout=300)
        return response


def _format_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        return json.dumps(payload, ensure_ascii=False)
    except ValueError:
        return response.text.strip() or response.reason


def extract_transcript(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()

    for key in ("text", "textData", "result", "transcript"):
        text = payload.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()

    segments = payload.get("segments")
    if isinstance(segments, list):
        lines = [
            segment.get("text", "")
            for segment in segments
            if isinstance(segment, dict) and segment.get("text")
        ]
        if lines:
            return "\n".join(lines).strip()

    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_clova(client: ClovaSpeechClient, clip: Path) -> None:
    response = client.req_upload(clip, completion=CLOVA_COMPLETION)
    if not response.ok:
        detail = _format_error(response)
        raise SystemExit(
            f"Clova Speech request failed for {clip.name} "
            f"({response.status_code} {response.reason}): {detail}"
        )
    transcript = extract_transcript(response)

    output_file = RESULT_DIR / f"{clip.stem}.txt"
    output_file.write_text(transcript, encoding="utf-8")
    print(f"Saved transcript -> {output_file}")


def main() -> None:
    if not VOICE_DIR.exists():
        raise SystemExit(f"Voice directory not found: {VOICE_DIR}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    invoke_url, secret, request_options = load_config(CONFIG_PATH)
    client = ClovaSpeechClient(invoke_url, secret, request_options)

    targets = select_targets(START_ID, TARGET_COUNT)
    if not targets:
        raise SystemExit("No MP4 files matched the requested start ID.")

    for idx, clip in enumerate(targets, start=1):
        print(f"[{idx}\\{len(targets)}] Uploading {clip.name}")
        run_clova(client, clip)

    if len(targets) < TARGET_COUNT:
        print(
            f"Only {len(targets)} clips were available from MYR_{START_ID} upward; "
            "all available clips were uploaded to Clova Speech."
        )


if __name__ == "__main__":
    main()
