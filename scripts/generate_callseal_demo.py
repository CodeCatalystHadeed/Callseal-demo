"""Generate the full staged Ali/Sarah CallSeal demo with Groq Orpheus TTS."""
from __future__ import annotations

import os
import json
import re
import shutil
import struct
import subprocess
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp" / "callseal_demo"
OUTPUT = ROOT / "app" / "static" / "audio" / "callseal_demo.wav"
TIMING_OUTPUT = ROOT / "app" / "static" / "audio" / "callseal_demo_timing.json"
TRANSCRIPT_OUTPUT = ROOT / "app" / "static" / "audio" / "callseal_demo_transcript.txt"
MODEL = "canopylabs/orpheus-v1-english"
INPUT_LIMIT = 200
MAX_ATTEMPTS = 4

conversation = [
    {"speaker": "Ali", "voice": "austin", "text": "Hi Sarah, it's Ali from Northside Warehouse. Have you got a few minutes to talk about that loading equipment we discussed?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "Hi Ali. Yeah, absolutely. I actually got the updated numbers from our team this morning."},
    {"speaker": "Ali", "voice": "austin", "text": "Great. So where are we sitting price-wise?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "For the unit we discussed, including installation, we're looking at approximately fifty-eight hundred dollars."},
    {"speaker": "Ali", "voice": "austin", "text": "Fifty-eight hundred? That's a little higher than I expected. I mentioned before that five thousand is really the maximum budget I've been given."},
    {"speaker": "Sarah", "voice": "autumn", "text": "Right, I remember. The problem is the installation crew adds quite a bit to the cost. The equipment itself isn't too far off your range."},
    {"speaker": "Ali", "voice": "austin", "text": "But the fifty-eight hundred includes the installation, right? I don't want to find out afterward that installation is another thousand dollars."},
    {"speaker": "Sarah", "voice": "autumn", "text": "Yes, installation is included in that figure."},
    {"speaker": "Ali", "voice": "austin", "text": "Okay. I'll have to see what I can do about the extra eight hundred. What about getting the equipment here? Is delivery included as well?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "We can arrange the delivery. Usually we coordinate that once we know the exact date and what's required at the site."},
    {"speaker": "Ali", "voice": "austin", "text": "Okay, but is that part of the fifty-eight hundred?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "It depends a little on the truck and access requirements. Let me check what we can do there."},
    {"speaker": "Ali", "voice": "austin", "text": "All right. Our loading entrance is pretty straightforward. I'll make sure the loading area is cleared and prepared by Thursday, so your team won't have anything blocking them."},
    {"speaker": "Sarah", "voice": "autumn", "text": "Perfect. I'll arrange the installation crew from our side."},
    {"speaker": "Ali", "voice": "austin", "text": "Great. And we're still thinking everything can be finished by next Friday?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "I think next Friday should be possible. I just need to confirm the equipment availability and the crew schedule."},
    {"speaker": "Ali", "voice": "austin", "text": "Okay. Friday would be ideal because we have shipments starting again the following Monday."},
    {"speaker": "Sarah", "voice": "autumn", "text": "Understood. I'll try to keep it around that timeline."},
    {"speaker": "Ali", "voice": "austin", "text": "Good. So assuming we can make the numbers work, we're basically looking at the equipment, installation, and hopefully completion next Friday."},
    {"speaker": "Sarah", "voice": "autumn", "text": "That's the idea, yes."},
    {"speaker": "Ali", "voice": "austin", "text": "What do you need from me to get things moving? Do you need the payment upfront?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "We normally take something before scheduling, but let me check with accounts because this order is slightly different."},
    {"speaker": "Ali", "voice": "austin", "text": "Is that fifty percent, or the whole amount?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "I don't want to give you the wrong figure. I'll confirm the payment schedule when I send the updated quote."},
    {"speaker": "Ali", "voice": "austin", "text": "Fair enough. Can you send me that today?"},
    {"speaker": "Sarah", "voice": "autumn", "text": "Yes. I'll update the quote with the installation included and check the delivery and payment details."},
    {"speaker": "Ali", "voice": "austin", "text": "Perfect. And I'll have the loading area ready by Thursday."},
    {"speaker": "Sarah", "voice": "autumn", "text": "Great. I'll work on getting the crew arranged and confirm whether Friday works."},
    {"speaker": "Ali", "voice": "austin", "text": "Sounds good. Thanks, Sarah."},
    {"speaker": "Sarah", "voice": "autumn", "text": "No problem, Ali. I'll be in touch."},
]

def split_naturally(text: str, limit: int = INPUT_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    clauses = re.split(r"(?<=[.!?;,])\s+", text)
    chunks, current = [], ""
    for clause in clauses:
        candidate = f"{current} {clause}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = clause
        while len(current) > limit:
            cut = current.rfind(" ", 0, limit + 1)
            cut = cut if cut > 0 else limit
            chunks.append(current[:cut].strip())
            current = current[cut:].strip()
    if current:
        chunks.append(current)
    assert " ".join(chunks).replace("  ", " ") == text.replace("  ", " ")
    return chunks

def valid_wav(path: Path) -> bool:
    try:
        if path.stat().st_size < 1024:
            return False
        with wave.open(str(path), "rb") as source:
            return source.getnframes() > 0 and source.getframerate() > 0
    except (OSError, wave.Error):
        return False

def ffmpeg_path() -> Path:
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    matches = list((Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"))
    if not matches:
        raise RuntimeError("FFmpeg is required to combine Groq streaming WAV files")
    return matches[0]

def generate_piece(client: Groq, voice: str, text: str, destination: Path) -> None:
    if valid_wav(destination):
        return
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.audio.speech.create(model=MODEL, voice=voice, input=text, response_format="wav")
            response.write_to_file(destination)
            if not valid_wav(destination):
                raise RuntimeError("Provider returned an invalid or empty WAV file")
            return
        except (RateLimitError, APIConnectionError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Groq TTS failed after {MAX_ATTEMPTS} attempts: {exc}") from exc
            delay = 2 ** (attempt - 1)
            print(f"  Transient error; retrying in {delay}s...")
            time.sleep(delay)
        except APIStatusError as exc:
            if exc.status_code >= 500 and attempt < MAX_ATTEMPTS:
                delay = 2 ** (attempt - 1)
                print(f"  Groq HTTP {exc.status_code}; retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Groq TTS request failed with HTTP {exc.status_code}: {exc.message}") from exc

def media_duration(path: Path) -> float:
    ffprobe = ffmpeg_path().with_name("ffprobe.exe")
    result = subprocess.run([str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=True, capture_output=True, text=True)
    return float(result.stdout.strip())

def combine(pieces: list[dict]) -> dict:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path()
    pauses = {160: TMP / "pause_160.wav", 450: TMP / "pause_450.wav", 550: TMP / "pause_550.wav"}
    for milliseconds, path in pauses.items():
        if not path.exists():
            subprocess.run([str(ffmpeg), "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{milliseconds / 1000:.3f}", "-c:a", "pcm_s16le", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    concat_file = TMP / "concat.txt"
    entries = []
    for item in pieces:
        piece, is_question, same_turn_continues = item["path"], item["is_question"], item["continues"]
        pause_ms = 160 if same_turn_continues else (550 if is_question else 450)
        entries.extend((f"file '{piece.as_posix()}'", f"file '{pauses[pause_ms].as_posix()}'"))
    concat_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
    subprocess.run([str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(OUTPUT)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(str(OUTPUT), "rb") as result:
        return {"duration": result.getnframes() / result.getframerate(), "sample_rate": result.getframerate(), "channels": result.getnchannels(), "sample_width": result.getsampwidth(), "frames": result.getnframes(), "size": OUTPUT.stat().st_size}

def write_synchronization(pieces: list[dict]) -> None:
    clock = 0.0
    turns = []
    for item in pieces:
        duration = media_duration(item["path"])
        if not turns or turns[-1]["turn"] != item["turn"]:
            turns.append({"turn": item["turn"], "speaker": item["speaker"], "text": item["full_text"], "start_time": round(clock, 3), "end_time": None})
        clock += duration
        turns[-1]["end_time"] = round(clock, 3)
        pause_ms = 160 if item["continues"] else (550 if item["is_question"] else 450)
        clock += pause_ms / 1000
    TIMING_OUTPUT.write_text(json.dumps({"audio": "/static/audio/callseal_demo.wav", "duration": round(clock, 3), "turns": turns}, indent=2), encoding="utf-8")
    TRANSCRIPT_OUTPUT.write_text("\n".join(f"{int(t['start_time']) // 60:02d}:{int(t['start_time']) % 60:02d} {t['speaker']}: {t['text']}" for t in turns) + "\n", encoding="utf-8")

def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is missing. Add it to D:\\CallSeal\\.env and rerun.")
    client = Groq(api_key=api_key)
    TMP.mkdir(parents=True, exist_ok=True)
    tasks = []
    total = sum(len(split_naturally(turn["text"])) for turn in conversation)
    counter = 0
    for turn_index, turn in enumerate(conversation, 1):
        chunks = split_naturally(turn["text"])
        for chunk_index, chunk in enumerate(chunks, 1):
            counter += 1
            suffix = f"_{chunk_index:02d}" if len(chunks) > 1 else ""
            path = TMP / f"{turn_index:03d}_{turn['speaker'].lower()}{suffix}.wav"
            state = "Reusing" if valid_wav(path) else "Generating"
            print(f"[{counter}/{total}] {state} {turn['speaker']}...")
            generate_piece(client, turn["voice"], chunk, path)
            tasks.append({"path": path, "is_question": chunk.rstrip().endswith("?"), "continues": chunk_index < len(chunks), "turn": turn_index, "speaker": turn["speaker"], "full_text": turn["text"]})
    print("Combining audio...")
    metadata = combine(tasks)
    write_synchronization(tasks)
    if metadata["duration"] < 90:
        raise RuntimeError(f"Generated recording is unexpectedly short: {metadata['duration']:.1f}s")
    shutil.rmtree(TMP)
    print(f"Created {OUTPUT}")
    print(f"Duration: {metadata['duration']:.2f}s")
    print(f"Sample rate: {metadata['sample_rate']} Hz")
    print(f"Channels: {metadata['channels']}")
    print(f"Sample width: {metadata['sample_width'] * 8}-bit")
    print(f"File size: {metadata['size']} bytes")

if __name__ == "__main__":
    main()
