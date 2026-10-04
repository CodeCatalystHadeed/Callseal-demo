"""Generate the staged CallSeal warehouse demo with Groq Orpheus TTS."""
from __future__ import annotations
import json, shutil, struct, urllib.error, urllib.request, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
WORK = ROOT / "scripts" / ".demo-audio-work"
OUTPUT = ROOT / "app" / "static" / "audio" / "warehouse-demo.wav"
MODEL = "canopylabs/orpheus-v1-english"
DIALOGUE = [
    ("Buyer", "hannah", "[calm] My maximum budget is five thousand dollars."),
    ("Supplier", "troy", "[professional] We can provide and install the equipment."),
    ("Supplier", "troy", "[thoughtful] So we're looking at approximately five thousand eight hundred dollars total."),
    ("Buyer", "hannah", "[curious] Can you deliver next week?"),
    ("Supplier", "troy", "[reassuring] We will take care of delivery."),
    ("Buyer", "hannah", "[confident] I will prepare the loading area tomorrow."),
]

def load_key() -> str:
    for raw in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        if raw.strip().startswith("GROQ_API_KEY="):
            value = raw.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError("GROQ_API_KEY is missing from .env")

def generate_clip(api_key: str, voice: str, speech: str, destination: Path) -> None:
    body = json.dumps({"model": MODEL, "voice": voice, "input": speech, "response_format": "wav"}).encode()
    request = urllib.request.Request("https://api.groq.com/openai/v1/audio/speech", data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "CallSeal-Demo/1.0"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors="replace")
        raise RuntimeError(f"Groq TTS failed with HTTP {exc.code}: {message[:500]}") from exc

def combine_wav(clips: list[Path], output: Path) -> None:
    params = expected = None
    frames = []
    for clip in clips:
        with wave.open(str(clip), "rb") as source:
            current = source.getparams()
            signature = (current.nchannels, current.sampwidth, current.framerate, current.comptype)
            if params is None:
                params, expected = current, signature
            elif signature != expected:
                raise RuntimeError("Generated WAV clips use incompatible formats")
            frames.append(source.readframes(current.nframes))
            frames.append(struct.pack("<h", 0) * round(current.framerate * .55) * current.nchannels)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as target:
        target.setparams(params)
        for frame in frames:
            target.writeframes(frame)

def main() -> None:
    key = load_key()
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    clips = []
    for index, (speaker, voice, speech) in enumerate(DIALOGUE, 1):
        clip = WORK / f"{index:02d}-{speaker.lower()}.wav"
        print(f"Generating turn {index} of {len(DIALOGUE)} for {speaker}...")
        generate_clip(key, voice, speech, clip)
        clips.append(clip)
    combine_wav(clips, OUTPUT)
    shutil.rmtree(WORK)
    print(f"Created {OUTPUT.relative_to(ROOT)}")

if __name__ == "__main__": main()
