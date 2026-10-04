"""Generate the staged demo with two public Microsoft neural voices."""
from __future__ import annotations
import asyncio, shutil, subprocess
from pathlib import Path
import edge_tts

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "scripts" / ".demo-audio-work"
OUTPUT = ROOT / "app" / "static" / "audio" / "warehouse-demo.mp3"
DIALOGUE = [
    ("Buyer", "en-US-AvaNeural", "My maximum budget is five thousand dollars."),
    ("Supplier", "en-US-AndrewNeural", "We can provide and install the equipment."),
    ("Supplier", "en-US-AndrewNeural", "So we're looking at approximately five thousand eight hundred dollars total."),
    ("Buyer", "en-US-AvaNeural", "Can you deliver next week?"),
    ("Supplier", "en-US-AndrewNeural", "We will take care of delivery."),
    ("Buyer", "en-US-AvaNeural", "I will prepare the loading area tomorrow."),
]

def ffmpeg_path() -> Path:
    found = shutil.which("ffmpeg")
    if found: return Path(found)
    candidates = list((Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"))
    if not candidates: raise RuntimeError("FFmpeg was not found")
    return candidates[0]

async def main() -> None:
    if WORK.exists(): shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    clips = []
    for index, (speaker, voice, text) in enumerate(DIALOGUE, 1):
        clip = WORK / f"{index:02d}-{speaker.lower()}.mp3"
        print(f"Generating turn {index} of {len(DIALOGUE)} for {speaker}...")
        await edge_tts.Communicate(text, voice, rate="-5%").save(str(clip))
        clips.append(clip)
    concat = WORK / "clips.txt"
    concat.write_text("".join(f"file '{clip.as_posix()}'\n" for clip in clips), encoding="utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ffmpeg_path()), "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c:a", "libmp3lame", "-q:a", "3", str(OUTPUT)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(WORK)
    print(f"Created {OUTPUT.relative_to(ROOT)}")

if __name__ == "__main__": asyncio.run(main())

