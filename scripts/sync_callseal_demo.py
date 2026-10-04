"""Align the canonical staged script to its generated WAV using Groq Whisper words."""
from __future__ import annotations
import json, os, re
from difflib import SequenceMatcher
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from generate_callseal_demo import ROOT, OUTPUT, TIMING_OUTPUT, TRANSCRIPT_OUTPUT, conversation

def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))

def value(item, name):
    return getattr(item, name) if hasattr(item, name) else item[name]

def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("GROQ_API_KEY")
    if not key: raise SystemExit("GROQ_API_KEY is missing from .env")
    with OUTPUT.open("rb") as audio:
        result = Groq(api_key=key).audio.transcriptions.create(file=audio, model="whisper-large-v3-turbo", language="en", temperature=0, response_format="verbose_json", timestamp_granularities=["word", "segment"])
    words = [{"word": value(w, "word"), "start": float(value(w, "start")), "end": float(value(w, "end"))} for w in result.words]
    observed = [normalize(w["word"])[0] if normalize(w["word"]) else "" for w in words]
    cursor, turns, scores = 0, [], []
    for index, turn in enumerate(conversation, 1):
        expected = normalize(turn["text"])
        best = None
        for start in range(cursor, min(cursor + 16, len(words))):
            for size in range(max(1, len(expected) - 4), len(expected) + 5):
                end = min(start + size, len(words))
                score = SequenceMatcher(None, expected, observed[start:end]).ratio()
                if best is None or score > best[0]: best = (score, start, end)
        if not best or best[0] < .55:
            raise RuntimeError(f"Could not confidently align turn {index} ({turn['speaker']}); score={best[0] if best else 0:.2f}")
        score, start, end = best
        scores.append(score); cursor = end
        turns.append({"turn": index, "speaker": turn["speaker"], "text": turn["text"], "start_time": round(words[start]["start"], 3), "end_time": round(words[end - 1]["end"], 3), "alignment_score": round(score, 3)})
    if len(turns) != 30 or sum(scores) / len(scores) < .8:
        raise RuntimeError("Transcript alignment did not meet the required quality threshold")
    duration = turns[-1]["end_time"]
    TIMING_OUTPUT.write_text(json.dumps({"audio": "/static/audio/callseal_demo.wav", "duration": duration, "alignment": "Groq Whisper word timestamps aligned to canonical script", "mean_alignment_score": round(sum(scores) / len(scores), 3), "turns": turns}, indent=2), encoding="utf-8")
    TRANSCRIPT_OUTPUT.write_text("\n".join(f"{int(t['start_time']) // 60:02d}:{int(t['start_time']) % 60:02d} {t['speaker']}: {t['text']}" for t in turns) + "\n", encoding="utf-8")
    print(f"Aligned {len(turns)} turns; mean score {sum(scores) / len(scores):.3f}; final speech ends at {duration:.2f}s")
    print(f"Created {TIMING_OUTPUT}")
    print(f"Created {TRANSCRIPT_OUTPUT}")

if __name__ == "__main__": main()

