"""P4.4 — THE RIVALS: OpenAI TTS / ElevenLabs / Hume Octave, same sentence, same judge.

Rules of the fight (mirrors the whole project):
  * SAME neutral sentence — semantics must not leak emotion; delivery only.
  * Each rival uses its own native emotion-control surface, as we did with
    IndexTTS-2 (emo_vector) and Chatterbox (reference+exaggeration):
      - OpenAI  gpt-4o-mini-tts : `instructions` prompt
      - ElevenLabs eleven_v3    : audio tags ([sad] ...) + one fixed premade voice
      - Hume Octave             : per-utterance acting `description`
  * One fixed voice per rival across all four emotions (timbre constant).
  * Raw API audio saved untouched to out/p44/raw/, then converted to 16 kHz mono
    WAV for the judge. Resumable: existing outputs are skipped.
  * Keys read from tts_steering/.keys.env at runtime; NEVER printed or persisted.

Run:  .venv_tts/bin/python tts_steering/rivals_p44.py
"""

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RAW = HERE / "out/p44/raw"
OUT = HERE / "out/p44"
RAW.mkdir(parents=True, exist_ok=True)

TEXT = "The table is in the room, and the door is closed."

KEYS = {}
for line in (HERE / ".keys.env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        KEYS[k.strip()] = v.strip()

STYLES = {  # one parallel acting instruction per emotion, phrased per-rival below
    "neutral": "completely neutral, flat, matter-of-fact",
    "sadness": "deeply sad, subdued, slow, heavy, on the verge of tears",
    "joy":     "genuinely happy, warm, bright, smiling while speaking, delighted",
    "anger":   "furious, harsh, clipped, seething with barely controlled rage",
}
V3_TAGS = {"neutral": "", "sadness": "[sad] ", "joy": "[happily] ", "anger": "[angry] "}
ELEVEN_VOICE = "SAz9YHcvj6GT2YYXdXww"  # River — Relaxed/Neutral/Informative, premade (free tier)


def synth_openai(emotion: str, raw_path: Path) -> None:
    r = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {KEYS['OPENAI_API_KEY']}"},
        json={"model": "gpt-4o-mini-tts", "voice": "alloy", "input": TEXT,
              "instructions": f"Speak in a {STYLES[emotion]} tone of voice.",
              "response_format": "wav"},
        timeout=120)
    r.raise_for_status()
    raw_path.write_bytes(r.content)


def synth_elevenlabs(emotion: str, raw_path: Path) -> None:
    body = {"text": V3_TAGS[emotion] + TEXT, "model_id": "eleven_v3"}
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
        "?output_format=mp3_44100_128",
        headers={"xi-api-key": KEYS["ELEVENLABS_API_KEY"]}, json=body, timeout=120)
    if r.status_code != 200:  # v3 may be gated on free tier -> fall back to v2
        print(f"    [eleven_v3 refused ({r.status_code}: {r.text[:120]}) "
              f"-> falling back to eleven_multilingual_v2 + style]", flush=True)
        body = {"text": TEXT, "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.3, "similarity_boost": 0.75,
                                   "style": 0.0 if emotion == "neutral" else 0.9}}
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
            "?output_format=mp3_44100_128",
            headers={"xi-api-key": KEYS["ELEVENLABS_API_KEY"]}, json=body,
            timeout=120)
        r.raise_for_status()
    raw_path.write_bytes(r.content)


def synth_hume(emotion: str, raw_path: Path) -> None:
    r = requests.post(
        "https://api.hume.ai/v0/tts",
        headers={"X-Hume-Api-Key": KEYS["HUME_API_KEY"]},
        json={"utterances": [{"text": TEXT,
                              "description": f"A speaker sounding {STYLES[emotion]}."}],
              "format": {"type": "wav"}, "num_generations": 1},
        timeout=180)
    r.raise_for_status()
    raw_path.write_bytes(base64.b64decode(r.json()["generations"][0]["audio"]))


RIVALS = {
    "openai-tts":  (synth_openai, "wav", "gpt-4o-mini-tts, voice=alloy, instructions"),
    "elevenlabs":  (synth_elevenlabs, "mp3", "eleven_v3 tags (v2+style fallback), voice=Rachel"),
    "hume-octave": (synth_hume, "wav", "octave acting description"),
}

manifest = []
for system, (fn, ext, control_desc) in RIVALS.items():
    for emotion in STYLES:
        name = f"{system}_{emotion}"
        raw_path = RAW / f"{name}.{ext}"
        wav_path = OUT / f"{name}.wav"
        manifest.append({"name": name, "system": system, "target": emotion,
                         "control": control_desc, "file": str(wav_path)})
        if wav_path.exists():
            print(f"  {name:26s} exists, skip", flush=True)
            continue
        try:
            if not raw_path.exists():
                t0 = time.time()
                fn(emotion, raw_path)
                print(f"  {name:26s} synthesized {time.time()-t0:.0f}s "
                      f"({raw_path.stat().st_size//1024} KB)", flush=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                            "-ar", "16000", "-ac", "1", str(wav_path)], check=True)
        except Exception as e:  # keep going — partial benchmark beats none
            print(f"  {name:26s} FAILED: {type(e).__name__}: {e}", flush=True)
            manifest[-1]["failed"] = True

ok = [m for m in manifest if not m.get("failed")]
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"\n{len(ok)}/{len(manifest)} clips ready -> {OUT}")
print("RIVALS_SYNTH_DONE")
