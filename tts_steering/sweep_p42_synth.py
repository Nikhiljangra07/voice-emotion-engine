"""P4.2 sweep — PHASE A: synthesize the slider-sweep clips (runs in VENDOR env).

Run:  cd tts_steering/vendor/index-tts && \
      PYTHONPATH=$PWD .venv/bin/python ../../sweep_p42_synth.py

Design (each clip one ledger row later):
  * sadness slider scaling: 0.4 / 1.2 (0.8 exists from the smoke test)
  * melancholic alone + sad+melancholic combo (which sad-adjacent slider works?)
  * joy and anger at 0.8 / 1.2
  * calm 0.8 and all-zeros baseline (controls)
  * prompt-emotion condition: same vectors with RAVDESS *sad*/*angry* speaker prompts
    (does prompt emotion leak into/help the output despite timbre-emotion disentangling?)

Resumable: existing outputs are skipped. Deterministic: use_random=False.
"""

import json
import time
from pathlib import Path

ROOT = Path("/Users/nikhil/Desktop/LoRa_WaveformEngine")
OUT = ROOT / "tts_steering/out/sweep_p42"
OUT.mkdir(parents=True, exist_ok=True)
TEXT = "The table is in the room, and the door is closed."

P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")   # neutral
P_SAD = str(ROOT / "data/ravdess/Actor_01/03-01-04-02-01-01-01.wav")   # sad, strong
P_ANG = str(ROOT / "data/ravdess/Actor_01/03-01-05-02-01-01-01.wav")   # angry, strong

def vec(**kw):
    """order: happy angry sad afraid disgusted melancholic surprised calm"""
    order = ["happy", "angry", "sad", "afraid", "disgusted",
             "melancholic", "surprised", "calm"]
    return [float(kw.get(k, 0.0)) for k in order]

#        name                target     vector                          prompt
SPECS = [
    ("baseline_zero",        "neutral", vec(),                          P_NEU),
    ("sad_04",               "sadness", vec(sad=0.4),                   P_NEU),
    ("sad_12",               "sadness", vec(sad=1.2),                   P_NEU),
    ("mel_08",               "sadness", vec(melancholic=0.8),           P_NEU),
    ("sad_mel_06_06",        "sadness", vec(sad=0.6, melancholic=0.6),  P_NEU),
    ("joy_08",               "joy",     vec(happy=0.8),                 P_NEU),
    ("joy_12",               "joy",     vec(happy=1.2),                 P_NEU),
    ("angry_08",             "anger",   vec(angry=0.8),                 P_NEU),
    ("angry_12",             "anger",   vec(angry=1.2),                 P_NEU),
    ("calm_08",              "neutral", vec(calm=0.8),                  P_NEU),
    ("sad_08_sadprompt",     "sadness", vec(sad=0.8),                   P_SAD),
    ("angry_08_angryprompt", "anger",   vec(angry=0.8),                 P_ANG),
]

manifest = []
todo = [(n, t, v, p) for n, t, v, p in SPECS
        if not (OUT / f"{n}.wav").exists()]
print(f"{len(SPECS)} specs, {len(todo)} to synthesize "
      f"({len(SPECS)-len(todo)} already exist)", flush=True)

tts = None
if todo:
    t0 = time.time()
    from indextts.infer_v2 import IndexTTS2
    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                    use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
    print(f"[model loaded in {time.time()-t0:.0f}s]", flush=True)

for name, target, v, prompt in SPECS:
    out_path = OUT / f"{name}.wav"
    if not out_path.exists():
        t1 = time.time()
        tts.infer(spk_audio_prompt=prompt, text=TEXT, output_path=str(out_path),
                  emo_vector=v, use_random=False, verbose=False)
        print(f"  {name:24s} synthesized in {time.time()-t1:.0f}s", flush=True)
    manifest.append({"name": name, "target": target, "vector": v,
                     "prompt": Path(prompt).name, "file": str(out_path)})

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"MANIFEST written: {len(manifest)} clips -> {OUT}/manifest.json", flush=True)
print("PHASE_A_DONE", flush=True)
