"""Chatterbox cross-model probe — the discriminating experiment (runs in .venv_cbx).

Question: is joy's failure the MOUTH (IndexTTS-2 can't produce positive valence) or
the EAR (our judge undervalues synthetic positive valence)?

Method: a SECOND, unrelated synthetic mouth (Chatterbox, MIT) attempts the same
emotions via its own control surface (emotional reference clip + exaggeration knob).
Same sentence, same judge.
  * Chatterbox joy scores V>0 to the judge  -> IndexTTS-2's mouth is broken; ear OK.
  * Chatterbox joy ALSO reads V<=0          -> suspicion shifts to the ear
                                               (synthetic-voice valence bias).
Anger/sadness/neutral included as cross-model controls.

Run:  .venv_cbx/bin/python tts_steering/cbx_probe.py
"""

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tts_steering/out/cbx"
OUT.mkdir(parents=True, exist_ok=True)
TEXT = "The table is in the room, and the door is closed."
RAV = ROOT / "data/ravdess/Actor_01"
REFS = {
    "neutral": RAV / "03-01-01-01-01-01-01.wav",   # neutral
    "joy":     RAV / "03-01-03-02-01-01-01.wav",   # happy, strong intensity
    "sadness": RAV / "03-01-04-02-01-01-01.wav",   # sad, strong
    "anger":   RAV / "03-01-05-02-01-01-01.wav",   # angry, strong
}
#        name              target     ref        exaggeration
SPECS = [
    ("cbx_neutral_e05",    "neutral", "neutral", 0.5),
    ("cbx_joy_e05",        "joy",     "joy",     0.5),
    ("cbx_joy_e09",        "joy",     "joy",     0.9),
    ("cbx_sad_e05",        "sadness", "sadness", 0.5),
    ("cbx_sad_e09",        "sadness", "sadness", 0.9),
    ("cbx_anger_e07",      "anger",   "anger",   0.7),
]

todo = [s for s in SPECS if not (OUT / f"{s[0]}.wav").exists()]
print(f"{len(SPECS)} specs, {len(todo)} to synthesize", flush=True)
if todo:
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=dev)
    print(f"[chatterbox loaded on {dev} in {time.time()-t0:.0f}s]", flush=True)
    for name, target, ref, exagg in todo:
        t1 = time.time()
        wav = model.generate(TEXT, audio_prompt_path=str(REFS[ref]),
                             exaggeration=exagg, cfg_weight=0.5)
        torchaudio.save(str(OUT / f"{name}.wav"), wav, model.sr)
        print(f"  {name:20s} {time.time()-t1:.0f}s", flush=True)

import json
manifest = [{"name": n, "target": t, "control": f"ref={r},exagg={e}",
             "file": str(OUT / f"{n}.wav")} for n, t, r, e in SPECS]
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("CBX_PROBE_DONE", flush=True)
