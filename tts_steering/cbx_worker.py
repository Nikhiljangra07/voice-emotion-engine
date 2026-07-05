"""Chatterbox synthesis worker — runs in .venv_cbx, driven by a JSON job file.

Jobs: [{"ref": path, "exaggeration": f, "cfg_weight": f, "text": s, "out": path}]
Loads the model once, skips existing outputs (resumable).

Run:  .venv_cbx/bin/python tts_steering/cbx_worker.py jobs.json
"""

import json
import sys
import time
from pathlib import Path

jobs = json.loads(Path(sys.argv[1]).read_text())
todo = [j for j in jobs if not Path(j["out"]).exists()]
print(f"{len(jobs)} jobs, {len(todo)} to synthesize", flush=True)

if todo:
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=dev)
    print(f"[chatterbox loaded on {dev} in {time.time()-t0:.0f}s]", flush=True)
    for j in todo:
        t1 = time.time()
        wav = model.generate(j["text"], audio_prompt_path=j["ref"],
                             exaggeration=j["exaggeration"],
                             cfg_weight=j["cfg_weight"])
        torchaudio.save(j["out"], wav, model.sr)
        print(f"  {Path(j['out']).name:34s} {time.time()-t1:.0f}s", flush=True)

print("CBX_WORKER_DONE", flush=True)
