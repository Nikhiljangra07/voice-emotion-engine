"""Synthesis worker — runs in the VENDOR env, driven by a JSON job file.

Loads IndexTTS-2 once, synthesizes every job, skips existing outputs (resumable).

Run:  cd tts_steering/vendor/index-tts && \
      PYTHONPATH=$PWD .venv/bin/python ../../synth_worker.py jobs.json
"""

import json
import sys
import time
from pathlib import Path

jobs = json.loads(Path(sys.argv[1]).read_text())
todo = [j for j in jobs if not Path(j["out"]).exists()]
print(f"{len(jobs)} jobs, {len(todo)} to synthesize", flush=True)

if todo:
    t0 = time.time()
    from indextts.infer_v2 import IndexTTS2
    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                    use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
    print(f"[model loaded in {time.time()-t0:.0f}s]", flush=True)
    for j in todo:
        t1 = time.time()
        if "emo_audio" in j:
            # emotion cloned from a reference clip (P4.12b channel);
            # mutually exclusive with emo_vector by API design
            tts.infer(spk_audio_prompt=j["prompt"], text=j["text"],
                      output_path=j["out"], emo_audio_prompt=j["emo_audio"],
                      emo_alpha=j.get("emo_alpha", 1.0),
                      use_random=False, verbose=False)
        else:
            tts.infer(spk_audio_prompt=j["prompt"], text=j["text"],
                      output_path=j["out"], emo_vector=j["vector"],
                      use_random=False, verbose=False)
        print(f"  {Path(j['out']).name:36s} {time.time()-t1:.0f}s", flush=True)

print("WORKER_DONE", flush=True)
