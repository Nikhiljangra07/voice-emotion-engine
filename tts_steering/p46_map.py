"""P4.6 — THE TRANSFER MAP: vector -> emotion across 19 sentences.

The declared recommencement point (log 2026-08-08), now carrying every open
question of the campaign:
  * sentence hypothesis for SADNESS (S3 unlocked it once — replicate + map)
  * sentence hypothesis for JOY (S1 closed across all channels; do warm
    sentences carry the valence the knobs cannot?)
  * the subtract-tension joy region from P4.13 (calm/mel microdoses)
  * stability of the anger/surprise winners across sentence types
  * Phase-5 Gate 3: training-grade data variety for the ledger

Design: 19 sentences x 6 semantic categories x 11 fixed configs (no
optimization — this is a MAP). Every clip a ledger row
(system=indextts2-p46), misses kept, resumable per sentence batch.
Sentence semantics are the INDEPENDENT VARIABLE here by design — the
delivery-only law applied to cross-system benchmarks, not to this map; the
variation is the experiment and is documented as such.

Run:  .venv_tts/bin/python tts_steering/p46_map.py
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "out/p46"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
REF_JOY = str(ROOT / "data/ravdess/Actor_01/03-01-03-02-01-01-01.wav")

DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
CENTROIDS = {"sadness": (-0.28, 0.39, -0.07), "joy": (+0.30, 0.58, +0.19),
             "anger": (-0.42, 0.70, +0.41), "surprise": (+0.05, 0.64, +0.26),
             "neutral": (-0.04, 0.45, +0.03)}

SENTENCES = {  # id: (text, category)
    "s01": ("The table is in the room, and the door is closed.", "flat"),
    "s02": ("The meeting starts at nine tomorrow morning.", "flat"),
    "s03": ("The report is on the desk in the main office.", "flat"),
    "s04": ("He parked the car outside the gray building.", "flat"),
    "s05": ("She finally got the letter she had been waiting for.", "warm"),
    "s06": ("The little garden was full of flowers this morning.", "warm"),
    "s07": ("We are going to see them again this weekend.", "warm"),
    "s08": ("She never came back after that day.", "somber"),
    "s09": ("The house has been empty for a long time now.", "somber"),
    "s10": ("He put the old photograph back in the drawer.", "somber"),
    "s11": ("I can't believe you did that again!", "exclaim"),
    "s12": ("Watch out, the glass is about to fall!", "exclaim"),
    "s13": ("That is exactly what I was afraid of.", "exclaim"),
    "s14": ("Why would anyone leave without saying goodbye?", "question"),
    "s15": ("Did you hear what happened at the station?", "question"),
    "s16": ("Stop it right now.", "short"),
    "s17": ("It's over.", "short"),
    "s18": ("When the lights went out, everyone in the room stayed quiet "
            "and waited for someone to speak first.", "long"),
    "s19": ("She opened the letter and read it slowly.", "somber"),
}


def vec(**kw):
    return [round(float(kw.get(k, 0.0)), 2) for k in DIMS]


CONFIGS = [  # (name, target, vector-or-None, emo_ref-or-None)
    ("anger_a07",     "anger",    vec(angry=0.7), None),
    ("anger_a08c02",  "anger",    vec(angry=0.8, calm=0.2), None),
    ("surp_08",       "surprise", vec(surprised=0.8), None),
    ("surp_12",       "surprise", vec(surprised=1.2), None),
    ("sad_m10c03",    "sadness",  vec(melancholic=1.0, calm=0.3), None),
    ("sad_s04",       "sadness",  vec(sad=0.4), None),
    ("joy_h035",      "joy",      vec(happy=0.35), None),
    ("joy_detension", "joy",      vec(calm=0.3, melancholic=0.2), None),
    ("joy_h02c04",    "joy",      vec(happy=0.2, calm=0.4), None),
    ("joyref_hm10",   "joy",      None, (REF_JOY, 1.0)),
    ("neutral_zero",  "neutral",  vec(), None),
]


def synth(jobs):
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if not todo:
        return 0.0
    jf = OUT / "_jobs.json"
    jf.write_text(json.dumps(todo))
    t0 = time.time()
    proc = subprocess.run(
        [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
         str(jf)],
        cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                              "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=7200)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"synth failed:\n{proc.stderr[-500:]}")
    return time.time() - t0


def main():
    results = (json.loads((OUT / "results.json").read_text())
               if (OUT / "results.json").exists() else {})
    ledgered = set(json.loads((OUT / "ledgered.json").read_text())) \
        if (OUT / "ledgered.json").exists() else set()

    t_start = time.time()
    for si, (sid, (text, cat)) in enumerate(SENTENCES.items(), 1):
        jobs, meta = [], []
        for name, target, v, ref in CONFIGS:
            wav = OUT / f"{sid}_{name}.wav"
            job = {"prompt": P_NEU, "text": text, "out": str(wav)}
            if ref:
                job["emo_audio"], job["emo_alpha"] = ref
            else:
                job["vector"] = v
            jobs.append(job)
            meta.append((name, target, v, ref, str(wav)))
        dt = synth(jobs)
        new = [w for _, _, _, _, w in meta if w not in results]
        if new:
            for wav, m in zip(new, judge(new)):
                results[wav] = {"V": m["V"], "A": m["A"], "D": m["D"],
                                "family": m["judge_family"],
                                "conf": m["judge_confidence"]}
            (OUT / "results.json").write_text(json.dumps(results))
        rows = []
        n = sum(1 for _ in open(LEDGER)) - 1
        hits = 0
        for name, target, v, ref, wav in meta:
            m = results[wav]
            c = CENTROIDS[target]
            d = ((m["V"] - c[0]) ** 2 + (m["A"] - c[1]) ** 2
                 + (m["D"] - c[2]) ** 2) ** 0.5
            hit = m["family"] == target
            hits += hit
            if wav not in ledgered:
                n += 1
                ctrl = (f"emo_ref=rav_happy_male;alpha=1.0" if ref
                        else f"emo_vector={v}") + f";sent={sid};cat={cat}"
                rows.append([n, "indextts2-p46", target, ctrl,
                             round(m["V"], 3), round(m["A"], 3),
                             round(m["D"], 3), m["family"],
                             round(m["conf"], 2), round(d, 3), int(hit)])
                ledgered.add(wav)
        if rows:
            with open(LEDGER, "a", newline="") as f:
                csv.writer(f).writerows(rows)
            (OUT / "ledgered.json").write_text(json.dumps(sorted(ledgered)))
        el = (time.time() - t_start) / 60
        print(f"[{si:2d}/19] {sid} ({cat:8s}) synth {dt:4.0f}s  "
              f"hits {hits}/11  ledger {n}  elapsed {el:.0f}m", flush=True)

    # ---------------- the map ----------------
    print("\n" + "=" * 74 + "\nTRANSFER MAP (judge family per sentence)\n"
          + "=" * 74)
    for name, target, v, ref, _ in [(c[0], c[1], c[2], c[3], None)
                                    for c in CONFIGS]:
        cells = []
        for sid in SENTENCES:
            m = results[str(OUT / f"{sid}_{name}.wav")]
            fam = m["family"]
            mark = fam[:4].upper() if fam == target else fam[:4]
            cells.append(f"{sid[1:]}:{mark}")
        print(f"  {name:14s} ({target:8s}) " + " ".join(cells))

    print("\nHIT RATE by sentence category x target:")
    cats = sorted({c for _, c in SENTENCES.values()})
    targets = ["anger", "surprise", "sadness", "joy", "neutral"]
    print(f"  {'cat':9s}" + "".join(f"{t:>10s}" for t in targets))
    for cat in cats:
        sids = [s for s, (_, c) in SENTENCES.items() if c == cat]
        row = f"  {cat:9s}"
        for t in targets:
            tot, hit = 0, 0
            for name, tgt, _, _ in CONFIGS:
                if tgt != t:
                    continue
                for sid in sids:
                    m = results[str(OUT / f"{sid}_{name}.wav")]
                    tot += 1
                    hit += m["family"] == t
            row += f"{hit}/{tot:>3d}   " if tot else "     --   "
        print(row)

    joy_hits = [(sid, name) for sid in SENTENCES
                for name, t, _, _ in CONFIGS if t == "joy"
                and results[str(OUT / f'{sid}_{name}.wav')]["family"] == "joy"]
    sad_hits = [(sid, name) for sid in SENTENCES
                for name, t, _, _ in CONFIGS if t == "sadness"
                and results[str(OUT / f'{sid}_{name}.wav')]["family"]
                == "sadness"]
    print(f"\nJOY verdicts anywhere: {joy_hits or 'NONE'}")
    print(f"SADNESS verdicts anywhere: {sad_hits or 'NONE'}")
    print("P46_MAP_DONE")


if __name__ == "__main__":
    main()
