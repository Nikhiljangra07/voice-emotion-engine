"""P4.14 — THE TRAJECTORY: a lengthier speech, read back window by window.

The mouth WRITES a four-act emotional journey using the P4.6 map's proven
configs; the ear READS it back at four candidate window sizes; the data
decides the appropriate interval; the fluctuation is graphed.

  Act 1 NEUTRAL  flat sentences,   zero vector
  Act 2 JOY      warm sentences,   reference channel (strongest human-agreed)
  Act 3 SADNESS  somber sentences, mel 1.0 + calm 0.3 (10-verdict config)
  Act 4 ANGER    exclaim sentences, angry 0.8 + calm 0.2 (19/19 universal)

Window-size selection is DATA-DRIVEN (no magic numbers): for each size in
{1.5, 2, 3, 4}s (50% overlap) compute the correlation between the commanded
V/A step-trajectory and the measured windowed V/A. Report all; the best size
is chosen by mean(corr_V, corr_A) with boundary lag as tiebreak.

Known limitation, documented: the mouth's emotion is per-utterance — the
journey is written act-by-act and concatenated (0.4 s gaps). Within-utterance
emotional velocity remains future work (the mouth cannot glide mid-sentence).

Run:  .venv_tts/bin/python tts_steering/trajectory_p414.py
Then: venv/bin/python tts_steering/plot_traj_p414.py   (the graph)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import wavlm_vad, e2v_family  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "out/p414"
WIN = OUT / "windows"
OUT.mkdir(parents=True, exist_ok=True)
WIN.mkdir(exist_ok=True)
VENDOR = HERE / "vendor/index-tts"
P_NEU = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
REF_JOY = str(ROOT / "data/ravdess/Actor_01/03-01-03-02-01-01-01.wav")


def vec(**kw):
    order = ["happy", "angry", "sad", "afraid", "disgusted",
             "melancholic", "surprised", "calm"]
    return [round(float(kw.get(k, 0.0)), 2) for k in order]


ACTS = [
    ("neutral", "The report is on the desk in the main office. The table is "
                "in the room, and the door is closed. He parked the car "
                "outside the gray building.",
     {"vector": vec()}),
    ("joy", "She finally got the letter she had been waiting for. We are "
            "going to see them again this weekend. The little garden was "
            "full of flowers this morning.",
     {"emo_audio": REF_JOY, "emo_alpha": 1.0}),
    ("sadness", "She never came back after that day. The house has been "
                "empty for a long time now. He put the old photograph back "
                "in the drawer.",
     {"vector": vec(melancholic=1.0, calm=0.3)}),
    ("anger", "I told you to stop, and you did it again! I can't believe "
              "you did that again! Stop it right now.",
     {"vector": vec(angry=0.8, calm=0.2)}),
]
CENTROIDS = {"neutral": (-0.04, 0.45), "joy": (+0.30, 0.58),
             "sadness": (-0.28, 0.39), "anger": (-0.42, 0.70)}
SIZES = [1.5, 2.0, 3.0, 4.0]
GAP = 0.4


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-200:]}")
    return r.stdout


def main():
    # ---- 1. the mouth writes the journey (act by act) ----
    jobs = []
    for name, text, ctrl in ACTS:
        wav = OUT / f"act_{name}.wav"
        job = {"prompt": P_NEU, "text": text, "out": str(wav), **ctrl}
        if "vector" not in job:
            job["emo_audio"] = job.get("emo_audio")
        jobs.append(job)
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if todo:
        jf = OUT / "_jobs.json"
        jf.write_text(json.dumps(todo))
        print(f"synthesizing {len(todo)} acts ...", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"),
             str(jf)],
            cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR),
                                  "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=3600)
        if "WORKER_DONE" not in proc.stdout:
            raise RuntimeError(f"synth failed:\n{proc.stderr[-400:]}")
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ---- 2. concatenate with gaps; record act boundaries ----
    full = OUT / "journey.wav"
    bounds, t = [], 0.0
    concat_parts = []
    for name, _, _ in ACTS:
        wav = OUT / f"act_{name}.wav"
        dur = float(sh(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(wav)]).strip())
        bounds.append({"act": name, "start": t, "end": t + dur})
        concat_parts.append(str(wav))
        t += dur + GAP
    total = bounds[-1]["end"]
    if not full.exists():
        lst = OUT / "_concat.txt"
        gap_wav = OUT / "_gap.wav"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            "anullsrc=r=22050:cl=mono", "-t", str(GAP), str(gap_wav)])
        lines = []
        for i, p in enumerate(concat_parts):
            lines.append(f"file '{p}'")
            if i < len(concat_parts) - 1:
                lines.append(f"file '{gap_wav}'")
        lst.write_text("\n".join(lines))
        sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe",
            "0", "-i", str(lst), "-ar", "22050", "-ac", "1", str(full)])
    print(f"journey: {total:.1f}s, acts at " +
          ", ".join(f"{b['act']}[{b['start']:.0f}-{b['end']:.0f}s]"
                    for b in bounds), flush=True)

    # ---- 3. the ear reads it back at each window size ----
    results = {}
    for w in SIZES:
        hop = w / 2
        starts, files = [], []
        s = 0.0
        while s + w <= total + 1e-6:
            f = WIN / f"w{int(w*10)}_{int(s*100):06d}.wav"
            if not f.exists():
                sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(s),
                    "-t", str(w), "-i", str(full), str(f)])
            starts.append(s)
            files.append(str(f))
            s += hop
        print(f"window {w}s: {len(files)} windows, reading V/A/D ...",
              flush=True)
        rows = []
        for i in range(0, len(files), 60):
            rows += wavlm_vad(files[i:i + 60])
        results[str(w)] = {"starts": starts,
                           "V": [r["valence"] for r in rows],
                           "A": [r["arousal"] for r in rows],
                           "D": [r["dominance"] for r in rows],
                           "wavlm_emo": [r["emotion"] for r in rows]}

    # ---- 4. data-driven window choice: commanded vs measured ----
    def commanded(t_c):
        for b in bounds:
            if b["start"] <= t_c <= b["end"] + GAP:
                return CENTROIDS[b["act"]]
        return CENTROIDS[bounds[-1]["act"]]

    import math
    def corr(x, y):
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        sx = math.sqrt(sum((a - mx) ** 2 for a in x))
        sy = math.sqrt(sum((a - my) ** 2 for a in y))
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) \
            if sx > 1e-9 and sy > 1e-9 else 0.0

    print("\nWINDOW-SIZE SELECTION (commanded-vs-measured correlation):")
    best = None
    for w in SIZES:
        r = results[str(w)]
        centers = [s + w / 2 for s in r["starts"]]
        cmd = [commanded(c) for c in centers]
        cv = corr([c[0] for c in cmd], r["V"])
        ca = corr([c[1] for c in cmd], r["A"])
        score = (cv + ca) / 2
        print(f"  {w:.1f}s: corr_V={cv:+.2f}  corr_A={ca:+.2f}  "
              f"mean={score:+.2f}  ({len(centers)} windows)")
        if best is None or score > best[1]:
            best = (w, score)
    chosen = best[0]
    print(f"\nCHOSEN WINDOW: {chosen}s (best mean correlation {best[1]:+.2f})")

    # ---- 5. judge families along the chosen trajectory ----
    r = results[str(chosen)]
    files = [str(WIN / f"w{int(chosen*10)}_{int(s*100):06d}.wav")
             for s in r["starts"]]
    fams = []
    for i in range(0, len(files), 60):
        fams += e2v_family(files[i:i + 60])
    r["judge"] = [f["family"] for f in fams]

    json.dump({"bounds": bounds, "total": total, "results": results,
               "chosen": chosen, "acts": [a[0] for a in ACTS]},
              open(OUT / "trajectory.json", "w"))
    print(f"\nsaved {OUT/'trajectory.json'} — run plot_traj_p414.py "
          f"for the graph")
    print("TRAJECTORY_DONE")


if __name__ == "__main__":
    main()
