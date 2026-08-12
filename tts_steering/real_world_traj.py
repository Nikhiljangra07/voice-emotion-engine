"""REAL-WORLD TRAJECTORY — the ear on a found 30-minute drama, no lab anywhere.

Input: "Sorry, Wrong Number" (Suspense, May 25 1943, public domain,
archive.org) — chosen BECAUSE its emotional arc is documented history: a
woman drifts from irritation into mounting anxiety into terror. If the ear
is real, the graph must show that drift without being told anything.

Protocol (P4.14's data-chosen interval): 3.0 s windows, 50 % overlap
(stride 1.5 s) across the full 1796 s broadcast -> ~1195 windows. Each
window: WavLM V/A/D + e2v family verdict (both frozen). Per-window RMS
recorded so music/silence segments can be flagged honestly — the ear is
speech-trained; announcer music is out-of-domain and stays in the data.

Output: out/real_world/traj.json (all windows) — graph by real_world_plot.py.

Run:  .venv_tts/bin/python tts_steering/real_world_traj.py
"""

import json
import subprocess
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import wavlm_vad, e2v_family  # noqa: E402

OUT = HERE / "out/real_world"
WIN = OUT / "win"
WIN.mkdir(parents=True, exist_ok=True)
SRC = OUT / "swn_16k.wav"
WLEN, STRIDE = 3.0, 1.5


def rms_of(path):
    import array
    with wave.open(str(path), "rb") as w:
        a = array.array("h", w.readframes(w.getnframes()))
    if not a:
        return 0
    return int((sum(x * x for x in a) / len(a)) ** 0.5)


def main():
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(SRC)], capture_output=True, text=True).stdout)
    starts = []
    t = 0.0
    while t + WLEN <= dur:
        starts.append(round(t, 1))
        t += STRIDE
    print(f"{dur:.0f}s -> {len(starts)} windows of {WLEN}s "
          f"(stride {STRIDE}s)", flush=True)

    files = []
    t0 = time.time()
    for i, st in enumerate(starts):
        f = WIN / f"w{i:04d}.wav"
        if not f.exists():
            subprocess.run(
                ["ffmpeg", "-v", "quiet", "-y", "-ss", str(st), "-t",
                 str(WLEN), "-i", str(SRC), "-c", "copy", str(f)],
                check=True)
        files.append(str(f))
    print(f"sliced in {time.time()-t0:.0f}s", flush=True)

    rows = []
    B = 100
    t0 = time.time()
    vads, fams = [], []
    for i in range(0, len(files), B):
        batch = files[i:i + B]
        vads += wavlm_vad(batch)
        fams += e2v_family(batch)
        print(f"  {min(i+B,len(files))}/{len(files)} judged "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
    for st, f, v, fm in zip(starts, files, vads, fams):
        rows.append({"t": st, "V": round(v["valence"], 3),
                     "A": round(v["arousal"], 3),
                     "D": round(v["dominance"], 3),
                     "family": fm["family"], "conf": round(fm["confidence"], 2),
                     "rms": rms_of(f)})
    (OUT / "traj.json").write_text(json.dumps(rows))
    fam_counts = {}
    for r in rows:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    print("family distribution:", dict(sorted(fam_counts.items(),
                                              key=lambda kv: -kv[1])))
    print(f"windows: {len(rows)}  mean V {sum(r['V'] for r in rows)/len(rows):+.2f}")
    print("REAL_WORLD_TRAJ_DONE")


if __name__ == "__main__":
    main()
