"""LIVE EAR — real-time emotion mapping of playing audio. The main product.

Audio plays; the ear listens in parallel. Every stride it maps the last
3-second window to V/A/D (WavLM-ft, resident in memory), names the emotion
from the data-grounded PAD centroids (domain-general — no enrollment
needed), and draws the trajectory on a LIVE graph while the sound is still
playing. Naming via centroids, not the kNN judge: the 1943 broadcast test
proved dimensions travel and enrollment-bound categories don't.

Modes:
  --input movie.wav --play    play the file aloud AND map it in real time,
                              paced to the clock (the parallel-playing demo)
  --input movie.wav --fast    no audio, no pacing — as fast as the GPU goes
  --window 3.0 --stride 1.5   the P4.14 data-chosen defaults

Outputs: live matplotlib window + rolling console verdicts +
out/live_ear/<name>_traj.json + final <name>_traj.png. Per-window inference
latency is printed — the real-time headroom proof.

Run:  .venv_diar/bin/python scripts/live_ear.py --input clip.wav --play
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.predict_wavlm_ft import (WavLMRegressor, load_audio,  # noqa: E402
                                      normalize_vad, Namer)
from silero_vad import load_silero_vad  # noqa: E402

COLORS = {"anger": "#d62728", "fear": "#9467bd", "joy": "#e6b800",
          "sadness": "#1f77b4", "surprise": "#ff7f0e", "neutral": "#909090",
          "disgust": "#2ca02c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--play", action="store_true",
                    help="play the audio aloud while mapping (paced)")
    ap.add_argument("--fast", action="store_true",
                    help="no pacing — process as fast as possible")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--stride", type=float, default=1.5)
    ap.add_argument("--speech-gate", type=float, default=0.5,
                    help="min mean speech prob (Silero) to emote a window; "
                         "0 disables the gate")
    ap.add_argument("--model", default="models/wavlm_vad_ft")
    ap.add_argument("--namer", default="models/namer_msp_final")
    args = ap.parse_args()

    out_dir = ROOT / "out/live_ear"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading WavLM-ft on {device} ...", flush=True)
    model = WavLMRegressor(str(ROOT / args.model))
    head_sd = torch.load(ROOT / args.model / "head.pt", map_location="cpu")
    model.head.load_state_dict(head_sd)
    model.to(device).eval()
    namer = Namer(str(ROOT / args.namer))
    vad = load_silero_vad()  # resident speech/music-noise gate

    y = load_audio(args.input, max_s=1e9)   # full file (loader caps at 8s by default)
    sr = 16000
    dur = len(y) / sr
    n_steps = max(1, int((dur - args.window) / args.stride) + 1)
    print(f"{args.input}: {dur:.0f}s -> {n_steps} windows "
          f"({args.window}s / stride {args.stride}s)", flush=True)

    import matplotlib
    matplotlib.use("MacOSX" if args.play else "Agg")
    import matplotlib.pyplot as plt
    live = args.play
    if live:
        plt.ion()
    fig, (axv, axa) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"LIVE EAR — {stem}  ({args.window}s windows)")
    axv.set_ylabel("Valence"); axv.set_ylim(-1, 1)
    axv.axhline(0, color="k", lw=0.5, alpha=0.4)
    axa.set_ylabel("Arousal"); axa.set_ylim(0, 1)
    axa.set_xlabel("seconds")
    lv, = axv.plot([], [], lw=1.8, color="#0b3d91")
    la, = axa.plot([], [], lw=1.8, color="#8b0000")

    player = None
    if args.play:
        player = subprocess.Popen(["afplay", args.input])
    t_start = time.time()
    rows, ts, vs, As = [], [], [], []
    lat = []
    sp_hist = []
    warm = None  # MPS warmup on first window
    for k in range(n_steps):
        t_pos = k * args.stride
        if args.play and not args.fast:
            wait = (t_pos + args.window) - (time.time() - t_start)
            if wait > 0:
                time.sleep(wait)
        seg = y[int(t_pos * sr):int((t_pos + args.window) * sr)]
        peak = float(np.max(np.abs(seg))) or 1.0
        seg = seg / peak
        t0 = time.time()
        # speech gate: Silero on 512-sample chunks, mean prob over window
        if args.speech_gate > 0:
            st = torch.from_numpy(seg)
            probs = []
            vad.reset_states()
            for c0 in range(0, len(st) - 512, 512):
                probs.append(float(vad(st[c0:c0 + 512], sr).item()))
            sp = float(np.mean(probs)) if probs else 0.0
            sp_hist.append(sp)
            sp_s = float(np.median(sp_hist[-3:]))  # causal 3-win median
            if sp_s < args.speech_gate:
                ms = (time.time() - t0) * 1000
                if warm is None:
                    warm = ms
                rows.append({"t": round(t_pos, 2), "V": None, "A": None,
                             "D": None, "emotion": "no-speech",
                             "speech_prob": round(sp, 2),
                             "ms": round(ms, 1)})
                print(f"[{t_pos:6.1f}s] --- no speech (p={sp:.2f}) --- "
                      f"{ms:5.0f}ms", flush=True)
                axv.axvspan(t_pos, t_pos + args.stride, color="#000000",
                            alpha=0.06)
                if live:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                continue
        else:
            sp = -1.0
        wav = torch.from_numpy(seg).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = model(wav, mask).float().cpu().numpy().ravel()
        raw = np.clip(out, 0.0, 1.0) * 6.0 + 1.0
        pad = normalize_vad(*raw)
        r = namer.predict(pad)
        ms = (time.time() - t0) * 1000
        if warm is None:
            warm = ms
        else:
            lat.append(ms)
        V, A, D = float(pad[0]), float(pad[1]), float(pad[2])
        emo = r["emotion"]
        rows.append({"t": round(t_pos, 2), "V": round(V, 3),
                     "A": round(A, 3), "D": round(D, 3), "emotion": emo,
                     "ambiguous": bool(r["ambiguous"]),
                     "speech_prob": round(sp, 2), "ms": round(ms, 1)})
        ts.append(t_pos); vs.append(V); As.append(A)
        bar = "#" * int((V + 1) * 10)
        print(f"[{t_pos:6.1f}s] V={V:+.2f} A={A:.2f} {emo:9s} "
              f"{'(?)' if r['ambiguous'] else '   '} {ms:5.0f}ms  {bar}",
              flush=True)
        lv.set_data(ts, vs); la.set_data(ts, As)
        c = COLORS.get(emo, "#cccccc")
        axv.axvspan(t_pos, t_pos + args.stride, color=c, alpha=0.10)
        axv.set_xlim(0, max(30, t_pos + args.window))
        if live:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    if player:
        player.wait()
    (out_dir / f"{stem}_traj.json").write_text(json.dumps(rows))
    fig.savefig(out_dir / f"{stem}_traj.png", dpi=130)
    med = np.median(lat) if lat else float("nan")
    gated = sum(1 for r in rows if r["emotion"] == "no-speech")
    fams = [r["emotion"] for r in rows if r["emotion"] != "no-speech"]
    top = max(set(fams), key=fams.count) if fams else "-"
    print(f"speech gate: {gated}/{len(rows)} windows suppressed "
          f"(music/silence/noise)")
    print(f"\n{len(rows)} windows · median inference {med:.0f} ms/window "
          f"(first {warm:.0f} ms incl. warmup) · stride budget "
          f"{args.stride*1000:.0f} ms -> real-time headroom "
          f"{args.stride*1000/max(med,1):.0f}x")
    print(f"dominant: {top} ({fams.count(top)}/{len(fams)}) · "
          f"V median {np.median(vs):+.2f} · A median {np.median(As):.2f}")
    print(f"saved: out/live_ear/{stem}_traj.json + .png")
    print("LIVE_EAR_DONE")


if __name__ == "__main__":
    main()
