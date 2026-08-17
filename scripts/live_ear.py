"""LIVE EAR — real-time emotion mapping of playing audio. The main product.

Audio plays; the ear listens in parallel. Every stride it maps the last
3-second window to V/A/D (WavLM-ft, resident), gates non-speech (Silero,
causal 3-window median), names the emotion from data-grounded PAD
centroids (domain-general), and draws the trajectory live.

Modes:
  --input f.wav --play      play the file aloud AND map it, clock-paced
  --input f.wav --fast      offline: as fast as the GPU goes
  --device 0                LIVE CAPTURE from an avfoundation audio device
                            (mic, or BlackHole loopback for system audio);
                            Ctrl-C or --duration N to stop
  --simulate f.wav          stream a file through the SAME live-capture
                            pipe at real-time pace (plumbing validation,
                            no driver needed)
  --speech-gate 0.5         min smoothed speech prob (0 disables)
  --window 3.0 --stride 1.5 the P4.14 data-chosen protocol

Outputs: live matplotlib window (in --play/--device/--simulate) + rolling
console verdicts + out/live_ear/<stem>_traj.json + <stem>_traj.png.

Run:  .venv_diar/bin/python scripts/live_ear.py --input clip.wav --play
      .venv_diar/bin/python scripts/live_ear.py --device 2   # BlackHole
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

from scripts.affectogram import (NameSmoother, flicker_rate,  # noqa: E402
                                 COLORS, render as render_affectogram)

SR = 16000


class Ear:
    """Resident models + live plot + rolling state. One instance per run."""

    def __init__(self, args, live):
        self.args = args
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"loading WavLM-ft on {device} ...", flush=True)
        self.model = WavLMRegressor(str(ROOT / args.model))
        head_sd = torch.load(ROOT / args.model / "head.pt",
                             map_location="cpu")
        self.model.head.load_state_dict(head_sd)
        self.model.to(device).eval()
        self.device = device
        self.namer = Namer(str(ROOT / args.namer))
        self.vad = load_silero_vad()
        self.smoother = (NameSmoother(args.smooth_k)
                         if args.smooth_k > 0 else None)

        import matplotlib
        matplotlib.use("MacOSX" if live else "Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.live = live
        if live:
            plt.ion()
        self.fig, (self.axv, self.axa) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True)
        self.axv.set_ylabel("Valence"); self.axv.set_ylim(-1, 1)
        self.axv.axhline(0, color="k", lw=0.5, alpha=0.4)
        self.axa.set_ylabel("Arousal"); self.axa.set_ylim(0, 1)
        self.axa.set_xlabel("seconds")
        (self.lv,) = self.axv.plot([], [], lw=1.8, color="#0b3d91")
        (self.la,) = self.axa.plot([], [], lw=1.8, color="#8b0000")
        self.rows, self.ts, self.vs, self.As = [], [], [], []
        self.lat, self.sp_hist = [], []
        self.warm = None

    def title(self, stem):
        self.fig.suptitle(f"LIVE EAR — {stem}  "
                          f"({self.args.window}s windows)")

    def process(self, seg, t_pos):
        """One window: gate -> V/A/D -> name -> plot -> record."""
        peak = float(np.max(np.abs(seg))) or 1.0
        seg = seg / peak
        t0 = time.time()
        sp = -1.0
        if self.args.speech_gate > 0:
            st = torch.from_numpy(seg)
            probs = []
            self.vad.reset_states()
            for c0 in range(0, len(st) - 512, 512):
                probs.append(float(self.vad(st[c0:c0 + 512], SR).item()))
            sp = float(np.mean(probs)) if probs else 0.0
            self.sp_hist.append(sp)
            sp_s = float(np.median(self.sp_hist[-3:]))
            if sp_s < self.args.speech_gate:
                ms = (time.time() - t0) * 1000
                if self.warm is None:
                    self.warm = ms
                if self.smoother:
                    self.smoother.gate()
                self.rows.append({"t": round(t_pos, 2), "V": None,
                                  "A": None, "D": None,
                                  "emotion": "no-speech",
                                  "speech_prob": round(sp, 2),
                                  "ms": round(ms, 1)})
                print(f"[{t_pos:6.1f}s] --- no speech (p={sp:.2f}) --- "
                      f"{ms:5.0f}ms", flush=True)
                self.axv.axvspan(t_pos, t_pos + self.args.stride,
                                 color="#000000", alpha=0.06)
                self._draw(t_pos)
                return
        wav = torch.from_numpy(seg).unsqueeze(0).to(self.device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = self.model(wav, mask).float().cpu().numpy().ravel()
        raw = np.clip(out, 0.0, 1.0) * 6.0 + 1.0
        pad = normalize_vad(*raw)
        r = self.namer.predict(pad)
        ms = (time.time() - t0) * 1000
        if self.warm is None:
            self.warm = ms
        else:
            self.lat.append(ms)
        V, A = float(pad[0]), float(pad[1])
        raw = r["emotion"]
        emo = self.smoother.update(raw) if self.smoother else raw
        self.rows.append({"t": round(t_pos, 2), "V": round(V, 3),
                          "A": round(A, 3), "D": round(float(pad[2]), 3),
                          "emotion": emo, "emotion_raw": raw,
                          "ambiguous": bool(r["ambiguous"]),
                          "speech_prob": round(sp, 2),
                          "ms": round(ms, 1)})
        self.ts.append(t_pos); self.vs.append(V); self.As.append(A)
        bar = "#" * int((V + 1) * 10)
        print(f"[{t_pos:6.1f}s] V={V:+.2f} A={A:.2f} {emo:9s} "
              f"{'(?)' if r['ambiguous'] else '   '} {ms:5.0f}ms  {bar}",
              flush=True)
        self.lv.set_data(self.ts, self.vs)
        self.la.set_data(self.ts, self.As)
        self.axv.axvspan(t_pos, t_pos + self.args.stride,
                         color=COLORS.get(emo, "#cccccc"), alpha=0.10)
        self._draw(t_pos)

    def _draw(self, t_pos):
        self.axv.set_xlim(0, max(30, t_pos + self.args.window))
        if self.live:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def finish(self, stem):
        out_dir = ROOT / "out/live_ear"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}_traj.json").write_text(json.dumps(self.rows))
        self.fig.savefig(out_dir / f"{stem}_traj.png", dpi=130)
        render_affectogram(self.rows, stem,
                           out_dir / f"{stem}_affectogram.png",
                           params={"window": self.args.window,
                                   "stride": self.args.stride,
                                   "speech_gate": self.args.speech_gate,
                                   "smooth_k": self.args.smooth_k})
        med = np.median(self.lat) if self.lat else float("nan")
        gated = sum(1 for r in self.rows if r["emotion"] == "no-speech")
        fams = [r["emotion"] for r in self.rows
                if r["emotion"] != "no-speech"]
        top = max(set(fams), key=fams.count) if fams else "-"
        print(f"\nspeech gate: {gated}/{len(self.rows)} suppressed")
        if self.smoother:
            seq_r = [r.get("emotion_raw") if r["emotion"] != "no-speech"
                     else None for r in self.rows]
            seq_s = [r["emotion"] if r["emotion"] != "no-speech"
                     else None for r in self.rows]
            print(f"name flicker: raw {flicker_rate(seq_r)*100:.0f}% -> "
                  f"smoothed {flicker_rate(seq_s)*100:.0f}% "
                  f"(majority-{self.args.smooth_k})")
        print(f"{len(self.rows)} windows · median inference {med:.0f} ms "
              f"· stride budget {self.args.stride*1000:.0f} ms -> "
              f"{self.args.stride*1000/max(med,1):.0f}x headroom")
        if fams:
            print(f"dominant: {top} ({fams.count(top)}/{len(fams)})")
        print(f"saved: out/live_ear/{stem}_traj.json + .png + "
              f"{stem}_affectogram.png")
        print("LIVE_EAR_DONE")


def stream_loop(ear, args):
    """Live capture (device) or simulated live stream (file, -re paced).
    Reads f32le mono 16k from an ffmpeg pipe in stride-sized chunks."""
    if args.device is not None:
        src = ["-f", "avfoundation", "-i", f":{args.device}"]
        stem = f"device{args.device}_{int(time.time())}"
        print(f"LIVE CAPTURE from avfoundation device :{args.device} "
              f"(Ctrl-C to stop)", flush=True)
    else:
        src = ["-re", "-i", args.simulate]
        stem = Path(args.simulate).stem + "_live"
        print(f"SIMULATED live stream from {args.simulate}", flush=True)
    ear.title(stem)
    # NO loudness normalization: tested (R128 loudnorm vs raw, 1943
    # broadcast, 741 shared speech windows) — V r=1.000, A r=0.999,
    # 99% name agreement. Per-window peak norm already absorbs it.
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", *src, "-ar", str(SR), "-ac", "1",
         "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE,
        start_new_session=True)  # Ctrl-C hits our loop, not ffmpeg —
    # we terminate it cleanly in finally; kills the muxer-error spam
    stride_bytes = int(args.stride * SR) * 4
    win_n = int(args.window * SR)
    buf = np.zeros(0, dtype=np.float32)
    t_pos = 0.0
    try:
        while True:
            chunk = proc.stdout.read(stride_bytes)
            if not chunk:
                break
            buf = np.concatenate(
                [buf, np.frombuffer(chunk, dtype=np.float32)])[-win_n * 2:]
            if len(buf) >= win_n:
                ear.process(buf[-win_n:].copy(), t_pos)
                t_pos += args.stride
            if args.duration and t_pos >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)
    finally:
        proc.terminate()
    ear.finish(stem)


def file_loop(ear, args):
    y = load_audio(args.input, max_s=1e9)
    dur = len(y) / SR
    n_steps = max(1, int((dur - args.window) / args.stride) + 1)
    stem = Path(args.input).stem
    ear.title(stem)
    print(f"{args.input}: {dur:.0f}s -> {n_steps} windows", flush=True)
    player = subprocess.Popen(["afplay", args.input]) if args.play else None
    t_start = time.time()
    for k in range(n_steps):
        t_pos = k * args.stride
        if args.play and not args.fast:
            wait = (t_pos + args.window) - (time.time() - t_start)
            if wait > 0:
                time.sleep(wait)
        ear.process(y[int(t_pos * SR):int((t_pos + args.window) * SR)],
                    t_pos)
    if player:
        player.wait()
    ear.finish(stem)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input")
    g.add_argument("--device", type=int,
                   help="avfoundation audio device index (see ffmpeg "
                        "-f avfoundation -list_devices true -i '')")
    g.add_argument("--simulate",
                   help="stream a file through the live pipe at real pace")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop live/simulated capture after N seconds")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--stride", type=float, default=1.5)
    ap.add_argument("--speech-gate", type=float, default=0.5)
    ap.add_argument("--smooth-k", type=int, default=5,
                    help="majority-vote name smoothing over last K speech "
                         "windows (0 disables; V/A/D never smoothed)")
    ap.add_argument("--model", default="models/wavlm_vad_ft")
    ap.add_argument("--namer", default="models/namer_msp_final")
    args = ap.parse_args()

    live = bool(args.play or args.device is not None or args.simulate)
    if args.fast:
        live = False
    ear = Ear(args, live)
    if args.input:
        file_loop(ear, args)
    else:
        stream_loop(ear, args)


if __name__ == "__main__":
    main()
