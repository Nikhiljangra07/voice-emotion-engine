"""WPE dereverberation — strip room/vocoder reverb from any wav.

Weighted Prediction Error (Nakatani et al.) on the STFT: predicts the late
reverberation tail of each frequency bin from its own past and subtracts
it. Single-channel variant. Used two ways here:
  1. POST: clean synthesized speech (immediate fix for audible reverb)
  2. PRE:  clean the cloning reference so the mouth never learns the room

Usage:  venv/bin/python tts_steering/dereverb.py in.wav out.wav [--taps 10]
"""

import argparse

import numpy as np
import soundfile as sf
from nara_wpe.wpe import wpe
from scipy.signal import stft, istft

TAPS_DEFAULT = 10
DELAY = 3          # frames before prediction starts (protects direct sound)
NFFT = 512
HOP = 128


def dereverb(y, sr, taps=TAPS_DEFAULT):
    _, _, Y = stft(y, nperseg=NFFT, noverlap=NFFT - HOP)
    # wpe expects (channels, freq, frames)
    Z = wpe(Y[None, ...], taps=taps, delay=DELAY, iterations=5)[0]
    _, out = istft(Z, nperseg=NFFT, noverlap=NFFT - HOP)
    out = out[:len(y)]
    peak = np.abs(out).max()
    if peak > 0:
        out = out * min(1.0, np.abs(y).max() / peak)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--taps", type=int, default=TAPS_DEFAULT)
    a = ap.parse_args()
    y, sr = sf.read(a.inp)
    if y.ndim > 1:
        y = y.mean(1)
    out = dereverb(y, sr, taps=a.taps)
    sf.write(a.out, out, sr)
    print(f"dereverbed: {a.inp} -> {a.out} ({len(y)/sr:.1f}s, taps={a.taps})")


if __name__ == "__main__":
    main()
