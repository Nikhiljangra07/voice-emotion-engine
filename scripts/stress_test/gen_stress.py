"""Generate adversarial stress-test clips for the waveform engine.

Every clip is designed to break a naive audio pipeline: degenerate signals,
corrupted containers, extreme durations, out-of-domain content.
"""
import numpy as np
import soundfile as sf
from pathlib import Path

OUT = Path(__file__).parent / "stress"
OUT.mkdir(exist_ok=True)
SR = 16000
rng = np.random.default_rng(7)

# load one real speech clip as raw material for degradations
speech, sr_in = sf.read("own_voice/001.wav", dtype="float32")
assert sr_in == SR, f"expected 16k, got {sr_in}"

def save(name, y, sr=SR):
    sf.write(str(OUT / name), y.astype(np.float32), sr)
    print(f"{name:34s} {len(y)/sr:8.2f}s")

# ── degenerate signals ─────────────────────────────────────────────
save("01_pure_silence_3s.wav", np.zeros(3 * SR))                       # all zeros
save("02_white_noise_3s.wav", rng.normal(0, 0.3, 3 * SR).clip(-1, 1))  # no voice
save("03_tone_440hz_3s.wav", 0.5 * np.sin(2*np.pi*440*np.arange(3*SR)/SR))  # pure tone
save("04_dc_offset_3s.wav", np.full(3 * SR, 0.7))                      # constant DC
# chord = crude "instrumental" proxy (out-of-domain content)
t = np.arange(4 * SR) / SR
chord = sum(0.2*np.sin(2*np.pi*f*t) for f in (261.6, 329.6, 392.0, 523.2))
env = np.exp(-t * 0.5)
save("05_synth_chord_4s.wav", (chord * env))

# ── extreme durations ──────────────────────────────────────────────
save("06_too_short_0p1s.wav", speech[: int(0.1 * SR)])                 # 0.1s
save("07_borderline_0p6s.wav", speech[: int(0.6 * SR)])                # just above min
save("08_single_sample.wav", speech[:1])                               # 1 sample

# ── degraded speech ────────────────────────────────────────────────
save("09_clipped_speech.wav", np.clip(speech * 25, -1, 1))             # hard clipping
save("10_whisper_quiet.wav", speech * 0.001)                           # -60 dB
noise = rng.normal(0, 1.0, len(speech)).astype(np.float32)
sp = speech / (np.abs(speech).max() + 1e-9)
save("11_snr_minus5db.wav", (sp + 1.78 * noise / (np.abs(noise).max()+1e-9)).clip(-1, 1))  # noise louder than speech
save("12_speech_8khz.wav", speech[::2], sr=8000)                       # low sample rate container
# stereo file (channels mismatch test)
sf.write(str(OUT / "13_stereo_speech.wav"),
         np.stack([speech, speech * 0.5], axis=1), SR)
print(f"{'13_stereo_speech.wav':34s} {len(speech)/SR:8.2f}s (stereo)")
# intermittent: 0.3s speech bursts inside long silence
gap = np.zeros(2 * SR, dtype=np.float32)
burst = speech[: int(0.3 * SR)]
save("14_sparse_bursts.wav", np.concatenate([gap, burst, gap, burst, gap]))
# reversed speech (voice-like spectrum, no linguistic content)
save("15_reversed_speech.wav", speech[::-1])

# ── corrupted containers ───────────────────────────────────────────
(OUT / "16_empty_file.wav").write_bytes(b"")
print("16_empty_file.wav                     0 bytes")
(OUT / "17_garbage_bytes.wav").write_bytes(bytes(rng.integers(0, 256, 5000, dtype=np.uint8)))
print("17_garbage_bytes.wav                  garbage")
full = (OUT / "01_pure_silence_3s.wav").read_bytes()
good = Path("own_voice/001.wav").read_bytes()
(OUT / "18_truncated_header.wav").write_bytes(good[: len(good) // 2])  # valid header, half the data
print("18_truncated_header.wav               truncated mid-data")
(OUT / "19_not_audio.txt.wav").write_bytes(b"hello, I am definitely not audio\n" * 100)
print("19_not_audio.txt.wav                  text as .wav")

print("\nDONE:", len(list(OUT.iterdir())), "stress files")
