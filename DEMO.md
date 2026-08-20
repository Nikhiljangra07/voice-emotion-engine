# DEMO.md — the 5-minute live demo

The pitch in one line: **"Play anything with a voice in it, and watch the
machine read its emotion in real time — locally, free, in two languages
it was never told apart."**

---

## Before they arrive (2 minutes, do once)

```bash
cd ~/Desktop/LoRa_WaveformEngine

# 1. route system audio to speakers + the ear simultaneously
out/ear_multiout

# 2. sanity check: this must print device list including "BlackHole 2ch"
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -A5 "audio devices"
```

If `out/ear_multiout` says *created* or *already exists — reusing*, you're routed.
You will still hear everything through the speakers normally.

## The demo (in front of them)

```bash
.venv_diar/bin/python scripts/live_ear.py --device 0
```

1. Wait ~20 s until `--- no speech ---` lines tick by (that's the gate proving
   it ignores silence).
2. Play **anything with talking** — a YouTube interview, a movie scene, a
   podcast. Emotional content demos best (an argument scene, a comedy bit).
3. Narrate the screen, top to bottom:
   - the colored bar under each new window = *what emotion, right now*
   - the terminal line = V (pleasant↔unpleasant), A (calm↔intense), name,
     and the inference time — **~250 ms per window, 6× faster than real time,
     all local**
   - when music or silence plays: `--- no speech ---` — *it knows the
     difference between a voice and a soundtrack*
4. After 3–5 minutes, **Ctrl-C**. The session ends by rendering the
   **Affectogram** — open it:

```bash
open out/live_ear/device0_*_affectogram.png
```

   Walk them through it: emotion ribbon (the story in one strip), the three
   dimension traces, the circumplex path ("the whole session as one journey
   through emotion space"), the fact box ("every number reproducible from
   this footer").

## The two killer facts to say out loud

- **The dub-vs-sub experiment.** Same anime episode played twice, English dub
  and Japanese original. Scene-level emotional shape correlated r = 0.6–0.8 —
  *the ear hears the story, not the language.* (And it caught a +0.13 valence
  offset in Japanese — it measures its own biases.)
- **Honesty by design.** The naming layer was scored against 4,083
  human-labeled clips: names are right ~38 % of the time on 8 classes, and the
  ceiling even with perfect dimensions is 42 % — so the UI treats names as
  hints and lets the dimensional graph carry the signal. Nothing here
  pretends to be better than it measured.

## Fallback demo (no loopback, no internet, 60 seconds)

If audio routing misbehaves, stream a local file through the identical live
pipe — same code path, clock-paced:

```bash
.venv_diar/bin/python scripts/live_ear.py --simulate tts_steering/out/real_world/swn_16k.wav --duration 240
```

(1943 radio drama, public domain, dramatic from minute one.)

## After they leave

```bash
out/ear_multiout revert     # audio back to plain speakers
```

## If something breaks

| Symptom | Fix |
|---|---|
| `BlackHole 2ch not found` | `sudo killall coreaudiod` (you'll run it), then `out/ear_multiout` again |
| No windows appearing while video plays | System Settings → Sound → Output must be **Ear Multi-Output** |
| Everything gated as no-speech | Content is music-heavy — normal; or lower the bar: `--speech-gate 0.3` |
| Wrong device index | List devices (command above); BlackHole's *audio* index goes to `--device` |
