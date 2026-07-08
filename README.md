# Voice Emotion Engine — Reading Emotion from *How* a Voice Sounds

**Predict what a speaker feels — not from their words, but from their voice.**
A dimensional (Valence / Arousal / Dominance) speech-emotion engine, built solo in two
phases: a fully handcrafted acoustic pipeline first, then a fine-tuned self-supervised
model that broke the axis everyone said was unrecoverable from audio alone.

> **Headline result:** fine-tuned WavLM-large on MSP-Podcast, evaluated on **46k held-out
> clips from unseen speakers** — CCC **Valence 0.705 / Arousal 0.714 / Dominance 0.626**.
> Valence from voice alone went **0.06 → 0.35 → 0.705** across the project — a 7× arc,
> each step earned and documented.

> 📦 **This repository contains TWO projects.**
> **Project 1 (this document): the Voice Emotion Engine** — the *ear*: audio in,
> emotion out. **Project 2 ([`tts_steering/`](tts_steering/)): Emotional TTS Steering
> & Fair Benchmark** — a separate project that points the finished ear at a synthetic
> *mouth* and closes a feedback loop. They are deliberately independent systems:
> zero shared code, zero shared dependencies, four isolated environments. Their only
> connection is a **thin bridge** ([`tts_steering/bridge.py`](tts_steering/bridge.py))
> that consumes the engine exactly as an external user would — subprocess → CLI →
> JSON. The engine powers the second project; it doesn't know it exists.
> → Project 2's own front door: **[tts_steering/WRITEUP.md](tts_steering/WRITEUP.md)**.

---

## What this engine does

Given any speech audio:

| Output | What it means |
|---|---|
| **V / A / D point** | Continuous emotion coordinates (pleasant↔unpleasant, calm↔activated, submissive↔in-control) — the PAD model, not a forced label |
| **Emotion family + confidence** | anger / fear / joy / neutral / sadness / surprise, via a decision-level **hybrid** of two SSL backbones (**≈ 69% cross-speaker**, chance 17%) |
| **Who is speaking** | ECAPA-TDNN diarization (~100% per-turn attribution with known speaker count) |
| **Expression & intensity** | Deviation from the speaker's own neutral baseline + PAD-radius intensity |

## Why it's interesting (the three findings)

1. **The valence wall fell.** Acoustic features are known-strong for arousal but weak for
   valence ("whether you feel good or bad lives in the words"). Handcrafted features
   capped at CCC 0.35 — diagnosed, documented, then broken by fine-tuning WavLM
   end-to-end: **0.705 on held-out speakers**. The lever was the representation, not the
   transcript.
2. **A frozen model out-generalized our fine-tuned one — for retrieval.** For naming a
   *stranger's* emotion, frozen emotion2vec beats the fine-tuned WavLM at cosine k-NN
   (63.4% vs 52.6%). Fine-tuning sharpened regression but *narrowed* the embedding space.
   Measured, not assumed — and the fix (score WavLM with a trained head, fuse both at the
   **decision level**, never mixing vector spaces) beats either backbone alone.
3. **We attacked our own engine and published the results.** 19 adversarial inputs
   (silence, noise, tones, corrupted files, 1-sample audio, −5 dB SNR…) against every
   path: **zero crashes, zero NaN, zero silent failures** — plus three honestly documented
   gaps (see *Known limits*).

## Architecture (three layers, strictly separated)

```
Audio ─▶ L1 ENGINE (constant)      111 acoustic features — openSMILE eGeMAPS
         │                          + Praat voice quality + prosody; validated,
         │                          adaptive F0, zero-NaN across 200k+ extractions
         ├▶ L2 MODELS (swappable)  fine-tuned WavLM-large (V/A/D regression)
         │                          + frozen emotion2vec (retrieval) — two tagged,
         │                          non-mixable vector spaces
         └▶ L3 OUTPUT              PAD point + emotion distribution + intensity
                                    + speaker turns → one JSON packet
```

The model layer was swapped **five times** (SVM → RF → SVR+calibration → frozen-SSL
fusion → fine-tuned WavLM) without touching the engine — the separation is load-bearing,
not aspirational.

> ✏️ The output space (V/A/D) began as a literal notebook drawing before any code —
> [the founding sketch](docs/pad_founding_sketch.jpg) is preserved in
> [JOURNEY.md § The Pivot](JOURNEY.md#the-pivot--from-categories-to-coordinates-june-2026).

## Results at a glance

| Task | Protocol | Result |
|---|---|---|
| V/A/D regression | MSP-Podcast Test1, 46k clips, speaker-independent, CCC | **V 0.705 · A 0.714 · D 0.626** |
| Emotion family (hybrid) | 175 real phone-mic clips, leave-one-speaker-out | **68.6%** (chance 16.7%) |
| Per-family recall | same | sadness 90 · joy 86 · surprise 70 · anger 69 · neutral 57 · fear 47 |
| Speaker attribution | stitched multi-speaker convos, k known | ~100% per turn |
| Adversarial robustness | 19 hostile inputs × all paths | 0 crashes · 0 NaN · explicit errors on all junk |

Full numbers, confusion patterns, and every negative result: **[WRITEUP.md](WRITEUP.md)**
· one page: **[RESULTS_AT_A_GLANCE.md](RESULTS_AT_A_GLANCE.md)**.

## Known limits (stated, not hidden)

- **No no-voice gate yet** — non-speech audio (music, noise) still receives an emotion
  label with confidence. Found by our own stress test; the obvious fix (embedding
  similarity) was tested and **refuted**; the working detector (Layer-1 voicedness) is
  scoped for the serving layer.
- **Fear is the weak family** (47% recall, drifts to sadness). Cross-speaker eval used 2
  speakers — indicative, not population-level.
- **Acted ≠ natural speech.** Phase-1 acted-dataset accuracy did not transfer; every
  headline number here is naturalistic (MSP) or real phone-mic audio.
- Weight-tuned hybrid reaches 72.6% but is tuned on the eval set — we report the honest
  **68.6%**.

## What's NOT in this repo (and why)

| Missing | Reason | If you want to reproduce |
|---|---|---|
| `data/`, MSP-Podcast audio | **Licensed, no-redistribution** (MSP) / large (RAVDESS, CREMA-D, MELD) | Request MSP access from UT-Dallas; public datasets download via `scripts/` |
| `models/` (1.2 GB WavLM-ft + adaptors) | Trained on licensed data + GitHub size limits | **Download the headline model:** [`Nikhil0097/wavlm-large-emotion-vad`](https://huggingface.co/Nikhil0097/wavlm-large-emotion-vad) → place at `models/wavlm_vad_ft/` — or retrain via `scripts/finetune_wavlm.py` (~3 h, ~$5 on an A100) |
| `own_voice/` recordings | Personal voice data of volunteers — privacy | Record your own; the pipeline is speaker-agnostic |
| `out/` artifacts | Derived from licensed data | Regenerated by the scripts |

Every result in the docs is reproducible from code in this repo + the datasets above.

## Roadmap (deliberate next steps, not loose ends)

1. **Serving layer** — one persistent process wrapping all models behind a single
   `SignalPacket` API; owns the input gates the stress test specified (voicedness,
   duration, safe model loading). *Spec already written — by the stress test itself.*
2. **Trajectory on the fine-tuned model** — windowed V/A/D over long audio → emotion
   *flow* through a conversation (the demo exists on the classical engine; the rebuild
   uses the 0.705-valence model).
3. **Two consumer projects** (separate repos): a relational-framing engine and a
   misinterpretation-recovery engine, consuming the packet — never the internals.
4. **Deferred, scoped, documented:** emotion2vec fine-tuning (~$5 GPU bet), instrumental
   music emotion (different field), text+voice fusion (breaks the "pure voice" identity —
   a choice, not a gap).

## Project 2 (same repo, separate system): Emotional TTS Steering & Fair Benchmark

An **extension project powered by this engine, not part of it.** The finished ear
turned around and pointed at a synthetic mouth: a **closed feedback loop** steers
IndexTTS-2 toward target emotions (judged by this engine, frozen), then benchmarks
the whole idea **fairly** against ElevenLabs v3, Hume Octave and OpenAI TTS —
every system given the same loop, 130 judged clips, and the headline result
retracted when a fairer test dissolved it.

**Why one repo, two projects:** the second project's judge IS the first project's
model — shipping them together keeps the benchmark reproducible. **Why they stay
apart:** the engine is frozen and additive-only; the TTS project connects through
one file ([`tts_steering/bridge.py`](tts_steering/bridge.py)), subprocess → CLI →
JSON, zero code imports in either direction, its own venvs. Delete `tts_steering/`
and the engine doesn't notice; break the engine and the bridge fails *loudly*.

→ **[tts_steering/WRITEUP.md](tts_steering/WRITEUP.md)** (the story + scoreboard) ·
[tts_steering/STEERING_LOG.md](tts_steering/STEERING_LOG.md) (chronological log) ·
[tts_steering/loop_ledger.csv](tts_steering/loop_ledger.csv) (the data).

## Repository map

| File | What's inside |
|---|---|
| [WRITEUP.md](WRITEUP.md) | The full technical report — methods, results, honest negatives |
| [ARTICLE.md](ARTICLE.md) | The narrative essay: the 0.06 → 0.35 → 0.705 valence arc, written for a general ML audience |
| [tts_steering/WRITEUP.md](tts_steering/WRITEUP.md) | **Project 2**: TTS steering loop + fair 5-system benchmark (bridge-linked, code-isolated) |
| [RESULTS_AT_A_GLANCE.md](RESULTS_AT_A_GLANCE.md) | One-page summary |
| [JOURNEY.md](JOURNEY.md) | The build story: every phase, every problem hit, every fix (31 and counting) |
| [TRAJECTORY_ENGINE.md](TRAJECTORY_ENGINE.md) | Phase-2/3 spec, binding laws, progress log |
| [REFERENCES.md](REFERENCES.md) | 21 citations, each verified against its primary source |
| `src/` | The engine: preprocessing, 111-feature extraction, classifiers, signal mapper — 160+ tests |
| `scripts/` | Every experiment: fine-tuning, hybrid fusion, diarization, stress tests — all runnable |
| `tests/` | pytest suite |

## Quick start (with your own audio)

```bash
python -m venv .venv_diar && source .venv_diar/bin/activate
pip install torch torchaudio && pip install -r requirements_diar.txt

# 16 kHz mono WAV in, emotion out (needs models/ — see reproduction table above)
python -m scripts.adaptors predict --use-case hybrid clip.wav      # family + confidence
python -m scripts.predict_wavlm_ft --inputs clip.wav --json        # V/A/D point
python -m scripts.diarize_neural --input convo.wav --out turns.csv --speakers 2
```

## Data & citation

Trained on **MSP-Podcast v2.0** (Busso et al., 2025, arXiv:2509.09791; Lotfian & Busso,
2019) under an individual academic license — cited per license terms. Supplementary:
RAVDESS, CREMA-D, MELD. Backbones: WavLM (Chen et al., 2022), emotion2vec (Ma et al.,
2024), ECAPA-TDNN (Desplanques et al., 2020). Full list: [REFERENCES.md](REFERENCES.md).

---

*Solo project by [Nikhil Jangra](https://github.com/Nikhiljangra07) — Phase 1
(March 2026) → packaged (July 2026). This engine is the signal layer of a larger
emotional-AI project; companion projects consume its output through a versioned packet.*
