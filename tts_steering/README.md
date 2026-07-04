# TTS Steering & Benchmark — closed-loop emotional speech (Phase 4, isolated)

> **Status: charter (July 2026).** Model selected, protocol drafted, no synthesis code yet.
> This folder is **isolated by law** from the Voice Emotion Engine: no imports from
> `src/` or `scripts/`, ever. The engine is consumed only as a black-box *pipeline*
> (subprocess/CLI → JSON), exactly as an external user would call it.

---

## 1. The idea (one paragraph)

The Voice Emotion Engine reads emotion from audio (V/A/D CCC 0.705 held-out; family
naming ≈69% cross-speaker). This project runs it **in reverse**: take an open,
*controllable* TTS, and steer its emotion controls in a **closed loop** — generate →
measure with our engine → adjust → repeat — until the audio *measurably* delivers a
target emotion. Then benchmark: our steered open model vs. frontier commercial TTS
(their emotion presets/tags), all scored by the same open measurement engine. We are
not competing with frontier TTS and not claiming novelty — this is a
read-the-papers-and-apply project with one honest edge: **most emotional TTS is
open-loop (render "sad" and hope); ours closes the loop with a calibrated judge.**

## 2. Scope

- **Emotions (3, data-chosen):** sadness (judge recall 90%), joy (86%), anger (69% +
  separates on the dominance axis we uniquely measure). **Fear is excluded** — it is
  the judge's known blind spot (47%); judging what you can't measure is self-sabotage.
- **Deliverables:** (a) steering loop demo, (b) benchmark table: steered-open vs
  commercial emotional TTS, per-emotion V/A/D fidelity + family hit-rate, (c) writeup
  with honest limits.
- **Non-goals:** training a TTS, real-time synthesis, covering all emotions,
  "first-ever" claims.

## 3. Foundation model decision (researched 2026-07-04)

**Primary: IndexTTS-2** (Bilibili, open-sourced 2025-09).
- **Why:** the only open model with an explicit **8-dim continuous emotion vector**
  — `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`, each
  slider 0–1 — i.e., a machine-searchable control space that maps almost 1:1 onto our
  judge's families. It also **disentangles speaker timbre from emotion**, so we can
  hold the voice constant and vary *only* emotion — clean experimental control.
- **API:** `tts.infer(spk_audio_prompt=..., text=..., emo_vector=[8 floats], ...)`.
- **Hardware:** CUDA-first; runs on Apple Silicon MPS with tweaks (disable DeepSpeed,
  ~5.9 GB weights) or a cheap RunPod GPU.
- **License (honest):** code Apache-2.0, but **weights are custom-licensed:
  commercial use requires Bilibili's written authorization.** This is a
  non-commercial research/portfolio benchmark → compliant. Documented, not hidden.

**Secondary (and license-safe backup): Chatterbox / Chatterbox-Turbo** (Resemble AI).
- **MIT license end-to-end** (weights included). First open TTS with an
  **`exaggeration` knob** (0 = monotone → 1+ = theatrical) + `cfg_weight`, plus
  voice cloning from ~5 s reference — steerable by *choosing/perturbing emotional
  reference clips* (we have labeled corpora for that). Coarser control (2 knobs vs 8
  sliders) but bulletproof license and strong blind-test quality. Included as a second
  open system in the benchmark regardless.

**Research-angle third (optional): StyleTTS 2** (MIT).
- Continuous style-vector space with documented **emotion clusters**; 2026 literature
  directly applicable: task-vector arithmetic for expressivity control
  (arXiv:2606.05367), cross-speaker emotion transfer via style latents
  (arXiv:2303.08329). The strongest "applied a paper, added our thinking" exhibit if
  time allows.

**Rejected for this use:** Kokoro (fast but minimal emotion control), Orpheus/Maya1
(tag-based — discrete tags can't be swept by an optimizer; fine as benchmark
*subjects*, wrong as the steering foundation), F5-TTS/Spark-TTS (weaker fit:
cloning-oriented / attribute tokens; CosyVoice 3 is Apache-2.0 and may join the
benchmark as an open baseline).

## 4. Benchmark opponents (frontier, mid-2026)

ElevenLabs v3 (audio tags, GA 2026-02) · Hume Octave 2 (self-adapting emotional
delivery) · OpenAI TTS (instruction-based) · optionally Cartesia Sonic 3.5 /
Gemini Flash TTS. Small API spend; each gets the same sentences and target emotions.

## 5. Protocol (decided BEFORE building — the "CCC discipline")

1. **Fixed, emotionally-neutral sentences** (RAVDESS-style: "the table is in the
   room") — otherwise we measure the words, not the voice. Same sentences for every
   system.
2. **Judge/steerer separation (anti-circularity law):** the loop is steered by ONE
   backbone and scored by the OTHER (fine-tuned WavLM steers, frozen emotion2vec
   judges — or vice versa; they live in different vector spaces by the engine's
   anti-mix law). Optimizing and grading with the same model would be Goodhart bait.
3. **Human anchor:** small listening check (~5 listeners × ~20 clips) to validate the
   judge against reality. Judge-vs-human gaps are *findings*, reported.
4. **Metrics:** distance to target-emotion V/A/D centroid; family hit-rate@1;
   judge confidence; per-emotion breakdown. No single vanity number.
5. **Optimizer:** black-box search over the 8-dim emotion vector (grid → CMA-ES if
   needed). Budgeted; iterations logged.
6. **Report negatives** exactly as the parent project does (fear exclusion, judge
   bias toward acted speech, license constraints).

## 6. Isolation contract (binding)

- **No code imports** from the engine. The judge is called as a black box:
  `.venv_diar/bin/python -m scripts.predict_wavlm_ft --inputs clip.wav --json`
  (and the adaptors CLI for family) — later replaced by the serving layer's
  `SignalPacket` when it exists. If the pipe shape changes, this project only ever
  touches the *call site*, never engine internals.
- **Own environment** (`.venv_tts/`, created when code starts) — TTS deps never touch
  the engine's venvs.
- **Own artifacts** (`tts_steering/out/`, gitignored when large/derived).
- Engine repo remains additive-only; this folder is the only place Phase-4 work lives
  until it graduates to its own repository.

## 7. Papers this project reads *and applies*

- IndexTTS2 (arXiv:2506.21619) — the foundation's emotion-disentanglement design.
- StyleTTS 2 (arXiv:2306.07691) — style-latent space; emotion clusters.
- Task-vector arithmetic for TTS expressivity (arXiv:2606.05367, June 2026).
- Cross-speaker emotion transfer via style latents (arXiv:2303.08329).
- Plus the parent project's PAD/CCC foundations (see ../REFERENCES.md).

## 8. Phases

1. **P4.0 — this charter.** ✅
2. **P4.1 — judge harness:** wrap the engine CLI as the scoring function; verify on
   labeled clips (sanity: known-sad clip scores sad).
3. **P4.2 — foundation online:** IndexTTS-2 running (MPS or pod); manual emotion-vector
   sweep; listen + measure.
4. **P4.3 — closed loop:** optimizer over the emotion vector against the judge.
5. **P4.4 — benchmark:** open + commercial systems, same protocol; tables + plots.
6. **P4.5 — writeup:** results, negatives, judge-vs-human gap.
