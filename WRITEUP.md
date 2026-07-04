# Reading Emotion from Voice: a Dimensional (PAD) Engine + Emotion Namer

**One-line thesis:** From *how* a voice sounds (not the words), you can reliably
measure how **activated** and how **in-control** a speaker is — and, once you swap
handcrafted features for a fine-tuned self-supervised model, you can recover **how
good or bad** they feel too, which was long considered the hard, language-bound
axis. This report proves that on a gold-standard corpus, honestly, with the failure
modes named.

*LoRa Waveform Engine — Phase 2 (dimensional PAD) + Phase 3 (SSL fine-tune, emotion
naming, speaker separation). Status: complete. Portfolio / workshop project —
faithful reproduction plus an honest engineering arc, not new theory.*

---

## 1. The question

Most speech-emotion systems output one categorical label per clip ("angry"). That
throws away two things: *how strong* the emotion is, and *where it sits* on the
underlying axes of feeling. This engine instead predicts three continuous numbers —
the **PAD** model (Mehrabian & Russell, 1974; Russell, 1980):

- **Valence** — pleasant ↔ unpleasant
- **Arousal** — calm ↔ activated
- **Dominance** — submissive ↔ in-control

Each prediction is a **point in 3-D space**, named only by proximity to
**data-derived** emotion clusters (reported as a *distribution*, never a forced
label). Separately, a retrieval **namer** answers the everyday question — "which of
the six emotion families is this?" — and a diarizer answers "**who** is speaking."

**Why 3-D, not the usual 2-D valence–arousal?** Dominance separates emotions that
collide in 2-D — most importantly **anger vs fear**. We confirmed this from data
(§5.4), not assumption.

---

## 2. The engineering arc (what actually happened)

This is the spine of the work — three walls, each diagnosed and cleared:

1. **Handcrafted features measure arousal/dominance well, valence barely.**
   88 eGeMAPS + Praat + prosody → SVR: A 0.61, D 0.52, but **V 0.35** — a wall.
2. **The valence wall was the wrong *tool*, not physics.** Fine-tuning a
   self-supervised model (WavLM) end-to-end on MSP-Podcast lifted valence from
   **0.35 → 0.705** on held-out test speakers. The words weren't required; a richer
   acoustic representation was.
3. **Naming a stranger's emotion needs generalization, not a sharper in-domain
   model.** A frozen emotion-specialized model (emotion2vec) *out-generalized* our
   own fine-tuned WavLM cross-speaker; fusing the two at the decision level beats
   either alone.

---

## 3. Method

**Layered architecture** (the engine is constant; the scorer is swappable):

```
Audio → [L1: ENGINE]      111 acoustic features        (dimensional, dataset-agnostic)
      → [L2: SSL BACKBONE] WavLM-ft / emotion2vec       (learned representations)
      → [L3a: DIM HEAD]    V, A, D regression (CCC)      → the (V,A,D) triple
      → [L3b: NAMER]       retrieval / centroid          → emotion family + distribution
      → [L4: DIARIZER]     ECAPA speaker embeddings      → who spoke each turn
```

- **L1 — features (111):** openSMILE eGeMAPS (88) + Praat voice quality (13: jitter,
  shimmer, HNR, formant freq+bandwidth) + prosody (10: speech rate, pauses, energy
  envelope). Zero-NaN/Inf across 200k+ extractions; adaptive F0 range; range-checked.
  This is the *interpretable baseline* and stays useful for expression/intensity.
- **L2 — SSL backbones:** (a) **WavLM-large fine-tuned end-to-end** on MSP-Podcast
  for V/A/D — CNN feature-extractor frozen, 24 transformer layers unfrozen, CCC loss,
  bf16, OneCycle LR. (b) **emotion2vec_plus_large** used **frozen** as an embedder.
- **L3a — dimensional head:** a small MLP head on WavLM predicts the **(V, A, D)
  triple** — never collapsed to a scalar. Metric is **CCC** (AVEC standard), not
  "accuracy," because dimensional prediction is regression.
- **L3b — namer:** **retrieval** ("Shazam for emotion") — a growable database of
  (vector, family, speaker) exemplars; a new clip is named by cosine k-NN. A
  data-centroid + Mahalanobis namer is the alternative for the dimensional path.
- **L4 — diarizer:** ECAPA-TDNN speaker embeddings + clustering → per-turn speaker
  attribution. Speaker identity is emotion-invariant, so it separates talkers where
  raw acoustic-emotion features can't.

**Anti-mix law (a binding correctness rule):** vectors from different backbones live
in different spaces — cosine between a WavLM vector and an emotion2vec vector is
meaningless. Backbones are fused **only at the decision level** (average the two
6-way probability distributions), **never** by concatenating raw vectors. This is
enforced structurally: each backbone has its own tagged database and loading asserts
the tag (§5.3, `scripts/adaptors.py`).

---

## 4. Data

**MSP-Podcast v2.0** (Busso et al., 2025; Lotfian & Busso, 2019) — the field's
largest naturalistic emotional-speech corpus, accessed under an individual academic
license.

- **264,705 segments**, 3,641 speakers, natural podcast speech, 16 kHz mono.
- **Native V/A/D** on a 1–7 SAM scale, consensus = mean of ≥5 annotators.
- Speaker-independent partitions: **Train 169k / Dev 34k / Test1 46k / Test2 15k**.
- 8 categorical labels (anger, sadness, happiness, surprise, fear, disgust,
  contempt, neutral) + "other"/"no-agreement" buckets.

**Own-voice evaluation set** — **175 labeled clips, 2 speakers** (female_A: 75,
female_B: 100), 6 families (anger 29, fear 30, joy 28, neutral 30, sadness 31,
surprise 27). Phone-mic, real (non-acted) delivery. Used for the family-namer
**leave-one-speaker-out** test — the honest cross-speaker protocol. Plus stitched
multi-speaker conversations for the diarizer.

**A deliberate honesty (a binding law):** the gold standard is *empirical, not
ground truth.* Labels are a *mean of human guesses* with a disagreement noise floor.
We trust these numbers because they **generalize** (speaker-independent,
naturalistic), not because they are exact.

---

## 5. Results

### 5.1 Dimensional prediction — the headline

**Fine-tuned WavLM, MSP-Podcast, speaker-independent, CCC:**

| Split | Valence | Arousal | Dominance | Mean |
|---|---|---|---|---|
| Dev (34k) | 0.728 | 0.675 | 0.598 | 0.667 |
| **Test1 (46k, held-out speakers)** | **0.705** | **0.714** | **0.626** | **0.681** |

All three axes are strong, and **valence (0.705) is now the equal of arousal** — on
held-out speakers. For a pure-acoustic system this is a genuine result, at or above
published WavLM-on-MSP baselines.

### 5.2 The valence story — a wall that fell in two moves

Valence is *supposed* to be the hard axis (it lives in words). We hit the wall, then
cleared it:

| Stage | Valence CCC |
|---|---|
| Probe (Dev split, Ridge, handcrafted) | 0.059 — basically guessing |
| SVR + calibration, handcrafted, full 169k | 0.347 — the handcrafted ceiling |
| **WavLM-large fine-tuned end-to-end** | **0.705** — the wall falls |

Two lessons, both defensible: (1) within handcrafted features the "wall" was
**model choice** — SVR + calibration beat Ridge/RF and 6×'d the naive probe, because
the valence signal is faint and nonlinear. (2) The *real* lever was the
**representation**: a fine-tuned SSL model recovers valence from acoustics that
eGeMAPS discards. We did not need the transcript.

### 5.3 Emotion naming — retrieval + hybrid fusion (own-voice, leave-one-speaker-out)

Naming a **stranger's** emotion (their voice never in the database) — the honest test:

| Scorer | Cross-speaker accuracy |
|---|---|
| emotion2vec-kNN (frozen) | 63.4% |
| WavLM-ft + kNN | 52.6% |
| WavLM-ft + trained head | 68.0% |
| **Hybrid — emotion2vec-kNN + WavLM head (equal weight)** | **68.6%** |
| Hybrid, weight-tuned (w=0.4, *directional only*) | 72.6% |

Two findings worth stating plainly:
- **The frozen, emotion-specialized model out-generalizes our own fine-tuned one**
  (63.4% vs 52.6% on kNN). Fine-tuning WavLM on MSP made it a superb *V/A/D
  regressor* but narrowed its embedding space for *cosine retrieval* on new speakers.
  This is why the fine-tuned model is scored with a trained **head** (68.0%), not
  kNN, in the hybrid.
- **The hybrid beats either backbone alone** — the two disagree in useful ways
  (emotion2vec is stronger on neutral; WavLM-head on fear/joy/sadness), so averaging
  their distributions is a real, free gain. (chance = 16.7%)

**Per-family recall (equal-weight hybrid):**

| Family | anger | fear | joy | neutral | sadness | surprise |
|---|---|---|---|---|---|---|
| Recall | 69.0% | 46.7% | 85.7% | 56.7% | 90.3% | 70.4% |

sadness and joy are reliable; **fear is the weak point** (§6). Per-speaker: female_A
**78.7%**, female_B **63.0%** — a real generalization gap, honestly reported (§6).

### 5.4 Point → emotion separability (the overlap ceiling)

Using *true* V/A/D points (data centroids on 137k clips), nearest-centroid naming
over the 8-class set scores **42.4%** (chance 12.5%). The data-grounded map:

| Emotion | Valence | Arousal | Dominance |
|---|---|---|---|
| anger | −0.42 | 0.70 | **+0.41** |
| contempt | −0.26 | 0.59 | +0.27 |
| disgust | −0.34 | 0.59 | +0.24 |
| fear | −0.21 | 0.51 | **−0.01** |
| joy | **+0.30** | 0.58 | +0.19 |
| neutral | −0.04 | 0.45 | +0.03 |
| sadness | −0.28 | 0.39 | −0.07 |
| surprise | +0.05 | 0.64 | +0.26 |

- **joy is the lone clearly-positive-valence emotion** — *the valence problem, drawn.*
- **anger (D +0.41) vs fear (D −0.01) separate on dominance** — the 3-D payoff,
  measured not assumed.
- **neutral sits dead-center** and is the top confuser — honest, not a bug.

### 5.5 Speaker separation (diarization)

ECAPA-TDNN embeddings + clustering on stitched conversations:

- **With speaker count known** (k given): ~100% per-turn attribution on the
  2-speaker demo.
- **With automatic speaker-count estimation:** per-turn attribution 78–90% on
  2-speaker conversations, dropping to ~60% on 3-speaker — and **auto speaker-count
  estimation is unreliable** (§6). Identity separation is solid; counting talkers
  from scratch is the weak link.

---

## 6. Honest negatives (read this section)

Named failure modes strengthen the result — here they are, unhidden:

- **No no-voice gate: non-speech audio gets confident emotions.** An adversarial stress
  test (19 hostile inputs, all paths — `out/stress_test/`) showed silence, noise, and a
  synth chord all receive emotion labels (chord → sadness@0.90). Crash-safety and
  NaN-freedom held 19/19, and Layer-1 features (voiced_fraction, F0) *detect* every
  non-voice clip — the gate just isn't enforced yet. Scoped for the serving layer;
  embedding-similarity gating was tested and refuted (junk scores cosine 0.98).
- **Fear collapses to sadness on acted-style clips** — often *confidently* (the
  hybrid gets fear right only 46.7% of the time; misfires land mostly on sadness).
  On the dimensional map fear and sadness are close in V and A; on retrieval the
  learned spaces don't pull them apart on these speakers.
- **Per-speaker generalization gap:** family accuracy is 78.7% (female_A) vs 63.0%
  (female_B). Two speakers is a small, non-representative eval — treat family numbers
  as *indicative*, not population-level.
- **Fine-tuning narrowed the retrieval space.** WavLM-ft is a better regressor but a
  *worse* cosine-kNN embedder than frozen emotion2vec (52.6% vs 63.4%). Fine-tuning
  has a cost, and we measured it.
- **The weight-tuned hybrid (72.6%) is directional only** — tuned on the same tiny
  set it's reported on. The defensible number is the **equal-weight 68.6%**.
- **Auto speaker-count estimation is unreliable** — the diarizer separates known
  speakers well but mis-counts talkers; count should be supplied when possible.
- **Per-speaker feature normalization hurt dimensional prediction** (arousal −0.30,
  dominance −0.32): MSP labels are *absolute*, and a speaker's absolute vocal level
  carries A/D. Centering deletes it. (Belongs in expression/intensity, not V/A/D.)
- **Dimensional beyond ~0.7 valence** would need text fusion — deliberately out of
  scope to keep the "pure voice" identity.
- **Handcrafted-feature (Phase-1 acted-speech) numbers are retired** — they did not
  transfer; all results here are naturalistic MSP or real own-voice.

---

## 7. What this proves

> **Voice reliably reveals how activated and how in-control a person is
> (CCC ≈ 0.63–0.71). Whether they feel good or bad — the axis everyone calls
> language-bound — is recoverable from acoustics alone at CCC ≈ 0.71 once you
> fine-tune a self-supervised representation. Naming the emotion family for an
> unseen speaker reaches ≈ 69% (chance 17%) by fusing a frozen emotion model with a
> fine-tuned one — and where it fails (fear→sadness), it fails legibly.**

The substance is the arc: a diagnosed valence wall, cleared 0.35 → 0.705 by changing
the representation; a counter-intuitive finding that a frozen model out-generalizes a
fine-tuned one for retrieval; and a decision-level hybrid that respects the anti-mix
law and beats both — every claim on held-out speakers, with the failures named.

---

## 8. Reproducibility

```bash
# Dimensional — fine-tune WavLM on MSP (GPU) and evaluate on held-out Test1
python scripts/finetune_wavlm.py --labels msp/Labels/labels_consensus.csv \
    --audio-dir msp/Audios --model microsoft/wavlm-large --epochs 4 --batch 32
.venv_diar/bin/python -m scripts.eval_family_ft            # family accuracy from V/A/D

# Emotion naming — build the two adaptors + run the hybrid (isolated env)
.venv_diar/bin/python -m scripts.adaptors build-all
.venv_diar/bin/python -m scripts.adaptors predict --use-case hybrid clip.wav
.venv_diar/bin/python -m scripts.adaptors enroll --use-case hybrid \
    --label joy --speaker alex a1.wav a2.wav      # grow the DB, no retrain

# Honesty report — clip-by-clip, leave-one-speaker-out
.venv_diar/bin/python -m scripts.report_clips
.venv_diar/bin/python -m scripts.exp_hybrid               # backbone comparison + fusion
```

- Saved models: `models/wavlm_vad_ft/` (fine-tuned backbone + head),
  `models/adaptors/{wavlm_ft,emotion2vec_plus_large}/` (tagged, non-mixable DBs).
- Engine (Layer 1): `src/`, reused unchanged from Phase 1 (160 tests passing).
- Full decision log: `TRAJECTORY_ENGINE.md`, `JOURNEY.md`. Citations: `REFERENCES.md`.

---

*Backbones: WavLM-large (fine-tuned) + emotion2vec (frozen) · Engine: 111 acoustic
features · Data: MSP-Podcast v2.0 (264k segments) + 175 own-voice clips · Metric:
CCC (dimensional), cross-speaker accuracy (naming). Headline: Test1 V 0.705 /
A 0.714 / D 0.626; hybrid family ≈ 69% cross-speaker. Cite MSP-Podcast:
Busso et al. 2025 (arXiv:2509.09791) + Lotfian & Busso 2019.*
