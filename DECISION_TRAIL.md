# DECISION TRAIL — voice-emotion-engine

> Every fork in the road, in order: what was decided, why, what the data said,
> and what happened next. Companion to the chronological logs
> ([BUILD_LOG.md](BUILD_LOG.md) for the classical ear,
> [tts_steering/STEERING_LOG.md](tts_steering/STEERING_LOG.md) — 32 entries —
> for the steering campaign). Format per decision:
> **Decision → Why → Evidence → Outcome.** Dead ends are kept deliberately;
> they carry more information than the wins.
>
> Last updated: 2026-08-13. Ledger: 1,135 rows.

---

## Era 0 — The reset (March 2026)

### D0.1 Nuke the first attempt entirely
- **Decision:** delete all pre-March-2026 code; keep only the spec.
- **Why:** honest audit scored the old codebase 32% reliable — magic numbers,
  3-class labels, two disconnected experiments, librosa-only features, no log.
- **Outcome:** clean restart with the failures written into CLAUDE.md as
  warnings. The "build log from day one" culture starts here and never stops —
  it is the single most consequential decision in the repo.

### D0.2 Non-negotiables fixed before any code
- **Decision:** Ekman-6 from day one · data-driven thresholds only · one
  pipeline, zero silent drops · dataset-agnostic engine layer · every step
  verified before the next.
- **Outcome:** all seven later phases inherit these. "No magic numbers"
  eventually forces Warriner-sourced centroids, data-driven ES weights,
  data-chosen window sizes, and pre-registered experiments.

---

## Era 1 — The classical ear (Phase 1, March 2026)

### D1.1 openSMILE eGeMAPS + Praat, not librosa
- **Why:** librosa cannot measure jitter/shimmer/HNR — the voice-quality
  biomarkers that separate emotion measurement from toy demos.
- **Evidence:** 111-feature vector (88 eGeMAPS + Praat voice quality +
  prosody + formant bandwidths + energy envelope), verified per-emotion
  differentiation on RAVDESS/CREMA-D at scale (22k+ extractions, zero NaN).
- **Outcome:** the feature engine survives to this day as the classical
  instrument (Phase-1 `venv`); every scaffold experiment in Era 4 runs on it.

### D1.2 SVM/RF, no deep learning
- **Evidence:** RAVDESS-within 64.5% (random 16.7%); hyperparameter tuning
  moved it <0.5pp → the ceiling was the data, not the model.
- **Outcome:** correct MVP call — and the ceiling finding justified the later
  pivot to a pretrained speech backbone rather than more classical tuning.

### D1.3 The finding that steered everything after: categories don't travel
- **Evidence:** train-RAVDESS→test-CREMA-D collapsed 64.5%→30.3% (sadness
  absorbed everything); own-voice test 20.7% (185 real files, 3 speakers);
  feature-stability report: 72/111 features drift >30% across datasets,
  MFCCs drift 200%, while pause-ratio/formants drift <3%.
- **Outcome:** the engine (Layer 1) was proven dataset-agnostic; the
  categorical classifier was proven domain-bound. This asymmetry — signal
  generalizes, labels don't — becomes the recurring theme of the entire
  project (it reappears in Era 5's real-world test almost verbatim).

---

## Era 2 — The pivot to dimensions (June 2026)

### D2.1 PAD trajectory engine as the anchor (TRAJECTORY_ENGINE.md, 2026-06-22)
- **Decision:** predict continuous Valence/Arousal/Dominance, then points →
  trajectories → an emotion web; categories become a naming layer on top.
- **Why:** Era 1 proved categories collapse across domains; dimensions are
  the transportable substrate. Binding laws set: **CCC not accuracy · keep
  the (V,A,D) triple, never a scalar · data-grounded centroids only ·
  validate on natural speech · report distributions.**
- **Outcome:** the origin PAD-plane notebook sketch is committed as an
  artifact; every instrument since speaks V/A/D.

### D2.2 Fine-tune WavLM for V/A/D; derive centroids from MSP labels
- **Evidence:** WavLM-large fine-tuned on MSP-Podcast, held-out Test1 CCC:
  **V 0.705, A 0.714, D 0.626**. Centroids derived (not guessed) from Train
  labels via ((EmoVal−4)/3, (EmoAct−1)/6, (EmoDom−4)/3): sadness
  (−0.28, 0.39, −0.07), joy (+0.30, 0.58, +0.19), anger (−0.42, 0.70, +0.41),
  surprise (+0.05, 0.64, +0.26), neutral (−0.04, 0.45, +0.03) — later fear
  (−0.21, 0.51, −0.01, n=794) and disgust (−0.34, 0.59, +0.24, n=1325) with
  the same formula.
- **Constraint honored:** MSP-Podcast is licensed, no-redistribution real-
  speaker data. **Statistics-only use; it must never train the mouth.**

### D2.3 Two naming adaptors, structurally non-mixable
- **Decision:** separate retrieval databases per backbone — WavLM-ft
  (enrolled use-case, 80.6% in-domain) and emotion2vec+ large (stranger
  use-case, 63.4% generalization; hybrid fusion 68.6% LOSO) — with hard
  anti-mix guards (vector-space dimension checks, tagged directories).
- **Why:** cosine similarity between vectors from different backbones is
  meaningless; making the mistake impossible beats documenting it.

---

## Era 3 — The mouth and the loop (Project 2)

### D3.1 Two projects, one thin bridge
- **Decision:** the engine repo stays additive-only; `tts_steering/` connects
  exclusively through `bridge.py` subprocess calls; vendor code (IndexTTS-2)
  is read but never modified.
- **Why:** the ear must remain a valid measuring instrument for the mouth —
  coupling them would destroy the evidential value of every verdict.

### D3.2 The judge-frozen law (absolute)
- **Decision:** loop data may improve the mouth's control policy, **never**
  retrain the ear/judge. Anti-circularity: steering signal (WavLM V/A/D) and
  verdict (emotion2vec family) live in different vector spaces.
- **Outcome:** held without exception through 1,135 ledger rows. When the
  judge needed a disgust class (Era 5), the answer was a *second* database
  (judge-v2), not a modified first.

### D3.3 IndexTTS-2 as the mouth, after a paid bake-off
- **Evidence:** ledger carries fair-generation campaigns for chatterbox,
  ElevenLabs, Hume Octave, and IndexTTS-2. IndexTTS-2 won on: local ($0),
  8-dim emotion vector interface, zero-shot voice cloning, steerability.
- **Outcome:** everything after runs on a laptop for $0. Two later "should
  we rent the RTX PRO 6000?" questions were both answered **no** — the
  feasibility spike (D5.1) vindicated this.

### D3.4 Every clip a ledger row; misses kept
- **Decision:** `out/loop_ledger.csv` records every synthesized-and-judged
  clip with its control string, V/A/D, verdict, confidence, distance, hit.
- **Outcome:** the ledger IS the science. It is what allows claims like
  "zero disgust verdicts in 911 rows was structural" to be checked in
  seconds.

---

## Era 4 — The steering campaign (P4.6–P4.14, 2026-08-08 → 08-11)

### D4.1 MSP is the ruler (A/B/C proposer test)
- **Decision:** three proposal arms competed to steer the mouth toward MSP
  centroids — A: deterministic rules; B: DeepSeek R1 (via OpenRouter, keys
  never printed/committed); C: the user's scaffold equation. Matching the
  MSP-derived target = win, because MSP is the only natural-speech ground
  truth in the room.
- **Incidents (kept honest):** R1's chain-of-thought arrived in a separate
  `reasoning` field → initial parse got 0 vectors; fixed and the tainted
  proposals deleted so arm B wasn't unfairly forfeited. Arm C suffered
  proposal starvation (dedup/clamp degeneration) — disclosed as floor, not
  ceiling.
- **Outcome:** no premature retirement of arm C (explicit user decision:
  "conduct more tests and find the best fit") → the scaffold ladder below.

### D4.2 The scaffold ladder — test the user's equation properly
- **Steps, each gated at $0:** Fisher-selected 32 dims → robust median/IQR
  z-space → family−neutral directions → contrast directions (family−rival)
  → weighted cosine scoring.
- **Key negative finding:** the mouth's hot knobs are **91–95% collinear**
  in classical feature space — static features cannot distinguish what the
  knobs do.

### D4.3 Fourier fingerprints WITH derivatives (after the criticism)
- **Trigger:** user: *"I'm just having a feeling you are not doing the work
  properly. So dig deep, be meticulous."* The first pass had summarized;
  the second pass built literal DCT-II contour equations (K=10, N=100
  time-normalized), derivation IN the equations (W_deriv = k² weighting).
- **Evidence:** derivative weighting broke the knob collinearity (Gate 0
  PASS under W_deriv: 0.76/0.61/0.64); expansion across corpora then showed
  acted↔natural sadness fingerprints **anti-correlated (−0.46)** and the
  mouth to be a "third dialect" — fingerprints are not universal.
- **Outcome:** fingerprints demoted from steering mechanism to diagnostic;
  the real lesson (text-sensitivity, non-universality) fed P4.6.

### D4.4 3-axis epicycles and the wavelet ear (exploratory gates)
- **Evidence:** cross-axis rotation invariants produced a surprise detector
  at 69% held-out recall (parked as ear-v2 candidate); the wavelet/Gabor
  micro-gesture gate FAILED for fear (−3.8pp, tremor hypothesis refuted) and
  passed modestly for sadness (+7.7).
- **Outcome:** both logged as gates with verdicts; neither promoted. The
  discipline of "explosive steps, verified" holds even for pet ideas.

### D4.5 The joy campaign
- **Path:** joy spike 0/6 → survey of other engines → EL clip forensics →
  the user's hunch ("joy and anger both high pitch… a little difference can
  do the job") → anger–joy axis analysis (Cohen's d on 60k clips) →
  **joy = anger minus tension** (all knobs move anger-ward; happy −3.39/unit).
- **Outcome:** the subtract-tension idea that later becomes the P5A smoke
  record and ultimately the subtraction identities of Era 5.

### D4.6 The P4.6 transfer map — sentence-conditionality discovered
- **Design:** 19 sentences × 11 configs, every cell judged.
- **Evidence:** anger config a08c02 19/19 (universal); surprise 16–17/19;
  JOY unlocked (6 verdicts, 3 routes) but **only on warm sentences**;
  SADNESS unlocked (10 verdicts, mel 1.0 + calm 0.3) but smearing onto warm
  sentences; zero-vector baseline non-neutral 7/19.
- **Outcome:** first full Ekman scoreboard, and the seed of the congruence
  law (D5.10).

### D4.7 Human blind session #1 → VOID, by its own controls
- **Evidence:** the user's answers mirrored the machine's anger↔surprise
  confusion; the built-in controls flagged the session as unratifiable
  (2–3s fragments, too thin). User's own caveat honored: "I suggest you not
  to blindly trust it."
- **Outcome:** VOID applied honestly; the redesign (same-text long clips +
  consistency duplicate + sealed key) becomes D5.8.

### D4.8 The trajectory (P4.14) — the window size is data-chosen
- **Design:** the mouth writes a 4-act journey; the ear reads it back at
  window sizes {1.5, 2, 3, 4}s (50% overlap); selection by correlation
  between commanded and measured V/A.
- **Evidence:** **3.0s wins (corr V +0.93, A +0.90).** Graph committed
  (docs/p414_trajectory_graph.png).
- **Outcome:** 3.0s/50% becomes the standing protocol (used unchanged on the
  1943 broadcast in D5.11).

### D4.9 Standalone product framing (2026-08-10)
- **Decision (user):** this repo is NOT part of LoRa — a whole separate
  product. Saved to persistent memory; all documentation framed accordingly.

---

## Era 5 — Phase 5: optimization in the mouth's native space (08-11 → 08-12)

### D5.1 Feasibility spike before any GPU spend
- **Evidence:** IndexTTS-2 ships inference only — no trainer, no loss. The
  entire emotion pathway reduces to ONE embedding:
  `emovec = Σ wᵢ·bank_i + (1−Σw)·speaker_emovec`, banks = plain tensors in
  the checkpoint.
- **Decision:** "fine-tuning" = **searching emovec space directly** with the
  frozen judge as objective. Phase 5A local, $0. LoRA (Maya1, $5–15 on a
  cheap GPU) demoted to 5B fallback. RTX PRO 6000: never needed.
- **Correction logged same day it mattered:** P4.12's "hidden bias" claim
  (normalize_emo_vec: happy×0.9375 … Σ≤0.8 cap) is **webui-only** — our
  inference path always delivered raw vectors. "Error mine; corrected the
  same day it mattered."

### D5.2 P5A smoke — signed coefficients work; synthesis is stochastic
- **Evidence:** `happy +0.60, angry −0.30` → d=0.096 to the joy centroid —
  the closest any clip had come to any centroid in 399 rows. But the parity
  clip (happy 0.35, s07) FLIPPED verdicts vs P4.6 → GPT sampling
  (do_sample, T=0.8) makes every single-clip verdict one draw.
- **Decisions forced:** all optimization ≥2 draws/candidate; multi-draw
  confirmation required for headline claims; hull constraint justified
  (happy 1.10 blew up arousal → surprise).

### D5.3 The P5A protocol (pre-registered, identical across emotions)
- **Design:** seeded (μ=3, λ=8) evolution strategy, 10 generations, 2 draws
  (two train sentences), score = 2·conf·(judge==target) + max(0, 0.5−dVAD),
  hull: coef ∈ [−0.6, 1.0], Σ|c| ≤ 1.6, Σc ∈ [−0.2, 1.1], deterministic
  seed → resumable clip cache, every clip ledgered, **held-out one-shot on
  never-seen sentences with a control, zero iteration.** Gate-4 mitigations:
  train/held-out split, dual instruments, hull, eval cap, human final gate,
  judge never retrained.
- **Engineering lesson:** a transient judge failure killed the first joy run
  at gen 5 → retry-with-cooldown added; resume-from-cache lost zero clips.

### D5.4 P5A results — four runs, four subtraction identities
| run | held-out | best vector shape | discovery |
|---|---|---|---|
| joy | 4/6, repeatable, control clean | happy +0.41, **−angry −mel −surp** | joy = subtract everything dark; V=+0.96 all-time record; text-conditional |
| sadness | 1/6 verdict, ~6/6 on-centroid (d=0.089 2nd all-time) | mel +0.78, calm +0.21 | acoustics generalize, categorical verdict needs a draw-level prosodic event; unclamped route smears the control — intensity buys smearing, not verdicts |
| fear | 4/6 + control 2/2 (text-independent) | sad +0.32, afraid +0.30, **−angry −calm** | subtract anger = dominance kill; "fear blind spot" was a never-seeded bank |
| disgust | 2/6 strict, 6/6 rejection-family | disg +0.36, **−sad −surp** | faintest emotion; judge structurally lacks the class |

### D5.5 Judge-v2 — an instrument added, not a judge changed
- **Trigger:** `FAMILIES` hard-codes 6 classes; zero disgust verdicts in 911
  rows was architecture, not acoustics.
- **Decision:** second retrieval DB (models/adaptors_v2/): v1's 175
  exemplars copied + 36 **dataset** disgust exemplars (24 RAVDESS actors +
  12 CREMA-D speakers — never mouth clips). V1 untouched byte-for-byte.
- **Evidence:** d02 held-out both draws: anger@38% → **disgust@100%** —
  exactly the two clips WavLM had flagged; **12/14 verdicts unchanged**
  (no collateral, no over-claiming of contempt clips).

### D5.6 Live demo — the loop closes on a human voice
- **Evidence:** user's recording → ear: joy@80% → reply synthesized in the
  user's cloned voice with the P5A joy vector → frozen judge on the reply:
  joy@80%, V=+0.86. Second exchange logged honestly: zero-vector neutral
  reply drifted to sadness@60%.

### D5.7 (ordering note) The blind pack and congruence gate below occurred
  interleaved with D5.5–D5.6; order in the log is authoritative.

### D5.8 Human blind gate #2 — the one that counts
- **Design fixes from the VOID:** 8 long clips, ONE emotionally-neutral
  paragraph for all (no text leakage), duplicate joy pair as consistency
  control, sealed key with machine verdicts recorded unseen.
- **Evidence:** **human 4/8 > machine 3/8.** Controls PASSED (identical
  labels on the duplicate pair; neutral control correct; human hits align
  with machine confidence). Anger + surprise human-ratified; **the human
  heard sadness where the categorical judge said neutral** — direct human
  confirmation of the sadness finding; joy's flat-text failure matched its
  known text-conditionality.
- **Verdict:** PARTIAL PASS — first counting human ratification.

### D5.9 The user's insight → a testable law
- **User:** "context matters a lot." Formalized as the carrier/resonance
  hypothesis and pre-registered (predictions committed BEFORE synthesis).

### D5.10 The congruence gate — law confirmed, accuracy doubles
- **Evidence:** 6 emotions × {flat, congruent} × 2 draws. Interaction
  **+0.67** (resonance +1.00, carrier +0.33). Long-form accuracy
  **4/12 → 8/12**. Joy fully rescued (0/2 → 2/2). Fear **reclassified**:
  short bursts carry on flat text, sustained fear needs narrative fuel
  (duration-dependent boundary). Surprise showed congruent text can HURT a
  carrier (one draw → anger).
- **Codified law:** resonance emotions (joy, sadness, disgust, long-form
  fear) are steered vector + congruent text together; carriers (anger,
  surprise, short-burst fear) by vector alone.

### D5.11 Real-world test — found audio, free ground truth
- **Decision:** "Sorry, Wrong Number" (Suspense, 1943, public domain) chosen
  BECAUSE its emotional arc is documented history — no hand labels needed.
- **Evidence (1,196 windows, 3.0s/50%):** the dimensional ear located the
  overheard-murder shock at 4–6 min (V dive to −0.85 raw, A 0.57→0.79),
  sustained the desperate-calls tension (A 0.74–0.81), put the angriest
  window (V=−0.91, A=1.00) at 18.1 min, and found a fear cluster at the
  28.3–29 min climax. The categorical judge FAILED out-of-domain (51% joy —
  1943 AM broadcast vs modern close-mic enrollment). Closing music washed
  the terminal climax out of 5-min medians.
- **Outcome:** Era 1's asymmetry confirmed on found audio: **dimensional
  signal generalizes; the categorical layer needs domain enrollment** (the
  judge-v2 mechanism is exactly that tool). Graph:
  docs/real_world_trajectory.png.

---

## Era 6 — The cloned-voice prototype (08-13)

### D6.1 Full speech in the user's voice → the NINTH BANK
- **Evidence:** 7-act story from an 11s user memo: 4/7 certified. Both joy
  acts failed with the prompt's own melancholy profile (A stuck at the
  prompt's 0.19). Mechanism identified in the vendor merge: speaker residue
  is weighted **(1−Σw)**, and the P5A joy vector's Σw = **−0.06** → it
  amplified the prompt's emotion at 1.06×. Carriers (Σw 0.8–1.1) were
  immune. **The P5A vectors are prompt-conditional; the speaker prompt is a
  hidden ninth bank.**
- **Cure, predicted then confirmed next day:** joy via the emo_audio
  reference channel (bypasses the Σw arithmetic) → 3-min single-file
  version: **5/7, reference-joy 2/2 (joy@100%, joy@80%)** — first joy ever
  certified in the user's own voice. Anger slipped to surprise on one draw
  (known boundary noise). Diagnosed Tuesday, cured Wednesday.

### D6.2 Register-bound identity (the user's ear finds the next law)
- **User's observation:** the calm first minute "sounds like I'm listening
  to myself"; the shouting acts are "a different person."
- **Analysis:** an 11s single-register prompt bounds which of the speaker's
  voices the model knows; high-arousal acts extrapolate outside it and lose
  identity. Identity is register-dependent.
- **Standing fix (not yet run):** multi-register prompt bank — user records
  calm/loud/soft memos; each act uses the register-matched prompt.

### D6.3 The narration — inheritance as a feature
- **Decision:** for a "sounds like me" deliverable, stay in the prompt's
  register and use the zero vector (emovec = pure speaker inheritance =
  maximum identity).
- **Evidence:** 1.86-min piece; all five paragraphs judged at the prompt's
  own profile (A 0.14–0.20) — register held across ~350 words.
- **User verdict:** "good for a prototype… ElevenLabs has much better
  cloning but THIS is what we made."

---

## Standing laws (consolidated)

1. **Judge frozen. Ever.** Instruments may be *added* (judge-v2); never
   changed.
2. **Anti-circularity:** steering space (WavLM V/A/D) ≠ verdict space
   (emotion2vec). Enrollment exemplars for any judge come from datasets,
   never from the mouth.
3. **Every clip a ledger row.** Misses kept. Pre-register before running;
   commit predictions before synthesis when possible.
4. **No magic numbers:** centroids from labels, windows from correlation,
   weights from importance, thresholds from data.
5. **Congruence law:** resonance emotions (joy, sadness, disgust, sustained
   fear) = vector + congruent text; carriers (anger, surprise, burst fear)
   = vector alone (text may interfere).
6. **Ninth-bank law:** speaker-prompt residue is weighted (1−Σw); for
   arbitrary voices keep Σw ≥ ~0.8, or use the reference channel, or
   re-optimize per voice. Zero vector = pure inheritance (max identity,
   zero emotional control).
7. **MSP-Podcast:** statistics only; never trains the mouth; never leaves
   the machine.
8. **Privacy:** own_voice/ recordings never published; keys live only in
   gitignored .keys.env; secret sweep before every commit; repo visibility
   is the user's decision alone.
9. **Vendor untouched; engine repo additive-only; coupling via bridge.py
   subprocess only.**

## Open decisions (queued, none taken)

- Multi-register prompt bank (fixes register-bound identity; needs ~40s of
  user recordings).
- Auto-retry certification loop (generate → judge → retry until certified;
  turns draw noise into a product non-issue).
- Stable neutral recipe (zero vector inherits/wobbles — the last emotion
  without a proven vector).
- Judge sadness enrollment via the v2 mechanism (natural-speech exemplars).
- Warm-text joy human trial (machine went 2/2; a human should hear it).
- Domain-enrollment recipe productization (per-deployment judge-v2).
- Real-time latency path (only item that costs money).
- Phase 5B (Maya1 LoRA, $5–15) — only if 5A saturates; it has not.

## Instrument & artifact index

- **Ear (dimensional):** `scripts/predict_wavlm_ft.py` — WavLM-large ft on
  MSP (CCC V 0.705 / A 0.714 / D 0.626), PAD centroids inline.
- **Judge v1 (frozen):** `scripts/adaptors.py` + `scripts/retrieval_namer.py`
  — e2v kNN, 175 exemplars, 6 families. **v2:** `models/adaptors_v2/` (+36
  dataset disgust, 211 exemplars) via `tts_steering/judge_v2.py`.
- **Bridge:** `tts_steering/bridge.py` (wavlm_vad, e2v_family, judge).
- **Mouth worker:** `tts_steering/synth_worker.py` (vector + emo_audio
  branches), vendor at `tts_steering/vendor/index-tts` (untouched).
- **P5A optimizers:** `p5a_joy_opt.py`, `p5a_sad_opt.py`, `p5a_fear_opt.py`,
  `p5a_disgust_opt.py` (+ `p5a_smoke.py`).
- **Gates:** `blind_pack_p5a.py` (human), `congruence_gate.py` (law),
  `real_world_traj.py`/`real_world_plot.py` (found audio),
  `trajectory_p414.py` (window selection).
- **Prototype:** `live_demo.py`, `user_voice_speech.py`,
  `user_voice_speech_3min.py`, `user_narration.py`.
- **Ledger:** `tts_steering/out/loop_ledger.csv` (live) /
  `tts_steering/loop_ledger.csv` (committed snapshot) — systems include:
  chatterbox*, elevenlabs*, hume-octave*, indextts2, -p43, -p46, -abc[ABC],
  -joyref, -joyspike, -p5a-{smoke,joy,sad,fear,disgust,sad-diag,blind},
  -congruence, -live-demo, -user-voice-{speech,3min}, -user-narration.
- **Graphs:** `docs/p414_trajectory_graph.png`,
  `docs/real_world_trajectory.png`.
- **Environments:** `venv` (py3.13 classical stack) · `.venv_diar` (engine/
  judge models) · `.venv_tts` (orchestrators/bridge) · `vendor/index-tts/
  .venv` (py3.11 synthesis).
