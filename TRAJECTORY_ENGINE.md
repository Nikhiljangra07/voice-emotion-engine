# TRAJECTORY_ENGINE.md — The Anchor (Phase 2)

> **This is the active priority.** Read this before touching code. It is the
> single source of truth for what we are building in Phase 2 and the laws that
> keep us from drifting. CLAUDE.md remains the Phase 1 spec + decision log;
> this file governs everything from here.
>
> **Created:** 2026-06-22 · **Phase:** 2 (PAD Trajectory Engine) → 3 (deep audio). · **Status:** ✅ COMPLETE + valence wall broken. Phase 2 study shipped (classical V 0.347 / A 0.612 / D 0.515, trajectory + emotion web + writeup). **Phase 3 headline (2026-06-30): fine-tuned WavLM-large, held-out MSP Test1 V 0.705 / A 0.714 / D 0.626 / mean 0.681 — valence 0.10→0.705 (7×), the "valence problem" solved.** Model at `models/wavlm_vad_ft/`, inference via `scripts/predict_wavlm_ft.py` (.venv_diar). **Phase 3.2 (2026-07-01): emotion namer (retrieval + frozen/fine-tuned hybrid ≈ 69% cross-speaker) + ECAPA diarization, packaged as routable enrollable adaptors — writeup shipped (`WRITEUP.md`, `RESULTS_AT_A_GLANCE.md`, `REFERENCES.md`).** Project packaged.

## PROGRESS LOG
- **2026-07-01 — ✅ PHASE 3.2: EMOTION NAMING + HYBRID + DIARIZATION + PACKAGED.** Built the
  everyday layer on top of the V/A/D engine: "which family, and who said it." **Retrieval namer**
  (`scripts/retrieval_namer.py`) — name a clip by matching its full embedding (cosine k-NN) against
  a growable exemplar DB, not by squeezing to V/A/D; this rescued fear (V/A/D naming 0% → 73%
  in-domain). **Counter-intuitive, measured result:** frozen **emotion2vec-kNN 63.4%** beats our
  fine-tuned **WavLM-kNN 52.6%** cross-speaker (leave-one-speaker-out, 175 own-voice clips, 2
  speakers) — fine-tuning made WavLM a great *regressor* but narrowed its *retrieval* space; on
  WavLM the fix is a trained **head** (68.0%), not kNN. **Hybrid** (`scripts/exp_hybrid.py`) fuses
  the two at the DECISION level (average the 6-way distributions — never mix vector spaces;
  anti-mix law): **68.6% equal-weight**, beating either alone (per-family: sadness 90 / joy 86 /
  surprise 70 / anger 69 / neutral 57 / **fear 47** = weak point; a weight-tuned 72.6% is
  directional only). **Packaged** in `scripts/adaptors.py`: two tagged **non-mixable** DBs
  (`models/adaptors/{wavlm_ft,emotion2vec_plus_large}/`), a routable **`--use-case hybrid`** mode,
  and an **enroll-a-speaker** path that grows both DBs + refits the head in <1s (no retraining) —
  one command: `.venv_diar/bin/python -m scripts.adaptors predict --use-case hybrid clip.wav`.
  **Diarization** (ECAPA-TDNN): ~100% per-turn attribution with speaker count known; 78–90% (2
  spk) / ~60% (3 spk) with auto-count (counting is the unreliable link, not identity). **Honest
  negatives kept:** fear→sadness (often confident), per-speaker gap 78.7% vs 63.0%, fine-tuning's
  retrieval cost — all in `WRITEUP.md` §6. **Packaging shipped:** rewrote `WRITEUP.md` with real
  numbers, added one-page `RESULTS_AT_A_GLANCE.md`, 21 primary-source-verified citations in
  `REFERENCES.md`. All numbers re-verified live from cached embeddings this session.
- **2026-06-30 — ✅ FINE-TUNING COMPLETE. VALENCE WALL BROKEN — the project's headline result.**
  Fine-tuned WavLM-large end-to-end on MSP-Podcast (full 169k Train, 4 epochs, CCC loss) on a
  rented RunPod **A100-SXM 80 GB** (~3 h wall-clock, ~$4.50). **Held-out Test1 (46k clips, never
  seen during training): valence CCC 0.705 / arousal 0.714 / dominance 0.626 / mean 0.681.**
  (Dev peaked epoch 3: V 0.728 / A 0.675 / D 0.598.) **The valence arc, end to end:
  0.10 raw classical → 0.35 SVR+calibration → 0.44 frozen-WavLM fusion → 0.705 fine-tuned**
  — a 7× lift on the held-out set, SOTA-competitive on MSP-Podcast. The "valence problem" that
  defined Phase 2/3 (valence "lives in words, weak from voice") is **solved**: valence from
  audio alone is no longer the weak axis (it now leads with arousal). Model pulled local →
  `models/wavlm_vad_ft/` (config.json + head.pt + 1.2 GB model.safetensors + train.log).
  Local inference wired: `scripts/predict_wavlm_ft.py` runs in **.venv_diar** (torch 2.12 +
  transformers 5.12 = exact pod match, MPS): AutoModel backbone + head.pt; audio → V/A/D in
  [0,1] → ×6+1 = 1–7 → PAD → inline CentroidNamer over the 8-class gold set. **Own-voice reality
  check:** valence SIGNS now correct on real speech (happy +0.53→joy ✓, sad −0.21→sadness ✓;
  the Phase-1 all-sadness collapse is GONE), but arousal is compressed on amateur ACTED phone
  clips so anger/fear slump toward sadness — an acted-vs-natural domain gap, not an engine fault
  (the held-out MSP number is the trustworthy one). **Ops lessons (data onto pod):** portal→pod
  `wget` is the compliant path (never route licensed audio local→pod through the assistant);
  **stream-extract** `wget -O - | tar xz` (the 37 G tarball + 44 G extracted together overflow
  the ~70 G volume quota — never store both); raise `ulimit -n` + `mp.set_sharing_strategy(
  "file_system")` to survive the 34k-sample eval DataLoader fd limit; launch unbuffered
  `PYTHONUNBUFFERED=1 setsid nohup …` so logs stream live and the run survives SSH drops. Pod
  must be TERMINATED in RunPod once the model is pulled (everything needed is local) to stop
  the ~$1.51/hr billing; leave the separate A40 (Granite) pod alone. **Next options:** rebuild
  trajectory + multi-speaker on the fine-tuned model; scale/tune further; or write it up.
- **2026-06-29 — FUSED ENGINE INTEGRATED into main pipeline (Phase 3 begun). Lift HOLDS at
  scale + sadness bias gone.** Graduated SSL fusion from probe to a real, integrated engine —
  SAFELY: torch/WavLM stay in `.venv_diar`; main pipeline orchestrates via subprocess/cached
  .npy (no torch in main venv; 170 tests still green). New: `scripts/extract_ssl.py` (batch
  WavLM, isolated), `scripts/train_fused.py` (fuse classical+SSL, train, save —
  `models/dim_fused_msp`), `scripts/embed_files.py` (arbitrary-clip WavLM, isolated),
  `scripts/predict_fused.py` (end-to-end: classical[main] + WavLM[subprocess] → fused → V/A/D
  + emotion). **Held-out Test1 (5k), trained on 15k:** classical V 0.186/A 0.663/D 0.562 (mean
  0.471) → **FUSED V 0.444/A 0.717/D 0.613 (mean 0.591)**. The fused engine on **15k beats our
  classical engine on 169k** on every axis (esp valence 0.44 vs 0.35). Valence wall broken
  (0.10 raw classical → 0.44 fused). End-to-end on real own-voice clips: 001→joy, 008→neutral,
  003→neutral with sensible V/A/D — **the Phase-1 all-sadness bias is GONE**. Next (user-flagged
  "language model" discussion): fine-tune WavLM (frozen+linear is the floor → ~0.5-0.6 valence),
  scale further, rebuild trajectory/multi-speaker on the fused engine. Artifacts:
  `out/fused_engine_result.txt`, `models/dim_fused_msp`, `out/ssl_*.npy`.
- **2026-06-29 — SSL FUSION VALIDATED: learned audio-vectors fix valence (the big lever).**
  Child experiment in `.venv_diar` (`scripts/exp_ssl_fusion.py`, parent untouched): WavLM-base
  embeddings vs our 111 features vs fused, same 5k train / 2k held-out Test1, Ridge linear-probe
  + balanced RF. **V/A/D CCC: valence classical 0.10 → SSL 0.34 → fused 0.33; arousal
  0.60→0.64→0.65; dominance 0.49→0.52→0.53; mean 0.40→0.50→0.50.** Valence TRIPLES on equal
  footing, and SSL on **5k clips (0.335) matches our best classical valence on 169k (0.347)** —
  34× more data-efficient. **Fusion best overall** (the two views are complementary: classical
  anchors A/D, SSL brings V). Emotion (balanced RF, majority 50%): classical 46%/bal 19.5 → ssl
  52%/18.6 → **fused 55%/bal 22.8** (best). Caveats: frozen WavLM + linear probe = the FLOOR
  (fine-tuning → ~0.5-0.6 valence in lit); balanced emotion still modest (rare classes hard);
  deep-learning tier (torch). **Conclusion: SSL embeddings are the real path past the classical
  ceiling, esp. valence — this is the Phase-3 lever. Confirms the user's two-view hypothesis.**
- **2026-06-29 — SINGLE-SPEAKER EMOTION REALITY CHECK (the gating test). Honest verdict:
  fine 8-way emotion from voice DOES NOT work; arousal/dominance LEVEL does.** Evaluated
  end-to-end on held-out single-speaker clips (Dev 27k + truly-held-out Test1 5k):
  • Centroid-naming pipeline (the trajectory's classifier): **27% Dev / 28% Test1 — BELOW the
    majority baseline (31% / 49%)**. Fails. (`scripts/eval_emotion_singlespeaker.py`)
  • Direct balanced-RF classifier (`scripts/train_direct_emotion.py`): **44% Dev / 57% Test1,
    beats raw-accuracy baseline (+13 / +8 pts) BUT balanced accuracy only ~20% (chance 12.5%)**
    — driven almost entirely by neutral (66–85% recall) + joy (53–57%); **contempt/disgust/
    fear/surprise/sadness ≈ 0%**. So it detects neutral/joy/(weak)anger and nothing else.
  • Coarse 3-level (low/mid/high) per dimension, held-out Test1 (chance 33%): **arousal 58%,
    dominance 53%, valence 39%** — i.e. voice reliably reads activation level, weakly reads
    pleasantness. Consistent with every prior layer.
  **Conclusion: the engine's real product is the continuous arousal/dominance signal + the
  trajectory, NOT fine discrete emotion. No classical tweak fixes the 0%-recall emotions
  (acoustically subtle + valence-bound); only SSL embeddings or text fusion would.** This
  gates the multi-speaker work: per-speaker output is trustworthy for arousal/dominance,
  not for fine emotion labels.
- **2026-06-29 — DIARIZATION evaluated on 5 controlled conversations + cosine fix.**
  Fixed a real Path-B bug: ECAPA embeddings were clustered with Euclidean/standardized
  distance — switched to COSINE (L2-normalize). Built `scripts/build_test_conversations.py`
  (stitch known RAVDESS actors → exact ground truth: mixed-gender, same-gender, 3-speaker,
  and a 1-speaker/many-emotion case) + `scripts/eval_diarization.py` (best-permutation
  attribution + auto-k detection). **Results across 5 audios:**
  **ATTRIBUTION (given true k): Path B 79% mean vs Path A 52% (≈chance); same-gender B 90%
  vs A 52%** — neural embeddings work where classical can't (gold result for the user's
  distinguisher idea). **AUTO speaker-COUNTING: unreliable for BOTH** (silhouette 1/5;
  calibrated cosine threshold best only 2/5 — `scripts/calibrate_diar_threshold.py`). Root
  cause: at 2-s windows the within-speaker embedding spread ≈ between-speaker spread, so
  unsupervised counting from fixed windows has no stable threshold. **Honest resolution:
  SPECIFY the speaker count (--speakers, usually known) → strong attribution; reliable
  auto-counting needs full VAD + speaker-change segmentation = pyannote territory.** 170 tests
  still green. Artifacts: `out/diarization_eval.csv`, `own_voice/test_convos/`.
- **2026-06-29 — PATH B: neural distinguisher (ECAPA) — validated, big attribution win.**
  Built the "one system, two angles" design: an ISOLATED `.venv_diar` (torch 2.12 +
  speechbrain 1.1, quarantined from the main venv — main stays 170 tests green) runs
  `scripts/diarize_neural.py` → ECAPA-TDNN embedding per window → cluster → who-spoke-when
  CSV; `scripts/diarize_neural_demo.py` (main venv) consumes it via
  `TrajectoryEngine.analyze_by_speaker(path, labels=...)` → per-speaker emotion + viz. The
  heavy dep never touches the emotion engine. **DECISIVE VALIDATION on long_1 (ground truth:
  first 27s = male only):** Path A classical purity **52%** (split the single man 14/13 — count
  right by luck, attribution random) vs Path B neural **96%** (26/27 to one speaker). This is
  the structural difference measured: classical features leak emotion → split one person; ECAPA
  embeddings are speaker-invariant → keep him together. **Kitchen Debate (1959):** BOTH auto-k=1
  — degraded mono + crosstalk + translator defeats even neural unsupervised separation (honest
  ceiling of the audio, not the method). Confirms the user's architecture: distinguisher
  (definite answer, Path B much better) + measurer (approximate, unchanged) = one system.
  +3 tests (label path) → **170 pass**. No token needed (ECAPA public). pyannote remains a
  future option if overlap-aware diarization is needed.
- **2026-06-29 — EXTENSION: speaker diarization front-end (Path A, classical).**
  Built `src/dimensional/diarization.py` (`SpeakerDiarizer`): clusters windows on the
  speaker-discriminative, emotion-STABLE feature subset (formants F1-F3, MFCC timbre means,
  pitch range — the opposite of the failed Phase-1 F0-threshold attempt; ties to our
  stability finding that formants are <3% drift). Agglomerative clustering + silhouette
  auto-k + min-turn smoothing; NaN windows → label -1. Integrated as a front-end:
  `TrajectoryEngine.analyze_by_speaker(path, diarizer)` → one trajectory per speaker
  (smoothing runs WITHIN each speaker's timeline). No new heavy deps, no deep learning (it's
  a front-end, not the emotion model). +7 tests → **167 pass**. Demo `scripts/diarize_demo.py`
  + multi-speaker viz. **HONEST VALIDATION:** `long_1.wav` (modern, distinct male+female) →
  auto-detected **2 speakers, correct**. Public-domain **Kitchen Debate (1959, downloaded)** →
  auto-detected **1 (WRONG**; has Nixon+Khrushchev+translator) — degraded/old audio + crosstalk
  + translator defeats classical features; forcing k=3 gives plausible-but-unverifiable
  clusters. **This is the boundary of Path A → Path B (pretrained neural embeddings, e.g.
  pyannote) is the upgrade, same interface.** Per-speaker valence/labels carry all the usual
  caveats (arousal/dominance reliable, labels ambiguous).
- **2026-06-29 — P2.6 DONE: writeup shipped. PHASE 2 COMPLETE.** `WRITEUP.md` — standalone
  technical/portfolio report: thesis, method (3-layer), MSP data (empirical-not-truth),
  results (definitive CCC table, the 6× valence arc, the speaker-norm negative result, the
  42.4% separability + data centroids, the trajectory demo + ambiguity honesty), limitations
  (handcrafted cap / text ceiling / per-tick not validated), reproducibility commands. All
  numbers pulled from actual runs. P2.0–P2.6 done; 160 tests green. Remaining items are
  optional polish only (trajectory animation, Test1/Test2 confirmation, SVR tuning) or
  Phase-3 (text fusion for valence).
- **2026-06-28 — P2.4 + P2.5 DONE: trajectory engine + emotion web.** Built
  `src/dimensional/trajectory.py` (`TrajectoryEngine`: window 2s/hop 1s → per-window 111
  features → SVR V/A/D → EMA smooth → centroid-name; per-window peak-normalize matches
  training; native 1-7 clipped→normalized PAD before naming; failed windows = recorded GAP,
  never dropped). +11 tests → **160 pass**. Demo on `own_voice/long_1.wav` (5-min interview):
  **319 windows, 0 gaps, full 319.6s**, PAD means V−0.07/A0.57/D+0.15. Viz
  `scripts/trajectory_viz.py` → `out/emotion_web_trajectory_long_1.png` (3-D path + V-A & V-D
  projections w/ centroids + V/A/D-over-time). HONEST READ: **304/319 windows AMBIGUOUS** —
  the continuous V/A/D PATH is the real signal; discrete per-window labels are low-confidence
  (conversational speech sits in the central overlap zone = the ceiling at window level, Law 4).
  Don't over-read the 46% "surprise". Per Law 18 this is a DEMO; per-tick accuracy needs a
  continuous corpus. Artifacts: `out/trajectory_long_1.csv`, `out/emotion_web_*.png`.
- **2026-06-28 — DEFINITIVE NUMBERS (P2.3 DONE). Canonical protocol: train=Train 169k →
  test=Dev 34k, speaker-independent.** Best model **SVR (calibrated, 25k cap)**: **valence
  CCC 0.347, arousal 0.612, dominance 0.515** (mean 0.491). Ridge 0.156/0.601/0.505; RF
  0.229/0.579/0.467. **Valence arc: 0.059 (probe ridge) → 0.180 (SVR 8k) → 0.347 (SVR 169k
  +calib)** = 6× — at/above the published handcrafted-acoustic baseline. Diagnosis confirmed:
  valence weakness was model choice, not a wall. Arousal/dominance solid + stable on full
  corpus. Beyond 0.35 valence needs SSL or text (deferred, documented). **Layer 4
  separability 42.4%** over 8-class gold set (chance 12.5%); centroids (137k): joy lone
  positive valence (+0.30); anger D=+0.41 vs fear D=−0.01 still separate; neutral central
  (−0.04/0.45/0.03) = top confuser. Saved `models/dim_svr_msp_final`, `models/namer_msp_final`,
  `out/study_msp_final.txt`. **P2.3 complete — next P2.4 (trajectory/windowing + demo).**
- **2026-06-28 — VALENCE LEVER FOUND: it was the model, not a wall. + Train run launched.**
  Quick check on Dev (`out/exp_valence_quick_dev.txt`): **SVR tripled valence** vs Ridge/RF
  — valence CCC raw 0.059 (ridge) / 0.110 (RF) / **0.180 (SVR, on 8k rows)**; with output
  calibration **0.086 / 0.138 / 0.228**. SVR beat Ridge/RF with LESS data → the faint valence
  signal is real and nonlinear; Ridge (linear) and RF (hedges to mean) just couldn't use it.
  Valence reframed: not "broken" (~0.06) but "weak-but-usable, at the published classical
  baseline" (~0.23). Arousal/dominance unchanged (~0.6, all models similar). ETV/EIV ruled
  out for this — they are COMPUTED FROM valence (EIV=f(valence,...)), downstream consumers,
  can't predict it (circular). Real ceiling-breaker remains TEXT (transcripts in the MSP
  download) → Phase-3 identity choice, not snuck in. **Made calibration a permanent,
  leakage-free, opt-in feature of `DimensionalRegressor`** (affine fit on TRAIN preds→targets;
  save/load; +2 tests → **149 pass**). **Study upgraded to SVR(+cap 15k)+calibration.**
  **Train split extraction LAUNCHED** (169,190 files, ~7h, ~68 checkpoints @ ~6min, resumable).
- **2026-06-28 — NEGATIVE RESULT: per-speaker norm HURTS dimensional regression.** Tested
  the cheap hypothesis (center each speaker on own mean → claw back valence). Result: valence
  unchanged (−0.01), arousal −0.30, dominance −0.32 (Ridge). MSP labels are ABSOLUTE; the
  A/D signal lives in a speaker's absolute vocal level, which centering deletes. Valence
  unmoved → its weakness is NOT a speaker-timbre confound, it's genuinely faint in acoustics.
  Verdict: per-speaker baseline belongs in expression-strength/intensity (deviation from
  neutral), NOT in absolute V/A/D regression → **Law 8 corrected below**. The real valence
  lever is more data (Train 169k vs Dev 28.9k) + possibly SVR. `out/exp_speaker_norm_dev.txt`.
  Good ROI: a 2-min experiment stopped a harmful transform from entering the 6h Train run.
- **2026-06-28 — FIRST REAL NUMBERS (P2.3 probe on MSP Dev, 34,398 clips, speaker-independent).**
  Layer 2 CCC (Ridge / RF): **arousal 0.624 / 0.558 ✅, dominance 0.567 / 0.495 ✅,
  valence 0.059 / 0.110 ❌**. Arousal/dominance match the acoustic-only literature; valence
  is at the noise floor AND below the classical eGeMAPS baseline (~0.2–0.3) → headroom, not a
  wall (no per-speaker norm yet; trained on Dev's 28.9k only, not Train's 169k; RF valence
  Pearson 0.22 vs CCC 0.11 = hedging to the mean). Do NOT headline mean CCC (0.42 mixes a
  strong pair with a broken axis). **Layer 4:** data centroids confirm the 3-D payoff —
  anger D=+0.45 vs fear D=−0.28 separate on dominance (Russell & Mehrabian). Separability
  ceiling **38.2%** over the 8-class gold set (chance 12.5%); top confusions are real
  (anger↔contempt centroids nearly identical; neutral central, bleeds everywhere). Saved
  `models/dim_ridge_msp_dev`, `models/namer_msp_dev`, `out/study_msp_dev.txt`. Next moves:
  (A) cheap evidence-driven per-speaker normalization on existing Dev features (Law 20
  unlocked — valence below baseline justifies it); (B) extract Train (169k, ~6h) for the
  definitive numbers.
- **2026-06-27 (late) — SCOPE LOCKED + corpus on disk + Dev extraction running.** Corpus
  extracted to `data/msp_podcast/` (267,905 WAVs + Labels). Real `labels_consensus.csv`
  header is an **exact match** to the loader aliases (`FileName,EmoClass,EmoAct,EmoVal,
  EmoDom,SpkrID,Gender,Split_Set`) — zero loader changes. Loader verified on real data
  (264,705 valid V/A/D; Dev 34,399; Dev-Ekman6 17,411). End-to-end smoke test 5/5 clean.
  Built `scripts/extract_msp.py` (carries native V/A/D + emotion + speaker, checkpoint/
  resume) and `scripts/study_msp.py` (speaker-independent split, CCC per dim, namer
  separability). Fixed a `np.savez` checkpoint bug (auto-appends `.npz`), salvaged 2,500
  rows. **Dev extraction running (~34k, ~80min, deterministic).** 147 tests pass.
  **Scope decisions this session (now LAW — see §5 additions 15–20):** (a) companion-app
  contract DROPPED — independent experiment, output is the V/A/D triple + distribution,
  ETV/EIV parked; (b) **MSP-Podcast is the PRIMARY spine**, all other corpora supplementary;
  (c) **Phase-1 numbers RETIRED** (acted/unverified) — regenerate every result on MSP;
  (d) gold standard is **empirical, not ground truth** — emotion is felt, we approximate
  via statistics; (e) trajectory needs **time-continuous** data to validate (MSP is
  per-segment — point predictor only); (f) **naming uses the gold set: 6 + contempt +
  neutral**, drop Other/X — NOT an Ekman-6 cage; (g) engine tweaks allowed but
  **evidence-first** — baseline on the unchanged engine before any change.
- **2026-06-27 — MSP-Podcast access GRANTED + loader built.** Busso approved an individual
  license; UTD OSP countersigned; credentials received. Implemented `load_msp_podcast` in
  `src/dimensional/loader.py` against the real readme schema (Labels/labels_consensus.csv +
  Audios/, native SAM 1-7, Ekman-6 mapping, Test3 auto-skip, split filtering, +
  `normalize_vad_msp` 1-7→PAD and `vad_matrix`). +7 tests → **147 pass**. P2.2 effectively
  done. Remaining before P2.3: user downloads 50GB corpus locally, confirm real
  labels_consensus.csv header matches loader aliases, then extract Dev (~34k) first.
- **2026-06-23 — P2.1 DONE (dimensional scaffold).** New package `src/dimensional/`:
  `metrics.py` (CCC/RMSE/Pearson + `dimensional_report`), `regressors.py`
  (`DimensionalRegressor` = the three V/A/D waves, reuses `FeatureNormalizer`
  train-only, SVR/RF/Ridge, save/load), `namer.py` (`CentroidNamer` — data centroids +
  Mahalanobis → softmax distribution + intensity radius + ambiguity flag, save/load),
  `loader.py` (`DimensionalSample`, `vad_from_categorical` smoke path [circular — wiring
  only], `load_iemocap`/`load_cmu_mosei` stubs raising NotImplementedError with guidance).
  +23 tests → **140 pass total**. Pipeline runs end-to-end on synthetic + functional data;
  ready to consume a real dimensional dataset the moment one lands.
- **2026-06-23 — P2.0 DONE (dominance gap closed).** `signal_mapper.py` now outputs a
  full **(valence, arousal, dominance)** triple. While doing it, discovered the previous
  2-D valence/arousal values were **unsourced** — they did not match the cited Warriner
  source for any word form (arousal systematically too high). Corrected the whole table to
  **verified Warriner CSV values** (emotion nouns, 1-9 → normalized) and added the
  dominance column from the same source. Anger D=+0.035 > fear D=−0.42 (the separation
  works). These remain an interim literature prior (Law 3 — replaced by data centroids in
  P2.3). **117 tests pass** (added `test_dominance_separates_anger_from_fear`). Data source
  cached as a note: `crr.ugent.be` / JULIELab XANEW mirror of `Ratings_Warriner_et_al.csv`.

---

## 0. THE NORTH STAR (one paragraph)

We are evolving the existing Waveform Engine into an **emotion *trajectory* engine**.
Phase 1 answered *"what one emotion is in this clip?"* Phase 2 answers
*"where is this voice in emotional space right now, and how is it moving over time?"*
We predict three numbers from voice — **Valence (X), Arousal (Y), Dominance (Z)** —
treat them as a **point** in a 3-D PAD space, track that point **window by window**
over a recording, and render the path as an **emotion trajectory ("the web").**
The numbers are *measured*; emotion *names* are assigned only by proximity to
**data-grounded centroids**, and reported as a **distribution**, never a forced
single label. This is a rigorous, honest, well-visualized **portfolio/workshop
project** — not a new theory.

---

## 1. THE FOUNDATION (what already exists — verified 2026-06-22)

We are **not starting from zero.** Phase 1 gives us a hardened, tested base.

### Layer 1 — ENGINE (reuse as-is, do NOT rebuild)
- `src/preprocessing.py` — load → 16kHz mono → trim → normalize, with min-duration
  guard, configurable trim, typed `AudioError`.
- `src/features/` — **`build_feature_vector()` → 111 features** (88 eGeMAPS + 13 Praat
  + 10 prosody) + 2 metadata (duration, SNR). Validated (no NaN/Inf), adaptive F0
  range, range-checked.
- `src/features/normalize.py` — `FeatureNormalizer` (StandardScaler + data-driven
  selection, save/load, train-only fit).
- `src/speaker_baseline.py` — `SpeakerBaseline` per-speaker neutral calibration.
- **This layer is dataset-agnostic and modality-agnostic. It is the constant.**

### Layer 2 — CLASSIFIER (exists, categorical — complemented, not deleted)
- `src/classifier/{train,predict,evaluate}.py` — SVM/RF, train/predict/evaluate.
- Saved models: `models/{svm_ravdess,svm_ravdess_meld,svm_tuned,rf_tuned}`.
- `predict()` already returns **`ekman6_weights`** — a 6-D blend vector. *This is
  already a blend; Phase 2 uses it and adds the dimensional route alongside it.*

### Layer 3 — SIGNAL MAPPER (exists, 2-D — to be EXTENDED to 3-D)
- `src/signal_mapper.py` maps to **valence + arousal only**. **Dominance is missing**
  — even though its cited source (Warriner et al. 2013) contains the dominance column.
  Closing this is the first concrete code task.

### Assets already on disk
- `out/features_{ravdess,crema_d,meld}.npy` + labels — **~15,000 feature vectors
  already extracted** (RAVDESS 1,152 · CREMA-D 6,355 · MELD 7,272).
- **116 tests passing.** `data/` has RAVDESS + CREMA-D; `own_voice/` real recordings.

### The gap that defines Phase 2
- **All current datasets (RAVDESS, CREMA-D, MELD) are CATEGORICAL-ONLY.** They have
  no V/A/D annotations. We **cannot validate three dimensional waves against them.**
  → We must add a **dimensionally-annotated** corpus. **IEMOCAP first.**

---

## 2. THE ARCHITECTURE (the revised map)

```
Audio (any length)
   │
   ▼
[ Layer 1: ENGINE ]  ── EXISTS ──  build_feature_vector() → 111 features
   │                                (per window in Phase 2)
   ▼
[ Layer 2: THREE WAVES ]  ── BUILD ──  3 independent regressors:
   │   Wave A → Arousal (Y)   [strong from voice]
   │   Wave V → Valence (X)   [WEAK from voice — the valence problem]
   │   Wave D → Dominance (Z) [recoverable, needs per-speaker calibration]
   │   Output: a TRIPLE (v, a, d)  ── NEVER collapsed to one scalar ──
   ▼
[ Layer 3: THE PAD PLANE ]  ── BUILD ──  place the triple as a POINT in 3-D
   │
   ▼
[ Layer 4: NAMING ]  ── BUILD ──  data-grounded centroid + covariance per emotion;
   │   classify point by MAHALANOBIS distance → softmax → DISTRIBUTION
   │   (+ intensity = radius from origin = expressionStrength)
   ▼
[ Layer 5: TRAJECTORY ]  ── BUILD ──  window the audio (2s window / 1s hop),
   │   run Layers 1–4 per window, SMOOTH the sequence (EMA/Kalman)
   │   → timestamped list of (t, v, a, d, distribution)
   ▼
[ Layer 6: VISUALIZATION ]  ── BUILD ──  the "emotion web": the point moving
       through PAD space over time. THE RESUME DIFFERENTIATOR.
```

### The 8 octants (Mehrabian) — the region map, not hand-placed labels
The Z axis doubles the 4 V-A quadrants into 8 octants. Anchors come from data, but
the *structure* is: +++Exuberant(joy) · ++−Dependent(awe) · +−+Relaxed(calm) ·
+−−Docile(serene) · −++Hostile(anger) · −+−Anxious(fear) · −−+Disdainful(disgust) ·
−−−Bored(sadness). Anger vs fear separate on **dominance** alone.

---

## 3. SCIENCE ANCHORS (grounded, cited — no intuition-only claims)

| Claim | Source |
|---|---|
| 3 axes V/A/D are "necessary and sufficient"; dominance splits anger/fear | Russell & Mehrabian (1977) |
| 8 named octants from the sign combinations | Mehrabian PAD temperament |
| Emotion (V,A,D) coordinates come from data/averaged human ratings | Warriner et al. (2013); AffectNet |
| Valence is the WEAK axis from acoustics (lives in words) | "Valence problem", SER literature |
| One V-A point ≈ 5–6 emotions; happy/fear clean, anger/sad overlap | VAD→category mapping studies |
| Names are constructed; numbers (core affect) are fundamental | Barrett, Theory of Constructed Emotion |
| Emotion space is ~27 fuzzy categories bridged by gradients | Cowen & Keltner (2017) |
| Dimensional prediction is scored by **CCC**, not accuracy | AVEC challenge series |
| Primary corpus: 264k natural-speech segments, consensus V/A/D (SAM 1-7), speaker-independent splits | MSP-Podcast v2.0 (Busso et al.), arXiv:2509.09791 |
| Labels are a mean of ≥5 annotators → empirical approximation w/ a human-disagreement floor, not ground truth | MSP-Podcast annotation protocol |

---

## 4. THE PLAN (phased, each step verified before the next)

- **P2.0 — Close the dominance gap.** ✅ DONE 2026-06-23. (V,A,D) triple in
  `signal_mapper.py`, values corrected to verified Warriner CSV, 117 tests pass.
- **P2.1 — Dimensional scaffold.** ✅ DONE 2026-06-23. `src/dimensional/`: three
  regressors (SVR/RF/ridge), CCC evaluator, Mahalanobis centroid→distribution namer,
  loader stubs. +23 tests, 140 pass total.
- **P2.2 — Get the dimensional corpus.** ✅ DONE 2026-06-27. **MSP-Podcast** (not IEMOCAP)
  is the spine: granted, downloaded to `data/msp_podcast/`, `load_msp_podcast` locked
  against the real schema. 264,705 segments, native V/A/D (SAM 1-7), speaker-independent
  splits. The only source that can train + validate the dimensional waves.
- **P2.3 — The study (on MSP).** ✅ DONE 2026-06-28. Canonical protocol (train=Train 169k →
  test=Dev 34k). **SVR+calibration: valence 0.347, arousal 0.612, dominance 0.515.** Gold-set
  centroids + 42.4% separability (chance 12.5%). Matched the prediction (A~0.6, D~0.5) and
  beat the valence expectation (0.35 vs predicted 0.3–0.4 floor) via SVR+calibration+full data.
  Artifacts: `out/study_msp_final.txt`, `models/dim_svr_msp_final`, `models/namer_msp_final`.
- **P2.4 — Trajectory.** ✅ DONE 2026-06-28. `TrajectoryEngine` (window/hop, per-window
  V/A/D, EMA smoothing, centroid naming, gap-safe). Demo: 319 windows / 0 gaps on a 5-min
  clip. DEMO only per Law 18 (per-tick accuracy needs continuous labels).
- **P2.5 — The web.** ✅ DONE 2026-06-28. `scripts/trajectory_viz.py` → 4-panel emotion web
  (3-D PAD path + V-A/V-D projections w/ centroids + V/A/D-over-time). Optional next: animation.
- **P2.6 — Writeup.** ✅ DONE 2026-06-29. `WRITEUP.md` — portfolio/technical report with the
  full honest results, limitations, and reproducibility. **PHASE 2 COMPLETE.**

---

## 5. THE LAWS (anti-drift — binding, for Claude and human alike)

1. **KEEP THE TRIPLE.** (V, A, D) is a *point*. **Never** collapse it into one scalar,
   and never put emotions on a 1-D number line. Collapsing destroys the orthogonal
   axes and re-creates the arbitrariness we rejected.
2. **CCC, NOT "ACCURACY", FOR DIMENSIONS.** V/A/D prediction is regression → report
   **CCC** (+ RMSE, Pearson). "Accuracy" applies ONLY to the final point→category step.
3. **NO HAND-PLACED COORDINATES.** Every emotion's position comes from **data
   centroids**, not opinion. The defensible claim is always *"lands where human-rated
   X clusters,"* never *"is X."*
4. **REPORT THE DISTRIBUTION, NOT A FORCED LABEL.** When a point is between clusters,
   output weights. Ambiguity is reported, never hidden. The blend IS the answer.
5. **THE ENGINE (LAYER 1) IS THE CONSTANT.** Dataset/modality-agnostic. If you find
   yourself writing `if dataset == ...` in the engine, you have made a mistake. Reuse
   `build_feature_vector` — do NOT rebuild or fork it.
6. **VALIDATE ONLY AGAINST DIMENSIONAL GROUND TRUTH.** No V/A/D claim without a
   dimensionally-annotated dataset (IEMOCAP/MSP/RECOLA). The current 3 datasets cannot
   support this study — do not pretend they can.
7. **VALENCE IS WEAK — SAY SO.** Never hide a low valence CCC. Trust arousal/dominance
   more; report valence honestly. The weakness is expected and documented.
8. **PER-SPEAKER BASELINE → EXPRESSION/INTENSITY ONLY, NOT V/A/D REGRESSION.**
   *(Corrected 2026-06-28 by experiment.)* Centering features per speaker DESTROYS absolute
   arousal/dominance signal (CCC −0.30/−0.32 on MSP) because MSP labels are absolute. Use
   `SpeakerBaseline` for expression-strength/intensity (deviation from a speaker's neutral),
   NOT as a pre-transform for the absolute V/A/D regressors. Original Hoffmann rationale
   applies to relative/expression measures, not absolute dimensional prediction.
9. **THIS IS A PORTFOLIO/WORKSHOP PROJECT.** No novelty claims, no new theory. Honest
   reproduction + the trajectory differentiator. Do not chase a top-venue paper.
10. **ZERO LEAKAGE.** Every window in → a point or an explicit error. No silent drops,
    no fallback values masking failure. Normalizer + regressors fit on TRAIN ONLY.
11. **NO DEEP LEARNING FOR MVP.** SVR / RandomForest / ridge. If they prove
    insufficient, document why first.
12. **NO MAGIC NUMBERS.** Every threshold, centroid, and weight comes from data, with
    its source named in code.
13. **EXPLOSIVE STEPS, VERIFIED.** Each P2.x produces a documented, tested result
    before the next begins. Keep tests green; log results.
14. **STAY ON THE ANCHOR.** Before adding anything not in §4, check it against these
    laws. If it isn't on the map, it waits.
15. **MSP-PODCAST IS THE PRIMARY SPINE.** It is the train + validate source for every
    dimensional result. RAVDESS/CREMA-D/MELD are **supplementary** (cross-corpus sanity
    on the categorical readout, Phase-1 legacy) — they have no V/A/D and never headline.
    Any other corpus is supplementary until it earns otherwise.
16. **PHASE-1 NUMBERS ARE RETIRED.** Every accuracy/CCC/confusion from acted datasets is
    untrustworthy for the real goal (proven by the cross-dataset collapse + own-voice
    bias). Do not cite them as results. Regenerate **all** Layer 2/4 numbers on MSP.
    "Regenerate from scratch" = re-run the model + evaluation; the **deterministic feature
    extraction is reproducible** and is NOT a trust variable.
17. **THE GOLD STANDARD IS EMPIRICAL, NOT TRUTH.** MSP labels are the *mean of ≥5 human
    annotators' guesses* — they carry a human-disagreement noise floor. Emotion is felt,
    not measured; we report **probabilistic approximations**, never "the true emotion."
    MSP numbers are trusted because they **generalize**, not because they are exact.
18. **TRAJECTORY NEEDS TIME-CONTINUOUS DATA TO VALIDATE.** MSP gives ONE label per
    2.75–11s segment → it validates a per-window **point predictor only**. The per-second
    **trajectory is a DEMO** until a continuous corpus (RECOLA/SEWA/SEMAINE) is added.
    Never report per-tick trajectory accuracy from MSP. Window size may be tuned on MSP
    (feature stability), but trajectory rigor waits for continuous labels.
19. **NAMING USES THE GOLD SET, NOT AN EKMAN CAGE.** The namer's classes are MSP's own:
    anger, sadness, happiness, surprise, fear, disgust **+ contempt + neutral** (neutral
    as the empirical center). Drop only Other/X (annotation catch-alls, not emotions) —
    they stay in regression but pollute centroids. Contempt's position is **measured from
    MSP**, not borrowed. The regressor (Layer 2) is taxonomy-free and uses all rows.
20. **ENGINE TWEAKS ARE EVIDENCE-FIRST.** Layer 1 may be adjusted (Law 5 still bars
    forking/dataset-coupling), but only **after** a baseline on the unchanged engine shows
    what's weak. No speculative edits to a validated extractor before the first honest
    number. Re-extract only when the engine actually changes.

> **Contract note (supersedes Phase-1 coupling):** This is an independent experiment, not
> a commercial companion-app feature. The engine's deliverable is the `(V,A,D)` point + emotion
> distribution + intensity. ETV/EIV remain on disk and are used ONLY if they demonstrably
> add value — they are not part of proving the perception engine.

---

## 6. WHAT SUCCESS LOOKS LIKE

1. Three-wave regressor: **CCC reported** per dimension on IEMOCAP, honestly incl. weak valence.
2. Data-grounded per-emotion **centroids + covariance** computed from real annotations.
3. Point→emotion **distribution** via Mahalanobis, with documented separability ceiling.
4. **Time-continuous trajectory** on a full clip, validated against RECOLA.
5. The **emotion-web visualization** — the demo nobody else in the project stack has.
6. An honest **writeup** with CCC numbers, the valence finding, and the overlap ceiling.
7. 116 tests still green + new tests for every Phase-2 module.

## 7. EXPLICIT NON-GOALS (we will NOT do these now)

- ❌ Real-time/streaming. Batch only.
- ❌ Deep learning. (Law 11)
- ❌ A new emotion theory or novelty claim. (Law 9)
- ❌ Instrumental/music emotion. Real future modality, **separate extractor** — not now.
- ❌ Rebuilding Layer 1 or forking the feature extractor. (Law 5)
- ❌ Collapsing V/A/D to a scalar or a 1-D emotion index. (Law 1)
- ❌ Validating dimensions on categorical-only datasets. (Law 6)
- ❌ Building/owning ETV/EIV here. They are companion-backend consumers, parked on disk; used only
  if they demonstrably add value. (Contract note, §5)
- ❌ Headlining or citing Phase-1 (acted-dataset) numbers as results. (Law 16)
- ❌ Reporting per-tick trajectory accuracy from MSP's per-segment labels. (Law 18)

---

*Anchor set Day 1, 2026-06-22. If work drifts from §2/§4 or violates §5, stop and re-read this file.*
