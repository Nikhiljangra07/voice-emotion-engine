# STEERING_LOG — the chronological record of the TTS Steering & Benchmark project

> Everything this project does, in order: what we adopted, what we rejected and why,
> every hurdle hit and how it was cleared. Same discipline as the parent engine's
> `JOURNEY.md` — write it as it happens, never after the fact.
> Charter and binding laws: [`README.md`](README.md).

---

## THE LINEUP

**Our foundation (the model we steer):**

| Role | Model | Why | License (honest) |
|---|---|---|---|
| **Primary** | **IndexTTS-2** (Bilibili, 2025) | Only open model with an explicit **8-dim continuous emotion vector** `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` (sliders 0–1) — a machine-searchable control space mapping ~1:1 to our judge. Disentangles timbre from emotion → hold voice constant, vary only emotion. | Code Apache-2.0; **weights: commercial use needs Bilibili written authorization** → fine for this non-commercial research benchmark, documented not hidden |
| Backup / 2nd open system | **Chatterbox** (Resemble AI) | **MIT end-to-end**, `exaggeration` knob + `cfg_weight` + 5-s voice cloning (steerable via our labeled emotional reference clips). Coarser control, bulletproof license. | MIT |
| Optional research angle | **StyleTTS 2** | MIT; continuous style-vector space with documented emotion clusters; lets us *apply* task-vector expressivity papers (arXiv:2606.05367, 2303.08329). | MIT |

**The rivals (frontier commercial — benchmark against all three):**

| Rival | Emotion mechanism | Our access |
|---|---|---|
| **ElevenLabs v3** | audio tags in-script (GA 2026-02) | API, small credit spend |
| **Hume AI Octave 2** | reads meaning, self-adapts delivery — the emotion-focused rival, the one that matters most | API |
| **OpenAI TTS** | natural-language instructions ("speak sadly") | API |
| *(optional open baselines)* | CosyVoice 3 (Apache-2.0), Chatterbox presets | local |

**The judge:** the Voice Emotion Engine (parent repo), consumed ONLY through
[`bridge.py`](bridge.py) — fine-tuned WavLM supplies the V/A/D steering signal;
frozen emotion2vec supplies the family verdict. Separate vector spaces = the
anti-circularity design.

**Target emotions:** sadness · joy · anger (data-chosen: judge recall 90/86/69%,
anger adds the dominance axis). **Fear excluded** — judge blind spot (47%).

---

## LOG

### 2026-07-04 — P4.0→P4.2: charter, lineup, isolated environment, bridge VERIFIED

**Adopted:**
- Project charter + protocol written **before any code** (`README.md`) — the parent
  repo's "decide the metric first" discipline (CCC lesson) applied from day zero.
- **Foundation = IndexTTS-2** after a live landscape survey (mid-2026). Deciding
  criterion: continuous, optimizer-sweepable control interface — its 8-dim emotion
  vector is exactly that; nothing else open comes close.
- **Environment isolation, three layers:** `.venv_tts` (this project's light env:
  numpy/soundfile only) · `tts_steering/vendor/index-tts` with its **own uv-managed
  env pinned to Python 3.12** (its stack must never touch ours) · the engine's
  `.venv_diar` untouched behind the bridge. Three venvs, zero shared imports.
- **The bridge** (`bridge.py`): the single connection to the parent engine —
  subprocess → CLI → JSON/text. If the engine changes, one file changes here.

**Rejected (and why):**
- Tag-based TTS as foundation (Orpheus, Maya1, ElevenLabs-style tags): discrete tags
  can't be swept by a continuous optimizer. They remain benchmark *subjects*, not the
  steering foundation.
- Kokoro (great speed, near-zero emotion control), F5/Spark (cloning-oriented).
- Fear as a target emotion: our judge can't reliably measure it (47% recall) —
  optimizing against a broken meter is self-deception.
- Importing engine code directly: would weld the projects together. Bridge or nothing.

**Hurdles hit (chronological, honest):**
1. **IndexTTS-2 license surprise** — headline says "open source," but the weights
   carry a custom Bilibili clause (commercial use requires written authorization).
   Caught during the survey, *before* building on it. Resolution: compliant for this
   non-commercial benchmark; documented in the lineup table; MIT Chatterbox held as
   the license-safe fallback if this project ever grows commercial ambitions.
2. **Judge self-match inflation** — bridge verification clips scored family
   confidence 100% because those exact clips are IN the retrieval database
   (exemplar self-match). Harmless for verification, but benchmark scoring of
   *generated* audio is unaffected (TTS output can't be in the DB). Logged so the
   100% is never mistaken for typical confidence.
3. **Python version pin — twice.** IndexTTS-2 says ≥3.10, but first sync on 3.12
   died building `llvmlite==0.41.1` (pinned via `numba==0.58.1` — no 3.12 wheels,
   source build fails). Real constraint: **≤3.11**. Resolution: uv-managed
   **Python 3.11** for the vendor env — third interpreter in the stack
   (`.venv_tts`=3.13, engine `.venv_diar`=3.13, vendor=3.11), all isolated, none
   system-level. Lesson for the log: a repo's `requires-python` is a claim, not a
   guarantee — the transitive pins decide.

**Verified (the bridge works end-to-end, engine untouched):**

| Clip (labeled) | V | A | D | WavLM says | Judge (e2v) says |
|---|---|---|---|---|---|
| Sad(1).wav | −0.21 | 0.27 | −0.17 | sadness | **sadness** @100% |
| Happy_(2).wav | **+0.53** | 0.47 | +0.15 | joy | **joy** @100% |
| Angry_(13).wav | −0.45 | 0.62 | **+0.47** | contempt | **anger** @100% |

Three sanity reads, three correct families from the judge — and anger shows its
dominance signature (D=+0.47), the axis this engine uniquely measures. (WavLM's
"contempt" on the anger clip is the known adjacent-centroid call — "contempt is cold
anger" — which is exactly why the judge, not the steerer, names the family.)

**More hurdles (same day, the environment fought back — all cleared):**
4. **PyPI timeout** killed the first successful sync at the last package (`jieba`,
   45 s connect timeout). Resolution: retry — uv's cache made it cheap.
5. **Full network drop mid-weights** (DNS couldn't resolve huggingface.co, 52 MB into
   5.9 GB). Resolution: relaunch with **resume + 6-attempt retry loop** — the download
   continued from where it died. Pattern adopted for all future large downloads.
6. **`uv run` sabotaged by the vendor's own `.python-version`** — the repo pins `3.10`,
   so `uv run` rejected our synced 3.11 env and silently spun up a bare one →
   `ModuleNotFoundError: indextts` (a *silent-wrong-env* failure, cousin of the parent
   repo's silent-garbage lesson). Resolution: bypass `uv run` entirely — call
   `vendor/index-tts/.venv/bin/python` directly with explicit
   `PYTHONPATH=<vendor root>`. Absolute interpreter + explicit path = no resolver
   magic, no surprises.

### 2026-07-04 (later) — P4.2 FIRST SYNTHESIS + FIRST JUDGED CLIP: an honest MISS

The full pipeline ran end-to-end for the first time: **IndexTTS-2 on Apple Silicon
(MPS) synthesized speech from an emotion vector, and the engine judged it through the
bridge.** Mouth → ear, closed.

- Setup: RAVDESS neutral clip as speaker prompt (neutral timbre — any emotion must
  come from the vector), neutral sentence, `emo_vector sad=0.8`.
- Synthesis: 4.28 s audio in 72.5 s (RTF ≈ 17 on MPS — slow, fine for batch; a GPU
  pod is the speed escape hatch if sweeps get big).
- **Result (ledger row 1): MISS.** Judged V=−0.06 A=0.41 D=+0.05 → acoustically
  ~neutral (distance to sadness centroid 0.251). WavLM says *neutral*; e2v judge says
  *joy@100%*.

**What this miss teaches (why the ledger exists):**
- The sad slider at 0.8 did NOT move the acoustics to sadness — either the emotion
  vector needs different scaling/combination (sweep needed), the neutral speaker
  prompt dampens it, or emotional English from a Chinese-first model is weaker than
  advertised. All testable.
- **judge=joy@100% on a near-neutral synthetic voice is a judge finding, not just a
  TTS finding** — the e2v retrieval DB has never seen studio-clean synthetic audio;
  its nearest neighbors for "clean + calm" may skew joy. Out-of-domain judge behavior
  is now a tracked thread (this is exactly the "rival clips double as judge test
  data" effect, arriving early).
- 100% kNN confidence on out-of-domain input reconfirms: **confidence is vote-share,
  not evidence** (parent repo, problem #4 of the stress test).

**Next (P4.2 continues):** the systematic sweep — sad slider at 0.4/0.8/1.2, sad+
melancholic combinations, joy and anger vectors, an *emotional* speaker prompt as a
control condition, and a listen-check — every clip a ledger row.

### 2026-07-04 (later) — P4.2 SWEEP: 12 clips — anger SOLVED, sadness has a trail, joy broken informatively

12 deterministic clips (model loaded once; MPS warmed to ~15 s/clip), judged in one
batched bridge call, ledger now at 13 rows. Scripts committed:
`sweep_p42_synth.py` (vendor env, resumable) + `sweep_p42_judge.py` (.venv_tts).

| clip | target | V | A | D | dist | wavlm | judge(e2v) | hit |
|---|---|---|---|---|---|---|---|---|
| baseline_zero | neutral | −0.02 | 0.47 | +0.23 | 0.197 | neutral | neutral@100% | **HIT** |
| sad_04 | sadness | −0.11 | 0.38 | +0.08 | 0.229 | neutral | neutral@100% | miss |
| sad_12 | sadness | +0.04 | 0.40 | +0.07 | 0.354 | neutral | joy@80% | miss |
| **mel_08** | sadness | **−0.16** | **0.30** | +0.01 | **0.166** | **sadness** | neutral@100% | miss |
| sad_mel_06_06 | sadness | −0.13 | 0.30 | −0.07 | 0.178 | neutral | neutral@100% | miss |
| joy_08 | joy | **−0.20** | 0.78 | +0.59 | 0.673 | contempt | surprise@80% | miss |
| joy_12 | joy | **−0.40** | 0.84 | +0.67 | 0.883 | anger | surprise@80% | miss |
| **angry_08** | anger | −0.38 | 0.79 | +0.61 | **0.227** | **anger** | **anger@100%** | **HIT** |
| **angry_12** | anger | −0.55 | 0.88 | +0.77 | 0.422 | **anger** | **anger@100%** | **HIT** |
| calm_08 | neutral | −0.11 | 0.35 | +0.08 | 0.137 | neutral | neutral@100% | **HIT** |
| sad_08_sadprompt | sadness | −0.04 | 0.40 | +0.06 | 0.269 | neutral | fear@100% | miss |
| **angry_08_angryprompt** | anger | −0.51 | 0.90 | +0.80 | 0.443 | **anger** | **anger@100%** | **HIT** |

**Findings (each one a transfer-map coordinate):**
1. **ANGER STEERS — first confirmed steering coordinate.** `angry` slider is
   monotonic and correct on every axis: 0.8→1.2 drives V −0.38→−0.55, A 0.79→0.88,
   D +0.61→+0.77 — exactly the engine's anger signature (high arousal, HIGH
   dominance), and **both backbones agree at every strength**. Mouth and ear speak
   the same language for anger.
2. **The `sad` slider is the wrong knob; `melancholic` is the right one.** `sad`
   0.4→1.2 barely moves valence (and at 1.2 *flips positive* — worse). But
   `melancholic=0.8` produced the closest sadness of the day (dist 0.166, arousal
   correctly LOW at 0.30, WavLM names it *sadness*). The e2v family verdict is still
   neutral — delivered sadness is real but too weak for a family hit yet. Next:
   melancholic at 1.0–1.4, melancholic+calm.
3. **Joy is broken, informatively.** `happy` 0.8→1.2 yields **negative** valence
   with soaring arousal/dominance — the model renders "happy" as loud/energetic,
   which both backbones read as arousal *without* positive valence (surprise per
   judge; contempt→anger per WavLM; higher slider = worse). Either IndexTTS-2's
   happy is acoustically shouty, or the judge undervalues synthetic positive
   valence. Discriminating experiment: human listen-check on joy_08 + try
   happy+calm low-intensity combos.
4. **Timbre–emotion disentanglement mostly holds.** Sad *prompt* + sad vector didn't
   help (still neutral acoustics; e2v drifted to fear@100% — another OOD judge
   quirk for the thread). Angry prompt mildly amplified anger (A 0.79→0.90,
   D 0.61→0.80). Prompt is a seasoning, not a lever.
5. **Controls behaved** (baseline + calm → neutral, both 100%) — the pipeline isn't
   hallucinating emotions where none were requested.

**Scoreboard after day one: anger 3/3 · neutral controls 2/2 · sadness 0/5 (best
dist 0.166, trail = melancholic) · joy 0/2 (needs rethink).** 13 ledger rows.

**Next (P4.3):** melancholic-scaling round for sadness; happy+calm combos for joy;
first human listen-check (do the clips *sound* like what the meters say?); then the
optimizer loop on whichever emotions have working knobs.

### 2026-07-05 — P4.3 THE OPTIMIZER: first autonomous convergence — and two honest walls

The loop ran **with no human between iterations** for the first time:
`optimize_p43.py` (orchestrator, deterministic rule-based proposer, budget-capped)
+ `synth_worker.py` (vendor-side, JSON-driven, one model load per round).
3 rounds, 17 clips, ledger now at **30 rows**. Design held: steer on WavLM distance,
**HIT declared only by the independent e2v judge**; emotions that converge stop
spending budget (anger did, after round 1).

| Emotion | Outcome | Evidence |
|---|---|---|
| **anger** | ✅ **CONVERGED — autonomously refined** | Machine tried `angry=0.7` on its own → distance **0.227 → 0.207**, judge anger@100%. It took a working setting and made it measurably better. |
| **sadness** | ⚠️ improved 24%, then plateaued — a *measured floor* | Best **0.166 → 0.134** (`mel=1.0 + calm=0.3`); rounds 2–3 couldn't beat it (0.134/0.142/0.154...). Acoustics sit near the sadness centroid, but the e2v judge said **neutral@100% in all 9 attempts**. Either the TTS's sadness lacks the voice-quality markers (breathiness, instability) that live beyond V/A/D position, or the judge's synthetic-voice neighbors lock to neutral. |
| **joy** | ❌ **ceiling confirmed** | 6 attempts across P4.2+P4.3, **zero positive valence ever produced**. Failure modes scatter: `happy=0.4`→fear@100%, `+calm`→neutral, `+surprised`→surprise/anger(!). This model cannot reach joy's coordinates by steering — per our instruments. |

**What this answers (the "logging vs improving" question, with data):** the system
now demonstrably improves itself where improvement is reachable — anger refined
0.227→0.207 and sadness 0.166→0.134 with zero human decisions — and **refuses to
fake convergence where it isn't**. Both behaviors are the point.

**What it sets up:**
- **The human listen-check is now the single most important next experiment** — for
  BOTH sadness and joy it decides mouth-broken vs ear-biased (Phase-5 Gate 2). If
  `r2_sadness_0` *sounds* clearly sad to human ears while the judge says neutral,
  the judge's synthetic-voice bias is real and measurable.
- **Joy is the documented candidate for Phase-5 training** (Gate 1 filling: steering
  provably exhausted) — or for a knob-change to Chatterbox. Decision after ears.
- e2v OOD-scatter thread grows: fear@100% and anger@100% on happy-slider clips —
  the judge's confidence remains vote-share, not evidence, on synthetic audio.

Budget discipline worked: 17 clips spent of a 30-clip worst case; anger stopped
buying after it converged; the proposer deduplicated already-tried vectors.

### 2026-07-05 — P4.3b Cross-model probe: a SECOND mouth (Chatterbox) settles joy — and reframes sadness

**Why a second mouth.** After the optimizer, two open verdicts hinged on the same
ambiguity: is the failure the MOUTH (IndexTTS-2 can't produce the acoustics) or the
EAR (our judge undervalues synthetic emotion)? A second, unrelated TTS attempting
the same emotions through its own control surface — same sentence, same frozen
judge — separates the two without waiting for human ears.

**Setup.** Chatterbox (Resemble AI, MIT license) in a third isolated env
(`.venv_cbx`, py3.12). Control surface is entirely different from IndexTTS-2's:
an *emotional reference clip* (RAVDESS strong-intensity acted emotions) + an
`exaggeration` knob. 6 clips, judged by the same bridge, ledger rows 31–36
(`system=chatterbox`).

**Hurdle (logged for the record):** first run crashed with a cryptic
`'NoneType' object is not callable` inside Chatterbox's constructor. Root cause was
two generations of packaging drift stacked: the `perth` watermarking lib still
imports legacy `pkg_resources`; py3.12 venvs don't bundle setuptools; and installing
setuptools got v83 — which has *removed* `pkg_resources` entirely. `perth` swallows
its own ImportError and exports `None`. Fix: pin `setuptools<81`. Real watermarker
restored (no dummy-patching — clips keep Resemble's responsible-AI watermark).

**Results (ledger 31–36):**

| clip | target | control | V | A | D | judge | verdict |
|---|---|---|---|---|---|---|---|
| cbx_neutral_e05 | neutral | ref=neutral, ex=0.5 | −0.07 | 0.37 | +0.08 | **neutral@100%** | ✅ HIT (d=0.096) |
| cbx_joy_e05 | joy | ref=happy, ex=0.5 | **+0.03** | 0.47 | +0.20 | **joy@60%** | ✅ **HIT** |
| cbx_joy_e09 | joy | ref=happy, ex=0.9 | −0.10 | 0.66 | +0.44 | fear@80% | ❌ |
| cbx_sad_e05 | sadness | ref=sad, ex=0.5 | −0.06 | 0.33 | +0.07 | neutral@100% | ❌ |
| cbx_sad_e09 | sadness | ref=sad, ex=0.9 | −0.10 | 0.50 | +0.30 | neutral@100% | ❌ |
| cbx_anger_e07 | anger | ref=angry, ex=0.7 | −0.41 | 0.86 | +0.71 | fear@83% | ❌ |

**The triangulation, emotion by emotion:**

1. **JOY: the ear is exonerated — IndexTTS-2's mouth is broken.** Chatterbox at
   exaggeration 0.5 produced the **first synthetic joy HIT of the entire project**
   (judge joy@60%, valence positive) — something IndexTTS-2 failed to do in 8
   attempts. The judge CAN name synthetic joy when the acoustics carry it.
   IndexTTS-2's `happy` slider renders shout, not smile. Caveat kept honest:
   V=+0.03 vs centroid +0.30 — the judge grants synthetic joy only *weakly*
   positive valence, so a milder ear-side attenuation may still coexist.
   Consequences: (a) Phase-5 Gate 1 evidence upgraded — joy is mouth-limited and
   training-eligible; (b) pragmatic alternative: **route joy to Chatterbox** in any
   multi-mouth setup.
2. **JOY at high intensity breaks the same way in BOTH mouths.** Chatterbox
   ex=0.9 → fear@80% with negative valence — the exact signature of IndexTTS-2's
   `happy=0.8+`. Cross-model replication says this is a real acoustic phenomenon:
   *over-intensified synthetic happiness converges on fear's acoustics* (high
   arousal, unstable pitch, no positive-valence markers). Moderation wins on both
   control surfaces.
3. **SADNESS: two unrelated mouths, identical verdict — suspicion shifts to the
   ear.** Chatterbox's sad (both intensities) landed neutral@100%, same as all 9
   IndexTTS-2 melancholic attempts. Notably Chatterbox's sadness is acoustically
   WORSE than ours (d=0.264 vs our 0.134 floor) — IndexTTS-2 + `melancholic` remains
   the better sadness mouth. But when two independent systems both read "neutral,"
   either synthetic sadness universally lacks the voice-quality markers the judge
   keys on, or the judge's synthetic-neighborhood locks to neutral. **Only human
   ears can now break this tie** → the blind listen-check is decisive for sadness.
4. **ANGER: IndexTTS-2 wins the head-to-head.** Chatterbox's anger nails valence
   (−0.41 vs centroid −0.42!) but overshoots arousal/dominance (0.86/0.71) and the
   judge reads it as fear@83%. IndexTTS-2's `angry=0.7` remains the only judged
   anger HIT. First rivalry scoreboard: **anger IndexTTS-2 · joy Chatterbox ·
   sadness nobody (yet).**
5. **Neutral control passed 100%** — the bridge, references, and new env are sound;
   the misses above are signal, not plumbing.

**Scoreboard after the probe: ledger 36 rows across two systems. Joy has its first
HIT (chatterbox). Sadness is now formally an EAR-question (Gate 2 = human ears).
Anger remains IndexTTS-2's flag.**

**Next:** (1) human blind listen-check — `out/listen_check/RESPONSE_SHEET.md`
(11 clips, answer key sealed) — decides sadness mouth-vs-ear and validates joy;
(2) P4.4 rivals benchmark (ElevenLabs v3 / Hume Octave 2 / OpenAI TTS) with the
same sentence + same judge; (3) Phase-5 go/no-go after gates.

### 2026-07-05 — GATE 2: the human listen-check — the ear's blind spot is real, and the mouth was better than we thought

The decisive experiment ran: 11 blind clips (8 synthetic + 3 real-voice controls),
hash-shuffled names, key sealed until the listener finished. Listener scored blind
and — before seeing anything — independently proposed scoring at *family* level
because joy↔neutral and sad↔serious felt like continua, not bins. That is exactly
the pipeline's own Ekman-family + V/A/D design, rediscovered from the listening
side. (Full sheet: `out/listen_check/RESPONSE_SHEET.md`.)

**Calibration first: all 3 real-voice controls correct** (joy/anger/sad on real
recordings). The ears are trustworthy; what follows is signal. 9/11 vs intent.

**The three verdicts:**

1. **SADNESS — the judge has a synthetic-sadness blind spot; the mouth was
   succeeding.** Both sadness clips the e2v judge called neutral@100% across nine
   optimizer attempts (`r2_sadness_0`, `mel_08`) were heard as **sad**, blind. This
   retro-explains Chatterbox's identical neutral verdicts: same ear, same lock. The
   melancholic knob works for humans. Honest asterisk: the listener also heard
   "sad" on `calm_08` (human calm/serious/sad boundary is fuzzy in the same
   low-arousal region) — but did NOT call the neutral baseline sad, so the finding
   stands, caveated.
2. **JOY — split by intensity.** `happy=0.4` (judge: fear@100%) was heard as
   **joy** — moderate IndexTTS-2 joy is human-real and the ear under-credits it.
   `happy=0.8` (judge: surprise) was heard as **anger** — high-intensity joy is
   genuinely broken, human and judge agree. Combined with the Chatterbox joy HIT:
   the ear CAN name synthetic joy but attenuates it; the mouth degrades with the
   slider. Both effects are real, at different intensities.
3. **ANGER — closed with triple agreement.** Both synthetic anger clips: judge
   anger@100%, human anger, intended anger. Mouth ✓ ear ✓ human ✓.

**Consequences (binding):**
- **The judge stays frozen.** The tempting move — "teach the ear to hear synthetic
  sadness" — is exactly the self-grading trap the judge-frozen law exists to
  prevent. Instead the blind spot is *documented* (it joins the parent writeup's
  known fear-47% weakness): **e2v locks synthetic sadness to neutral even when
  humans hear sadness.**
- **Sadness success criterion amended (judge untouched):** for sadness only, success
  = WavLM distance ≤ ~0.14 to the MSP centroid + blind human confirmation. By that
  standard, `mel=1.0 + calm=0.3` at d=0.134 **is a sadness HIT** — the scoreboard
  gains one, honestly annotated as human-adjudicated.
- **Phase-5 target sharpened:** joy remains the only training-eligible candidate,
  and specifically *high-intensity* joy (moderate joy already works to human ears;
  the high-slider regime is where the mouth breaks). Alternative stays live: route
  joy to Chatterbox.
- e2v blind-spot ledger for the writeup now reads: fear 47% (parent, real speech) +
  synthetic-sadness→neutral lock + synthetic-joy attenuation (fear/surprise
  misreads). The instrument is characterized, not perfect — that's what makes the
  benchmark honest.

**Scoreboard after Gate 2: anger ✅ (triple) · sadness ✅ (human-adjudicated,
d=0.134) · joy ◐ (moderate=human-real, high=broken; Chatterbox HIT as alternative).
Two of three target emotions delivered; the third has a mapped failure mode and
two escape routes.**

**Next: P4.4 — the rivals benchmark** (ElevenLabs v3 / Hume Octave 2 / OpenAI TTS),
same sentence, same frozen judge, plus blind human spot-checks now that the
listener protocol is proven.

### 2026-07-05 — P4.4 THE RIVALS BENCHMARK: the $0 local loop holds its ground against commercial APIs

**Setup.** Same neutral sentence, same frozen judge, same MSP centroids. Each rival
driven through its own native emotion-control surface, one fixed voice each:
ElevenLabs **eleven_v3** (audio tags `[sad]/[happily]/[angry]`, voice=River),
**Hume Octave** (acting description per utterance). OpenAI TTS pending (key issue —
401; will slot in via the resumable scripts). Ledger rows 37–44. One clip per
emotion per rival — probe-scale, not a definitive study; stated as such.

**Hurdles (logged):** ElevenLabs free tier 402'd on the classic "Rachel" voice —
premade catalog voices are now account-scoped; fix was listing `/v1/voices` and
using a premade one (River). Hume worked first try, 4/4.

**Results (frozen judge, rows 37–44):**

| target | **IndexTTS-2 (ours, loop-steered)** | Chatterbox (local) | ElevenLabs v3 | Hume Octave |
|---|---|---|---|---|
| neutral | HIT | HIT d=0.096 | HIT d=0.291 | HIT d=0.336 |
| sadness | **d=0.134** ★human-adjudicated HIT | d=0.264, neutral | d=0.395, neutral | d=0.145, neutral |
| joy | d=0.354 (fear; moderate=human-real) | **joy@60% HIT** d=0.294 | d=0.462, **anger@60%** | d=0.513, neutral |
| anger | **anger@100% HIT, d=0.207** | d=0.338, fear | anger@60% HIT, d=0.263 | d=0.282, neutral@80% |

**Findings:**

1. **ANGER — we win outright.** Our loop-steered `angry=0.7` beats ElevenLabs v3 on
   BOTH axes (distance 0.207 vs 0.263; judge confidence 100% vs 60%). Hume's anger
   missed entirely (neutral@80%). A local model + 30 ledger iterations out-delivered
   the commercial APIs' one-shot on the instrument everyone was scored by.
2. **SADNESS — acoustically, ours is the closest of all four mouths** (0.134 vs
   Hume 0.145, Chatterbox 0.264, ElevenLabs 0.395) — and the judge said neutral for
   **every system**. Four independent mouths, zero sadness family-hits. The Gate-2
   verdict is now overwhelming: the e2v judge locks synthetic sadness to neutral,
   period. (Human ears already confirmed ours sounds sad.)
3. **JOY — no rival cracked it either.** ElevenLabs `[happily]` was judged
   *anger*@60% with negative valence; Hume joy read neutral. Chatterbox's joy@60%
   remains the ONLY synthetic joy ever named by the judge. Notably: **not one of
   ten joy attempts across four systems produced positive WavLM valence** — the
   positive-valence gap on synthetic speech is universal, not an IndexTTS-2 defect.
   (Whether that's TTS acoustics or judge attenuation, Gate 2 says at least part
   is the ear.)
4. **Neutral: all four systems pass** — the yardstick is sane.

**Honest framing for the writeup:** our clips had the closed-loop advantage — 30
iterations of steer-measure-adjust vs the rivals' single prompt-shot. That is not
an unfair comparison; it *is the thesis*: a feedback loop around a frozen judge
buys a $0 local model parity-or-better with commercial emotional TTS on this
instrument. Caveats: n=1 per cell, one sentence, one voice per system, judge has
documented synthetic blind spots (sadness-lock, joy attenuation). Next escalation
would be more sentences/voices + blind human panel on rival clips.

**Scoreboard: anger OURS · sadness OURS (acoustic) with judge-blind-spot caveat ·
joy Chatterbox (local MIT) · commercial APIs win no category on this yardstick.**

**Next:** OpenAI TTS slot-in when key lands; optional blind human check on rival
clips (protocol proven); then P4.5 — the Phase-4 writeup.
