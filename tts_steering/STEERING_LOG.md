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

### 2026-07-05 — P4.4 addendum: OpenAI TTS slots in — the flattest mouth of the five

Key fixed, resumable scripts did their job (only the 4 missing clips ran). Ledger
rows 45–48. `gpt-4o-mini-tts`, voice=alloy, steered via the `instructions` prompt.

| target | V | A | D | d | judge |
|---|---|---|---|---|---|
| neutral | −0.04 | 0.41 | +0.20 | 0.178 | neutral@100% ✅ |
| sadness | −0.14 | 0.27 | −0.00 | 0.199 | neutral@100% |
| joy | −0.11 | 0.44 | +0.26 | 0.440 | neutral@100% |
| anger | −0.13 | 0.43 | +0.23 | **0.434** | neutral@100% |

**Finding: OpenAI's instruction-following barely moves the acoustics.** All four
clips cluster near neutral; its "furious" (d=0.434) is more than 2× farther from
the anger centroid than our steered clip (0.207) and its joy/anger were judged
neutral@100% — the emotional dynamic range is minimal on this voice. Its sadness
leans the right way (arousal drops to 0.27) but stays in the neutral basin. On
this yardstick OpenAI TTS optimizes for clarity, not affect.

**FINAL five-system scoreboard (frozen judge, one sentence, probe-scale):**

| | ours (steered) | Chatterbox | ElevenLabs v3 | Hume | OpenAI |
|---|---|---|---|---|---|
| neutral | ✅ | ✅ 0.096 | ✅ 0.291 | ✅ 0.336 | ✅ 0.178 |
| sadness | **0.134** ★human | 0.264 | 0.395 | 0.145 | 0.199 |
| joy | 0.354 | **✅ HIT** 0.294 | 0.462 (anger!) | 0.513 | 0.440 |
| anger | **✅ HIT@100% 0.207** | 0.338 | ✅ HIT@60% 0.263 | 0.282 | 0.434 |

Anger: OURS. Sadness acoustics: OURS (judge-blind-spot caveat, human-confirmed).
Joy: Chatterbox (local, MIT). The two local, free systems took every category;
neither commercial API won one. Benchmark closed — on to P4.5, the writeup.

### 2026-07-05 — THE CATCH: stepping back — what this benchmark does NOT show

Right after the final scoreboard landed, the project owner asked the question a
sharp interviewer would ask: *"Don't you feel it's too good to be true? We beat
multimillion-dollar infrastructure with a little tweaking and a feedback loop?
There must be a catch."* There is. Five, recorded here **before** the writeup, so
the record shows the skepticism came from inside the project, not from a reviewer.

1. **We won on our own scoreboard, in our own stadium.** The yardstick is our WavLM
   judge and our MSP centroids. We iterated against that metric ~30 times; every
   rival got one prompt-shot. Anyone allowed to iterate against a fixed metric will
   beat one-shot competitors *on that metric* — Goodhart's law working in our favor.
   The result is a claim about **loops and instruments**, not about who has the
   better TTS.
2. **We never measured what the rivals actually sell.** Naturalness, cloning
   fidelity, 30 languages, sub-300 ms streaming, stability across arbitrary text,
   pronunciation, uptime. Our benchmark scores none of it. Their clips almost
   certainly *sound more human* than ours — no MOS/naturalness test was run. A
   slightly robotic clip with exaggerated acoustics can win our metric while losing
   every human preference test.
3. **OpenAI's flatness is plausibly a product choice, not incapacity.** Overacted
   emotion is cringe in a product that reads text aloud a billion times a day.
   Commercial TTS deliberately underplays; we measured "won't chew the scenery"
   and scored it "can't act."
4. **We may have handicapped ElevenLabs ourselves.** The voice used (River —
   "Relaxed, Neutral, Informative") was chosen as a clean neutral base — which is
   like testing emotional range on a newsreader. A theatrical catalog voice plus
   the same 30-iteration budget we gave ourselves could tell a different story.
5. **n=1 per cell.** One sentence, one voice per system, one clip per emotion.
   ElevenLabs anger@60% vs ours @100% is, at this sample size, an anecdote wearing
   a table's clothes.

**What survives the deflation (the actual value):**
- The **methodology**: frozen judge, anti-circularity law, ledger that keeps every
  miss, deterministic optimizer, blind human gate with real-voice controls.
- The **judge blind-spot discovery** — synthetic sadness → neutral across five
  independent mouths, broken only by blind human ears. That finding is solid.
- The **anger steering map** on IndexTTS-2 — monotonic, repeatable, both backbones
  agree at every strength.
- The **universal synthetic-joy valence gap** — ten attempts, four synthetic
  mouths, zero positive WavLM valence. Whatever its cause, it's real and measured.

**THE DELICATE CLAIM (the sentence the writeup must not exceed):**
> A closed feedback loop around a frozen perceptual judge lets a $0 local TTS
> match or exceed commercial emotional TTS **on that judge's scale**, at
> probe-scale (n=1/cell, one sentence) — while the loop's ledger doubles as a
> characterization of both the TTS's control surface and the judge's own blind
> spots.

Nothing more. The moment it becomes "we beat ElevenLabs," it stops being true.

---

## WHERE WE STAND (as of 2026-07-05 — P4.0 through P4.4 complete)

**Pipeline built and proven:** IndexTTS-2 (mouth, MPS) → bridge.py (subprocess-only)
→ WavLM V/A/D (steering signal) + frozen e2v (family verdict) → deterministic
optimizer → ledger. Four isolated envs, zero shared imports, engine repo untouched.

**The ledger: 48 rows, 5 systems** (indextts2 smoke+sweep+p43 · chatterbox ·
elevenlabs · hume-octave · openai-tts). Every miss kept. Artifacts: `out/loop_ledger.csv`,
`out/p43/summary.json`, `out/p44/manifest.json`, `out/listen_check/` (sheet + sealed key).

**Findings, in one place:**
1. Anger is fully solved and won: monotonic steering, autonomous convergence
   (`angry=0.7`, d=0.207, anger@100%), triple agreement (judge+human+intent),
   beats all four other systems.
2. Sadness is delivered but the judge can't see it: best clip d=0.134 (closest of
   all five systems), heard as sad by blind human ears; e2v locks ALL synthetic
   sadness to neutral (5/5 systems) — a characterized instrument blind spot.
   Success criterion amended for sadness only: WavLM distance + blind human
   confirmation (judge stays frozen).
3. Joy splits by intensity: moderate IndexTTS-2 joy is human-real but
   judge-attenuated; high-intensity joy collapses into fear/anger acoustics in BOTH
   local mouths (cross-model replication). Chatterbox e=0.5 holds the only judged
   joy HIT. No system, commercial or local, produced positive WavLM valence (0/10).
4. The loop improves itself where improvement is reachable (anger 0.227→0.207,
   sadness 0.166→0.134, zero human decisions) and refuses to fake convergence
   where it isn't (joy, honestly failed).
5. Judge OOD behavior on synthetic audio is now a documented thread: sadness→neutral
   lock, joy attenuation (fear/surprise misreads), confidence = vote-share not
   evidence. Joins the parent's fear-47% in the instrument-limitations ledger.

**Laws still binding:** judge frozen (no retraining the ear on loop data, ever) ·
bridge-not-merge (subprocess/CLI/JSON only) · every clip a ledger row · misses kept ·
no magic numbers (centroids from 137k MSP; thresholds data-derived) · claims sized
per THE CATCH above.

**Open threads:**
- **P4.5 writeup** — next deliverable. Structure: thesis (the delicate claim) →
  method → scoreboard → the catch → blind-spot discovery → what we'd do with a
  bigger budget.
- **Phase 5 (gated)**: reward-guided fine-tuning of IndexTTS-2 for high-intensity
  joy. Gate 1 (steering exhausted) ✅ filled. Gate 2 (human listen-check) ✅ run.
  Gates 3–4 (training-grade data variety; reward-hacking mitigation plan) open.
  Alternative that needs no training: route joy to Chatterbox (MIT).
- StyleTTS2 research angle (task-vector papers) — untouched, optional.
- Rival clips as blind human listening material — protocol proven, optional.

### 2026-07-05 — P4.4b THE FAIR REMATCH: every system gets the loop — and our crown mostly melts

THE CATCH demanded a fair fight; this is it. **Every rival got the same closed
loop** (3 seeds → judge → deterministic error-keyed refinements, ≤5 judged
attempts/emotion), **voice selection by the meter** (ElevenLabs: Charlie/Harry/
Laura tested; OpenAI: ash/coral/ballad tested), richer native control surfaces
(multi-tags, acting descriptions, method-actor instructions, exaggeration×cfg
grid), and a **generalization round** — every system's best config, including
ours, re-judged on 2 sentences no steering ever saw. Same frozen judge, same
centroids, semantics untouched. Ledger rows 49–130 (52 steering + 30
generalization). Harness: `fair_p44b.py` + `cbx_worker.py`.

**Hurdles:** (1) mid-run network outage killed the generalization round AND
revealed the ledger was only written at the end — rows from 3 completed rounds
would have been lost. Fix: per-round ledger persistence + control-strings
reconstructed for cached clips; full resume worked. (2) e2v judge also needs
network at init (modelscope ping) — noted as an ops constraint.

**FAIR steering results (S1, ≤5 attempts each):**

| emotion | ours (historical) | Chatterbox | ElevenLabs v3 | Hume | OpenAI |
|---|---|---|---|---|---|
| anger | HIT@100% d=0.207 | **HIT@60% d=0.171** | HIT@59% d=0.380 | HIT@100% d=0.256 | **HIT@80% d=0.187** |
| joy | ✗ 0.354 | ✗ 0.309 | **HIT@80% d=0.144** | ✗ (anger) | ✗ 0.495 |
| sadness | 0.134 ★human | 0.162 | 0.357 | 0.210 | 0.240 |

**Generalization (best configs, 2 unseen sentences; family hits / 6 clips):**
ours **3/6** · OpenAI **3/6** · ElevenLabs 2/6 · Chatterbox 2/6 · Hume 1/6.
Nobody is stable. Many off-target clips judged joy@80% on the new sentences —
the judge shows a joy-attractor basin there (echoes the parent project's MELD
joy-absorption finding). And two firsts: **the judge named synthetic sadness for
the first time ever** — ours (sadness@40%) and Chatterbox (sadness@60%), both on
sentence S3 only. The sadness lock is real but *sentence-dependent* — softer
than we claimed.

**RETRACTIONS (what the fair fight takes away):**
1. ~~"Anger: OURS"~~ → **anger is a four-way tie.** Given the loop, Chatterbox
   (0.171) and OpenAI (0.187) beat our distance (0.207); we keep only the
   confidence edge (100% vs 60–80%, shared with Hume). Our previous win was
   mostly the iteration asymmetry, as suspected.
2. ~~"Commercial APIs win no category"~~ → **ElevenLabs v3 wins joy outright**
   (d=0.144@80%, best joy of the project) once given an expressive voice + tag
   combos. OpenAI's "flatness" was half our voice choice: coral+loop found
   anger d=0.187@80% where alloy one-shot gave 0.434 neutral.
3. ~~n=1 doesn't matter much~~ → it mattered enormously. Steering-round winners
   are S1-specialists; cross-sentence stability is poor for every system
   including ours.

**What survives — and is strengthened:**
1. **The loop is the product, not the model.** It improved every control surface
   it touched (sliders, tags, descriptions, instructions, exaggeration) in ≤5
   attempts — that's the transferable engineering result.
2. **The instrument characterization replicates across 5 systems:** the
   sadness→neutral lock (now with its sentence-dependence mapped), the
   joy-attractor on certain texts, and the **moderation law** — every
   max-intensity refinement overshot into fear/surprise (Hume "ecstatic" →
   d=0.956 surprise; Chatterbox ex=1.0 → fear@100%), on every system, every time.
3. Ours still holds: best sadness acoustics (0.134, human-confirmed), highest-
   confidence anger, and the only system with a fully mapped control surface.

**THE DELICATE CLAIM, revised smaller (supersedes the previous):**
> A deterministic feedback loop around a frozen perceptual judge finds the best
> emotional operating point of ANY TTS control surface — local or commercial —
> in ≤5 judged attempts, and in doing so characterizes both the mouths (all
> five share a high-intensity collapse) and the ear (sentence-dependent sadness
> lock, joy attractor). On this instrument, a $0 local stack remains
> competitive with commercial APIs, winning no category cleanly and losing
> only joy.

That is what we actually gained. It is smaller than the P4.4 headline and it is
true — which makes it worth more.

### 2026-07-05 — P4.5: writeup shipped. PHASE 4 CONCLUDED.

[`WRITEUP.md`](WRITEUP.md) condenses this log into the presentable story: the
claim (v3, sized honestly), the loop, the fair scoreboard, the human gate, the
catch, and the gated next steps. A snapshot of the full ledger
([`loop_ledger.csv`](loop_ledger.csv), 130 rows) is now committed alongside it —
the data travels with the claims. Root README links the phase.

Final state: 2 days, 5 systems, 130 judged clips, 12+ hurdles logged, 1 headline
result retracted by our own fairer test, 2 instrument blind spots discovered and
characterized, 1 human blind protocol run with 3/3 controls. The project ends the
way it was designed to: with the smallest claim that is fully true.

---

## RECOMMENCEMENT — 2026-08-08 (after a deliberate one-month pause)

**Why the pause:** Phase 4 concluded gracefully on 2026-07-05 (P4.5 writeup, claim
v3, 130-row ledger, retractions published). The month since went to packaging and
applications — the repo became the portfolio it was built to be (two-projects
framing, ARTICLE.md, CI check, HF weights pointer, founding sketch). No
experimental work happened in the gap; the working tree stayed clean. This entry
exists so the record shows the seam: everything above it is the concluded Phase 4;
everything below is the next campaign.

**Why we commence at P4.6 (the transfer map):**
1. It was the log's own declared next step at conclusion ("no-regret").
2. The fair rematch's sharpest finding was *sentence instability* — best systems
   managed only 3/6 family hits on unseen sentences. P4.6 maps exactly that
   weakness: how the vector→emotion transfer function varies across sentence
   types (~20 varied sentences × known-good vector regions ≈ 500 clips, local,
   $0, judge frozen as ever).
3. Its ledger IS Phase-5 Gate 3 (training-grade data variety) — so the training
   question advances as a side effect, not a gamble.

**Still open after P4.6 (the standing map):** Phase 5 fine-tune behind Gate 4 +
feasibility spike (fallback: ledger-distilled control policy) · engine serving
layer (stress-test spec) · trajectory rebuild on the fine-tuned model · RFE/MRE
consumer projects · optional: StyleTTS2, rival blind-listen pack, e2v fine-tune.

Laws unchanged and binding: judge frozen · bridge-not-merge · every clip a ledger
row · misses kept · claims sized to evidence.

---

## P4.7 — THE A/B/C TEST: three proposers, one mouth, one frozen judge (2026-08-09)

**Detour note:** the recommencement entry named P4.6 (transfer map) as next; the
user redirected first to a question worth answering before spending 500 clips:
*does a smarter proposer beat the deterministic rules?* Three arms, equal budgets:

- **ARM A** `indextts2-abcA-det` — the incumbent P4.3 deterministic rules.
- **ARM B** `indextts2-abcB-r1` — DeepSeek R1 (OpenRouter) reasons over the arm's
  own judged history and proposes vectors. Honest-forfeit on parse failure; spend
  hard-capped $2.50 (actual: **$0.13**).
- **ARM C** `indextts2-abcC-scaffold` — the user's **MSP equation scaffold**
  ("training wheels"): steer toward the certified acoustic *recipe* of each
  emotion, fit from real human speech, instead of a bare V/A/D point.

**The scaffold (new instrument, `scaffold_msp.py` → `out/scaffold_msp.json`):**
statistics-only fit from 132,276 MSP train clips (license-safe, nothing
redistributed, nothing trains the mouth): 32 equation terms selected by Fisher
score from the engine's own 111 classical features (categories excluded:
text/timing/absolute-level). Per-family templates + *direction-from-neutral*
vectors; clip score = weighted cosine between (clip − synth-baseline) and
(family − neutral)_MSP — differencing both sides cancels the channel gap between
podcast audio and clean TTS. **Validated before use** on 24,909 held-out MSP dev
clips: 33.3% nearest-template 5-way accuracy vs 20% chance (anger 67.9%, sadness
34.2%, joy 18.3%, surprise 14.5%). Sanity check on our own mouth: old `angry_08`
clip scores +0.76 vs certified anger direction, +0.13 vs sadness.

**New lawful constant:** surprise centroid **(+0.05, 0.64, +0.26)** — derived by
exactly reproducing the original centroid computation on MSP Train labels
(existing three verified to the third decimal; surprise n=3,220). Surprise had
never been a steering target before this run.

**Pre-registered before the run (both confirmed by it):**
1. The hot emotions are entangled in scaffold space — for the anger target the
   *happy* knob aligned +0.88, above the angry knob itself (+0.82).
2. No knob moves features convincingly toward certified sadness (best: sad knob
   at +0.28).

**Setup:** 4 targets × 3 arms, 3 rounds × 2 candidates, stop-on-HIT, arms blind
to each other, S1 + neutral prompt, frozen e2v judge = the only law. Calibration
(disclosed, unscored): P4.2 single-knob clips + 3 new probes (afraid/disgusted/
surprised @0.8) → knob→feature response map. 42 scored clips, ledger 132 → 174.

### Scoreboard (frozen judge)

| Target | A (rules) | B (R1) | C (scaffold) |
|---|---|---|---|
| **anger** | **HIT@80% att.1, d=0.244** | HIT@100% att.1, d=0.393 | no HIT (fear@82%, surprise@80%) |
| **surprise** (new) | HIT@80% att.2, d=0.482 | HIT@80% att.1, d=0.692 | HIT@60% att.2 · best d=**0.179** |
| **joy** | no HIT, best d=0.415 | no HIT, best d=0.507 | no HIT, best d=**0.402** |
| **sadness** | no HIT, best d=**0.163** | no HIT, d=0.210 | no HIT, d=0.206 |

### Findings

1. **SURPRISE IS A NEW CONFIRMED STEERABLE FAMILY — 3/3 arms produced
   judge-confirmed HITs on first exposure.** The mouth's steerable set is now
   anger + surprise (reliable), joy (split), sadness (locked). The `surprised`
   slider at 0.8–1.4 is the direct route (A att.2, B att.1); C got there via
   happy+angry blend.
2. **The incumbent rules remain the best proposer overall.** A matched or beat
   the challengers on hits and won sadness/anger distance. Four lines of
   deterministic rules ≥ a frontier reasoning model, on this mouth, at this
   budget. R1 was competitive (2 attempt-1 HITs) but wilder (worse distances)
   and found nothing the rules didn't.
3. **The scaffold's compass is real but valence-blind between hot emotions**
   (pre-registration #1 confirmed in action): steering anger by certified-recipe
   alignment, C pushed the *happy* knob and the judge heard fear/surprise. Yet C
   scored the two best distances of the entire run (surprise 0.179, joy 0.402) —
   the training wheels DO pull clips into the right acoustic *region*; they
   cannot pick the right *neighbor* within it.
4. **Third witness on the sadness lock — and it dissents in an informative way.**
   The melancholic=1.0 clip (best V/A/D distance, humans blind-ruled it SAD in
   P4.3) scores **−0.09** against the certified MSP sadness recipe, while plain
   sad=0.4 scores +0.42. So: WavLM-distance says mel-clips are sad-like; humans
   agree; but feature-level comparison with *real conversational* sadness sides
   with the e2v judge — the acoustics are NOT natural-sadness acoustics. Reading:
   the mouth produces *performed/acted* sadness cues (slow, flat, low), not the
   voice-quality signature of genuine conversational sadness, and e2v keys on
   exactly that gap. The sadness miss is jointly mouth- and ear-shaped, not
   ear-only. (Claim v3 unchanged; this sharpens the diagnosis.)
5. **Disclosed harness inefficiency:** arm C under-spent its budget (anger 2/6,
   sadness 3/6 attempts) — residual-driven proposals starved on dedup/clamp
   (once it even proposed the zero vector). C's numbers are therefore a floor,
   not a ceiling. Fix before any rematch: fallback candidate pool when the
   residual step degenerates.
6. **Cost of the whole question: $0.13 of R1 + ~40 min MPS.**

**Verdict for the record:** the A/B/C question is answered — keep the
deterministic rules as the loop's proposer; keep the scaffold as a *diagnostic
instrument* (its per-clip recipe-alignment column is new evidence the ledger
never had); R1 adds no lift at this budget. The scaffold's true value was not
steering but *witnessing*: it independently corroborated the judge on synthetic
sadness and quantified exactly how far each clip sits from certified human
emotion. Next: P4.6 transfer map as planned, now with the scaffold column
recorded for every clip.

Laws unchanged: judge frozen · every clip a ledger row (174) · misses kept ·
claims sized to evidence · MSP used as statistics only.

---

## P4.8 — GATE 1: the scaffold must earn the training-wheel role before any clip is spent (2026-08-10)

**User's discipline, pre-registered:** test the scaffold C *separately*, at $0,
no synthesis — and only if it demonstrably picks meaningful directions does it
keep the training-wheel role (and only then explore foundations beyond MSP,
"somewhere more prominent for emotional calibration"). Amends the P4.7 verdict:
"retired" → **v1 superseded; question tested properly here.**

**Gate 1 (free):** with the mouth's measured knob→feature responses, does
hill-climbing the scaffold score rank an emotion-appropriate knob first per
target? Pass bar (pre-registered): 3/4 targets sane, anger mandatory (v1's
exact failure: it ranked `happy` first for anger).

**Round 1 — scaffold v2 on MSP foundation** (`scaffold2_msp.py`): added the
missing Fourier harmonics — pairwise CONTRAST directions (anger−joy, …), blend
weight λ fit by grid on held-out MSP dev (no magic numbers; λ=0.25). Dev
validation improved modestly: hot-3-way 39.6%→43.2% (chance 33%).
**Gate 1: FAIL (2/4).** Anger still picks `happy` (S +1.08 vs +1.00) — even the
contrast term rates the happy knob's output as more anger-vs-rivals than the
angry knob's own. Natural conversational speech carries the hot emotions too
subtly; λ couldn't be pushed higher without hurting held-out accuracy.

**Round 2 — scaffold-R on RAVDESS foundation** (`scaffold_ravdess.py`): the
user's foundation hypothesis tested — acted speech is emotionally *prominent*
(professional actors, validated labels, exaggerated delivery, same 111-feature
language, neutral anchor = 288 RAVDESS neutral/calm from the same channel).
Validation: hot-3-way separation jumps to 63.9% (resubstitution, labeled as
such) with balanced recalls — the prominent foundation genuinely carries the
valence slice. **Gate 1: FAIL (2/4), but differently:** anger FIXED (`angry`
ranks first — the original failure cured), joy sane, but sadness broke (acted
sadness is *stage* sadness — loud, projected, so its direction is
arousal-heavy and the `angry` knob matches it) and surprise still picks happy.

**Root cause found (the decisive diagnostic):** weighted-cosine similarity
matrix of the mouth's own knob effects in scaffold feature space:

| | happy | angry | surprised | afraid | sad | melanc | calm |
|---|---|---|---|---|---|---|---|
| happy | 1 | **+0.95** | **+0.95** | +0.89 | +0.37 | −0.07 | −0.05 |
| angry | | 1 | **+0.91** | +0.88 | +0.52 | −0.10 | −0.05 |
| surprised | | | 1 | +0.84 | +0.25 | −0.21 | −0.06 |

**The mouth's hot knobs are ~95% collinear in classical feature space.** In
F0/loudness/formant/voice-quality terms, happy = angry = surprised ≈ the same
acoustic action. No foundation can make Gate 1 pass for hot emotions when the
compass's measurement basis cannot see any difference between the knobs it must
choose among. The judge CAN tell the outputs apart (it named surprise vs anger
correctly in P4.7) because e2v lives in a learned embedding space that captures
what IndexTTS-2 actually changes — something beyond these 32 classical terms.
The distinguishable subspace is low-arousal: sad / melancholic / calm / disgusted
effects are mutually distinct (cos +0.17…+0.48) — exactly where the scaffold
made its one real P4.7 discovery (sad-knob +0.42 vs mel-knob −0.09 against
certified sadness).

**Verdict (per the pre-registered gate):**
1. The scaffold does **not** currently earn the steering training-wheel role for
   hot emotions — not because the idea is wrong, and not because of the
   foundation (RAVDESS measurably improved it and fixed anger), but because the
   *measurement basis* is blind to how the mouth's hot knobs differ. No
   synthesis run was spent; the if-and-only-if held.
2. The user's foundation hypothesis is **half-validated with data**: prominent
   acted emotion sharpens the equations (63.9% vs 43.2% hot separation, anger
   knob-choice cured). The remaining failure is not the foundation's fault.
3. The scaffold keeps: its **diagnostic role** (per-clip recipe alignment) and a
   candidate steering niche in the **low-arousal subspace** where knobs are
   distinguishable.
4. **The honest path to the training-wheel vision** (future work, gated):
   refit the scaffold directions in a *learned* embedding space on the steering
   side — e.g., WavLM embeddings (steering ear, so anti-circularity is
   preserved; the e2v judge stays untouched) — fit from RAVDESS/MSP, where the
   hot knobs are presumably separable, since the judge separates them. That is
   a bigger build; it goes behind the same gate: embedding-space Gate 1 first,
   $0, before any clip.

Cost of P4.8: $0.00, zero clips synthesized, three scripts, one root cause.
Laws unchanged: judge frozen · no magic numbers (λ grid-fit; gate pre-registered)
· claims sized to evidence · MSP/RAVDESS statistics only.

---

## P4.9 — GATE 0: derivatives + V/A/D + scaffold (the fluctuation hypothesis) (2026-08-10)

**User's proposal:** the hot knobs collide in static features because static
functionals integrate away the temporal shape — surprise is a *spike*, anger a
*plateau*, joy a *melody*. Add DERIVATIVES (fluctuation features) to the basis;
if the collaboration works, upgrade the whole system around it.

**Gate 0 ($0, pre-registered):** 14 trajectory features (F0 slope/curvature/
spike-rate/rise-fall asymmetry/final slope; energy modulation rate/depth/attack
asymmetry; voicing rhythm) computed from the calibration clips already on disk
(same sentence S1 → text-dependence cancels for knob comparison). Pass bar:
all three hot-knob pairwise cosines < 0.80 in the combined basis.

**Result: FAIL — 2 of 3 pairs stay ≥ 0.80. But the movement is real:**

| hot pair | static (P4.8) | dynamics only | combined | bar |
|---|---|---|---|---|
| happy \| surprised | +0.95 | **+0.52** | **+0.72** | ✓ passes |
| happy \| angry | +0.95 | +0.77 | +0.86 | ✗ |
| angry \| surprised | +0.91 | +0.85 | +0.88 | ✗ |

**Findings:**
1. **Fluctuation carries real discriminative signal** — happy vs surprised
   decollinearized dramatically (0.95 → 0.52 in pure dynamics). The user's
   spike-vs-melody intuition is measurably correct for that pair.
2. **The stubborn core is angry vs surprised (+0.85 even in dynamics):** the
   mouth performs nearly the same temporal gesture for both — a sudden
   high-arousal burst. Acoustically plausible; even humans separate these two
   partly by valence/context, not contour.
3. **Caveat, disclosed:** one deterministic clip per knob-level (IndexTTS-2
   use_random=False) → dynamics estimates have irreducible single-sample noise
   at this gate's cost level.
4. Per the user's own rule ("if it works then we upgrade the rest according to
   it") — it did not pass, so **no systemic upgrade**. Three $0 gates in a row
   (P4.8 ×2, P4.9) now triangulate the same conclusion: the discriminative
   information the frozen judge demonstrably uses lives beyond hand-crafted
   classical + dynamics features. The remaining honest path to the
   training-wheel vision stays the embedding-space scaffold (P4.8 verdict #4),
   behind the same $0-gate-first discipline.

Cumulative cost of the entire scaffold investigation (P4.8 + P4.9): $0.00,
zero clips synthesized, root cause + two half-validated user hypotheses on the
record. Laws unchanged.

---

## P4.9b — THE FOURIER FINGERPRINT LADDER (the user's equations, taken literally) (2026-08-10)

**User's correction, valid:** Gate 0 had summarized derivatives into statistics
— still integrating the shape away. The reels' idea taken literally: the
contour ITSELF is the equation. Each clip's F0 and energy trajectories are fit
as truncated cosine series (DCT-II, K=10, time-normalized):
F0(t) ≈ c0/2 + Σ c_k cos(πkt). The coefficient vector IS the fingerprint;
"derivation in the equation" is exact calculus — d/dt scales harmonic k by k,
so W_deriv (w_k = k²) compares pure fluctuation. Three weightings declared
before running (shape / deriv / combined; disclosed multiplicity).

**The fitted fingerprint equations (RAVDESS, printed in full in
`fourier_gate.py` output)** confirm the spike/plateau/melody intuition
quantitatively: surprise is high-harmonic-rich (−0.94cos3πt +0.95cos4πt — the
spike), anger concentrates in the 1st harmonic (+2.22cos1πt — sustained fall),
joy has the largest swing + strong 5th harmonic (melody), neutral is a smooth
near-pure decline. Held-out actor validation: hot-3-way 48.6–51.0% vs 33%
chance — the fingerprints separate acted emotions.

**The ladder (each rung $0 except the disclosed calibration synthesis):**

| rung | question | result |
|---|---|---|
| Gate 0 (knob distinguishability) | can the basis tell the mouth's hot knobs apart? | **PASS under W_deriv**: happy\|angry 0.95→**0.76**, happy\|surprised 0.95→**0.61**, angry\|surprised 0.91→**0.64** — the user's derivative weighting is what does it |
| Gate 1 (cross-text alignment) | do knob effects align with RAVDESS directions? | FAIL — max cos +0.29 ≈ noise. Contour equations are text-sensitive; mouth spoke S1, actors spoke "Kids..." |
| Gate 1-bis (matched text; 12 unscored calibration clips, mouth speaks the RAVDESS sentence) | same, text held constant | transfer fixed (alignments 0.6–0.8) **but a common-expressivity axis dominates**: every emotion-vs-neutral direction shares "more contour movement than neutral," and the melancholic knob adds the most of it → melancholic tops every target → FAIL |
| Exploratory (contrast directions subtract the common axis; post-hoc, labeled as such) | hot-vs-hot geometry, matched text, W_deriv | **anger: `angry` knob decisively sane (+0.66, #2 at +0.34)** — first scaffold basis ever to solve anger. joy: `happy` weakly sane (+0.13). sadness/surprise: fail (no knob out-fluctuates rivals). **2/4** |

**Honest verdict:**
1. **The user's two ideas are now both partially vindicated with data:** the
   Fourier-equation fingerprints are real and validated; the derivative
   weighting is the single ingredient that made the mouth's hot knobs
   distinguishable (Gate 0's first-ever pass).
2. The best configuration found (Fourier + derivation + matched text +
   contrasts) reaches 2/4 sane — anger decisively, joy marginally — but was
   assembled adaptively after the pre-registered gates failed, so it is a
   **hypothesis for confirmation, not a confirmed pass**. The 3/4 bar for
   spending steering clips is not met.
3. Remaining wall, precisely located: the mouth's `sad` and `surprised` knobs
   do not produce fluctuation signatures that beat their rival emotions'
   directions — jointly a mouth expressiveness limit and a single-clip noise
   floor (deterministic synthesis → no replicates at fixed text).
4. Paths from here (user's call): (a) a narrow pre-registered joy-only spike —
   the one target where this compass could add value the rules never achieved
   (judge-named joy = the only win condition, ~6 clips); (b) the
   embedding-space scaffold (P4.8 verdict #4) — still the deep fix; (c) park
   and return to P4.6.

Artifacts: `fourier_gate.py`, `gate1_fourier.py`, `gate1b_matched.py`,
`out/fourier_gate.json`, 12 matched-text calibration clips (unscored,
disclosed). Laws unchanged: judge frozen, never consulted here · calibration
clips unscored per P4.7 precedent · adaptive analyses labeled exploratory ·
claims sized to evidence.

---

## P4.9c — THE EXPANDED VIEW: fingerprints across three human corpora (2026-08-10)

**User's directive:** don't judge the equations on one corpus — expand to
CREMA-D and MSP audio, then work from the results. Extracted Fourier
fingerprints for **10,955 human clips** (RAVDESS 1,056 · CREMA-D 4,899 ·
MSP sample 5,000 seeded, 1 failure in 10,956): one canonical z-space,
per-corpus neutral anchors, W_deriv throughout. MSP audio used as statistics
only, license-safe.

**Finding 1 — THE FLUCTUATION FINGERPRINTS ARE NOT UNIVERSAL.** Same-family
direction similarity across corpora: acted↔acted moderate (anger rav~crema
+0.49, joy +0.44), but acted↔natural collapses (anger rav~msp +0.25,
crema~msp +0.06, joy ~0) and **sadness is ANTI-correlated: rav~msp −0.46** —
stage sadness and real conversational sadness are *opposite gestures* in
derivative space. Cross-corpus transfer classification confirms: 25.7–44.3%
vs 25/33% chance ≈ noise. **Performed emotion and felt emotion differ at the
contour-dynamics level.** (This independently corroborates the P4.3/P4.7
sadness triangulation: humans can hear performed sadness, but its acoustics
are not natural sadness.)

**Finding 2 — the mouth is a THIRD style.** Every one of our judge-confirmed
anger and surprise HITs scores best as "joy" against the pooled human
directions (anger HITs score *negative* on pooled anger). The frozen judge
recognizes the mouth's emotions; pooled human fluctuation-equations do not.
Mouth ≠ acted ≠ natural — three dialects of emotional prosody.

**Finding 3 — the pre-registered pooled gate: FAIL, 2/4** (declared before the
run; one shot, honored). anger → `angry` decisively sane (+0.61, replicating
matched-text +0.66 — anger is now solved in equation space under two
independent foundations). joy → `happy` sane (+0.23). sadness ✗, surprise ✗
(the `surprised` knob scores −0.15 on its own target).

**Where the whole scaffold program now stands (P4.8→P4.9c, exhaustive):**
static features, dynamics statistics, Fourier equations, derivative weighting,
contrast directions, matched text, and a three-corpus pooled foundation have
each been gated. Stable conclusion across all of it: **anger and joy are
solvable in interpretable equation space (2/4, twice, independent
foundations); sadness and surprise are not, under any hand-crafted basis or
foundation** — their knob signatures don't resemble any human style, and human
styles don't even agree with each other. The discriminative information the
judge provably uses for those families is embedding-level.

**Supported next steps (from results, user's call):** (a) joy-only
pre-registered spike — the compass's valid domain (anger solved, joy open);
win = frozen judge says "joy", nothing else counts. (b) embedding-space
scaffold for sadness/surprise. (c) park + P4.6.

Cost of the expanded view: $0, ~40 min compute, 10,955 fingerprints cached.
Laws unchanged.

---

## P4.9d — THE EPICYCLE GATE: 3-axis Fourier with cross-axis circles (2026-08-10)

**User's idea:** the 1-axis series was a shadow — represent the gesture as a
curve in 3-D acoustic space (F0 × Energy × HNR-over-time, the third axis chosen
to attack sadness's missing voice-quality dynamics) with the epicycle form:
complex harmonics + cross-axis phase terms (lead/lag rotation invariants —
time-shift invariant, genuinely new physics the DCT concatenation cannot see).

**Pre-registered (before run):** matched-text contrast sanity ≥3/4 incl. anger
AND at least one of sadness/surprise must flip sane ("open the blockages").
Fear/disgust declared out of scope (judge vocabulary).

**Results:**
1. **Foundation-side gem: surprise recall jumps to 69%** (held-out actors,
   W_deriv) — from 14.5% (static v1) and weak in 1-axis Fourier. The human
   surprise signature IS a cross-axis timing event (pitch spike vs energy lag),
   exactly as the epicycle hypothesis predicted. This is the best surprise
   detector built anywhere in this project — a candidate feature set for the
   EAR (Project 1) as future work.
2. Sadness recall collapsed to 8% — HNR *dynamics* did not reveal acted
   sadness; 4-way overall 35.9% (below the 1-axis 38–40%).
3. **Gate: FAIL (1/4 — anger only, both weightings).** Joy lost sanity (calm
   tops), sadness/surprise still insane on the mouth side.

**The decisive relocation of the wall:** the epicycle basis *can* see human
surprise (69%) — yet the mouth's `surprised` knob still scores negative
against human surprise contrasts. The blockage is therefore no longer "we
can't measure the difference" — **the mouth does not perform the human
gesture.** Third-dialect finding (P4.9c) confirmed from a second direction:
the judge names the mouth's surprise correctly, human-timing templates do not.

**Representation program closed.** Four bases gated ($0 total): static,
dynamics statistics, 1-axis Fourier + derivative, 3-axis epicycles. Convergent
conclusion: anger is solved in equation space; joy is marginal; sadness and
surprise are mouth-gesture-limited (steering) and ear-limited (naming), not
measurement-limited. Remaining honest paths unchanged: joy spike ·
embedding-space scaffold · sentence-aware sadness (S3) · ear v2 for
fear/disgust — plus the new spin-off: epicycle timing features as an ear
upgrade candidate for surprise.

Laws unchanged. Every gate pre-registered, every failure kept.

---

## P4.10 — THE EAR GATE: Gabor micro-gestures vs the blind families (2026-08-10)

**Scope shift, user-approved:** aim the uncertainty-principle math (Gabor
acoustic quanta: tremor band 3–10 Hz, transient band 10–40 Hz, multi-scale
responses, all in ABSOLUTE time) at the ear's blind families. Benchmark:
16,548 clips (RAVDESS full · CREMA-D full · MSP sample with ALL fear/disgust
kept), held-out speakers, identical nearest-centroid protocol for four bases
(DCT-22 / epicycle-90 / wavelet-48 / epicycle+wavelet-138). Pre-registered:
PASS = fear recall +10pp over epicycle, surprise within −5pp.

**Verdict: FAIL — fear gain −3.8pp.** The folk hypothesis "fear = tremor" is
refuted at scale: 3–10 Hz micro-fluctuation energy does not distinguish fear in
any of three corpora (wavelet-alone fear 14.1% ≈ DCT 14.2%). Fear's expression
is acoustically heterogeneous; it needs learned features, not hand-crafted
bands. Important negative knowledge, cheaply bought.

**Two secondary findings (exploratory — not the pre-registered target):**
1. **Surprise detection: best-ever, again.** epi+wav 40.7% mean vs epicycle
   29.0% (+11.7pp); on natural MSP speech 30% vs 16% — nearly doubled. Third
   consecutive confirmation that surprise is a time-localized transient event;
   micro-transient features are its natural detector.
2. **Natural sadness: unexpected +7.7pp** (epi+wav 32.3% vs epicycle 24.6%;
   MSP 37% vs 23%). Static HNR level never marked sadness — but voice-quality
   MICRO-INSTABILITY does. The "breathiness dynamics" hypothesis was right at
   millisecond scale, wrong at contour scale.

**Honest placement:** these classical detectors are far below the production
ear's embedding-based performance in absolute terms — their value is
interpretability and complementarity (surprise/sadness micro-channels as
auxiliary features for a future ear-v2), not replacement. Fear and disgust
remain learned-feature problems. The judge stays frozen; nothing here touches
it.

Classical-feature program (mouth P4.8–P4.9d + ear P4.10) now fully gated and
closed: six pre-registered gates, zero steering clips wasted, every hypothesis
answered with data. Laws unchanged.

---

## P4.11 — THE JOY SPIKE: six clips at the last open door on S1 (2026-08-10)

**Pre-registered:** win = frozen judge says "joy" on S1, any confidence;
budget 6 clips, round-2 rules declared before synthesis. Probes built from the
P4.7 gap (happy 0.3→neutral, 0.4→fear, joy never between).

**Result: NO JOY. 0/6.** The verdict cycle on S1 is now fully mapped:
cool it → neutral@100%; warm it → fear@60-100%; add transient or cool the
happy-prompt variant → surprise@80%. Joy is not on the path at any intensity.

**The decisive diagnosis is in the valence column: V never went positive.**
All six clips: V ∈ [−0.18, −0.08] against a joy centroid of +0.30. Even the
happy speaker prompt (timbre warmth, the untested channel) only lifted V from
−0.18 to −0.08 — the best S1 valence ever recorded for this mouth, still
negative. **The mouth's happy knob adds arousal, not positivity; on a
semantically neutral sentence it cannot produce positive-valence acoustics.**
S1-joy is closed: mouth-limited (valence ceiling), consistent with the P4.7
moderation pattern and the fair-rematch finding that only ElevenLabs ever
produced a strong synthetic joy.

**What remains for joy:** sentence dependence — untested. Sadness was locked
on S1 and unlocked on S3 (P4.4b); joy may behave the same. P4.6's transfer map
answers this as part of its design. Ledger: 174 → 180 rows.

Laws unchanged: pre-registered, misses kept, every clip a row.

---

## P4.12 — SURVEY: how open-source engines solve emotion (and two discoveries in our own mouth) (2026-08-10)

**Local discovery 1 — the hidden bias.** IndexTTS-2's `normalize_emo_vec()`
silently rescales every vector: happy ×0.9375, angry ×0.875, **surprised
×0.6875, calm ×0.5625**, then clamps the TOTAL emotion sum to **0.8** (not the
1.5 our code enforced). Every ledger `emo_vector` is the *requested* vector;
the *effective* vector was smaller (surprised=1.4 → 0.8 actual; angry=0.7 →
0.61). Conclusions unaffected (all systems shared the compression) but the
intensity axis was ~45% narrower than documented for surprise/calm. Recorded
as a correction note; future control strings should log effective vectors.

**Local discovery 2 — the unused channel.** `infer()` accepts
`emo_audio_prompt` + `emo_alpha`: emotion cloned from a REFERENCE AUDIO clip
(mutually exclusive with emo_vector; speaker prompt supplies timbre
separately). 180 ledger rows and we never used it once. A real human joyful
clip as the emotion target bypasses the 8-slider bottleneck entirely. There is
also `use_emo_text` (Qwen maps a description → the same 8-dim vector — no new
physics, confirms the vector is that path's bottleneck).

**The survey — four mechanisms across the open-source field:**
1. **Reference-audio style transfer** (GST/reference encoders; IndexTTS-2's
   emo_audio_prompt): emotion = embedding from a real emotional clip.
2. **Semantic conditioning** (CosyVoice-instruct "speak happily";
   EmotiVoice/PromptTTS style prompts; StyleTTS2 style diffusion conditioned
   on emotional text): the TEXT carries the emotion. Note: this is exactly
   what our delivery-only law holds constant — other engines' "joy" is partly
   semantic leakage our benchmark forbids. In LoRa production, semantics vary,
   so joy may be far easier in real use than on S1.
3. **Paralinguistic event tokens** (Orpheus <laugh>/<chuckle>; CosyVoice2
   [laughter][breath]; ElevenLabs v3 tags): joy is delivered by DISCRETE VOCAL
   EVENTS — laughter, smiling voice — not by continuous prosody knobs. This
   mechanistically explains our valence ceiling: real joy's markers (laughter
   bursts, smile-raised formants) are events the 8 sliders cannot emit. EL's
   P4.4b joy win used exactly these tags.
4. **Embedding arithmetic on frozen models** (EmoKnob, EMNLP 2024): emotion
   direction vectors in speaker-embedding space from neutral↔emotion pairs,
   applied with a strength knob — **the published version of our proposed
   embedding-space scaffold (P4.8 #4).** The field converged on the same idea.

**Actionable next (pre-registered, awaiting go): P4.12b joy-by-reference** —
emo_audio_prompt = real joyful clips (RAVDESS/MELD), neutral speaker prompt,
emo_alpha sweep, S1 text, ~6 clips, win = judge says joy. The last untested
channel of our own mouth, and the field's standard mechanism for exactly this.

---

## P4.12b — JOY BY REFERENCE: the last channel, closed (2026-08-10)

**Pre-registered probe of the newly-discovered `emo_audio_prompt` channel**
(emotion cloned from reference audio — Voxtral's mechanism, unused in 180
prior rows). 6 clips, win = judge says joy on S1. **Result: 0/6. NO JOY.**

**But the channel is measurably the best joy instrument the mouth has:**
- `rav_happy_male @ alpha=1.0`: **V = +0.02 — the first non-negative valence
  ever recorded for this mouth** (all-time prior best: −0.08), and
  **d = 0.318, the best joy distance ever** (prior best 0.402).
  Judge verdict: neutral@100%.
- Female acted joy ref: V=+0.02 as well, neutral@60%.
- Natural MELD joy refs backfired: conversational TV audio carried chaos, not
  warmth — judged anger@100% / fear@40% at arousal 0.73–0.87. Reference
  quality matters more than reference authenticity.

**Verdict: S1-joy is closed across ALL FOUR channels** — emo_vector (P4.7,
P4.11), speaker prompt (P4.11), text-implicit, and now reference audio. The
mouth gets warmer by reference than by any vector, but cannot cross the joy
threshold on a semantically neutral sentence. The failure pattern now exactly
mirrors sadness-on-S1: nearest-ever acoustics, neutral@100% verdict. Both
locks point at the same suspect — the SENTENCE. S3-class sentences unlocked
sadness (P4.4b); the sentence hypothesis for joy is the single remaining path,
and P4.6 (transfer map) tests it by design.

Ledger: 180 → 186 rows. Laws unchanged: pre-registered, misses kept.

---

## P4.13 — THE ANGER↔JOY AXIS: the user's "little difference" hunch, measured (2026-08-10)

**Hunch:** joy and anger share high pitch and fast delivery; the separation
must be one small specific difference. **Measured on 59,656 real clips
(MSP joy 37k + anger 22.6k; RAVDESS 192+192): the hunch is structurally
CORRECT.** The separator is a narrow tension/relaxation delta on top of shared
arousal — 12/15 top features agree in sign across natural and acted speech
(axis cosine +0.54):
- Joy sits LOWER than anger on F1/F2/F3 formants (relaxed vs pressed vocal
  tract; F2 d=−0.64/−0.74 — the strongest separator).
- Joy's pitch is slightly LOWER than anger's (d≈−0.5) — correcting the "pitch
  is the same" premise: same register, small offset.
- Joy is warmer in spectrum (mfcc1V/mfcc3 up) with a SMALLER loudness range
  (less spiking). In one line: **joy = anger minus tension.**

**The mouth's knobs on this axis (the whole joy predicament in one table):**
angry −4.61 · sad −4.88 · surprised −3.99 · **happy −3.39** · afraid −2.40 ·
disgusted +0.64 · melancholic +0.62 · calm +0.39 (per unit slider, + = joy-ward).
**Every expressive knob — including `happy` — moves the voice ANGER-ward on
the axis that separates joy from anger.** Pressing "happy" adds tension. The
only joy-ward knobs are weak (+0.4..0.6) and capped: max reachable ≈ +0.5
against a needed ≈ +0.9 (EL's judged joy) from −0.07 (our warmest clip).
The little difference is real, but it lies in the one direction the mouth's
control space cannot travel. Mouth-limitation now proven *in the exact plane
of the user's hunch*.

**Forensics of the only judge-approved synthetic joy (EL d=0.144):** vs our
warmest clip it is LOWER-pitched, softer-onset (attack +1.4z), more flowing
(mean pause −5.6z), with a contained loudness range — plus its [laughs]
events. The recipe of synthetic joy the judge accepts: warm, low-tension,
flowing, event-punctuated — not "excited." (Caveat kept honest: EL's other
joy HIT sits at −0.97 on this axis — the e2v judge does not strictly follow
classical-axis geometry; embeddings see more.)

**Actionable residue:** joy-ward = SUBTRACT tension. Within the vector space
the only legal moves are microdoses of calm/melancholic/disgusted with no hot
knob — a region P4.6 can sweep across sentences at zero extra design cost.

$0. Ledger untouched (analysis only). Laws unchanged.

---

## P4.6 — THE TRANSFER MAP: 19 sentences × 11 configs, 209 clips (2026-08-10)

The declared recommencement point, finally run — and it broke two locks in one
night. Every clip judged by the frozen bridge, every clip a ledger row
(ledger 186 → 395). Per-sentence zero-vector baseline included, so every
verdict is ATTRIBUTABLE: config-caused vs text-caused.

### JOY: UNLOCKED — six verdicts, three independent routes
| route | sentences | note |
|---|---|---|
| joyref_hm10 (reference channel) | s05 warm, s07 warm, s19 somber | replicated 3× |
| joy_detension (calm.3+mel.2 — the P4.13 subtract-tension region) | s11 exclaim, s19 | the user's axis hunch, vindicated in synthesis |
| joy_h035 (the gap vector) | s07 warm | even the plain vector works on the right sentence |
Attribution clean: on every joy-verdict sentence the zero-vector baseline read
neutral — the configs caused the verdicts, not the text alone. S1-era
conclusion refined: joy was never mouth-impossible, it was
SENTENCE-CONDITIONAL, exactly like sadness. Both keys (channel × sentence)
were required.

### SADNESS: UNLOCKED WIDE — ten verdicts
sad_m10c03 (mel 1.0 + calm 0.3) hits on NINE sentences (s02 — replicating the
original P4.4b S3 unlock — plus s06,s07,s08,s09,s13,s14,s15,s16); sad_s04
adds s08. The lock is now precisely bounded: only dead-flat declaratives
(s01/s03/s04-type) stay locked; somber, question, short, even warm sentences
open it.

### ANGER: UNIVERSAL — anger_a08c02 went 19/19
Perfect across every sentence category. anger_a07: 17/19 (2 drift to
surprise). The fair-rematch "sentence instability" criticism is answered:
a stable config exists and is now proven.

### SURPRISE: NEAR-UNIVERSAL — 16-17/19
Drifts to anger only at the short/long extremes (s17 "It's over.", s18 long).

### The confound, quantified (and why the map survives it)
The zero-vector baseline itself reads non-neutral on 7/19 sentences (exclaim
category 0/3 neutral; s12 zero-vector even reads joy@judge) — the mouth
performs sentence semantics on its own, and the judge follows. This is why
every claim above is stated against the per-sentence baseline; the
text-prosody confound is measured, not ignored.

### Standing after this run
Mouth scoreboard: anger universal · surprise near-universal · sadness
sentence-conditional (10 verdicts) · joy sentence-conditional (6 verdicts,
3 routes). Fear/disgust remain ear-limited (unchanged). Phase-5 Gate 3
(data variety): ledger now 395 rows across 19 sentence types + 5 systems —
GREEN, pending human ratification of the new verdicts (blind pack next: the
6 joy clips + sample of the 10 sadness clips, user's ears, pre-registered
protocol as in GATE 2).

Cost: ~75 min MPS, $0. Laws unchanged: judge frozen · misses kept (all 209
rows) · per-sentence baselines for attribution · claims sized to evidence.

---

## P4.6-HG — HUMAN GATE on the map's new verdicts: VOID by controls, with signal (2026-08-10)

13-clip sealed blind pack (6 joy verdicts, 3 sadness verdicts, 4 controls),
scored against pre-registered rules. Listener's own caveat, recorded: clips
2-3s, thin lines between hot emotions, context missing.

**Controls: 2/4 → the session is formally VOID** (rule declared before
listening: scrambled controls void the session — a protection, applied).
The two failures are themselves a finding: the listener heard the ANGER
control as surprise and the SURPRISE control as angry — the exact pair our
instruments confuse (knob collinearity 0.85-0.91, P4.9). Human ears, classical
features, and acted corpora all blur anger↔surprise on short clips; only the
frozen judge separates them consistently.

**Void-but-informative per-route signals (not claims):**
- **Sadness 3/3 heard sad-family** — even a skeptical listener corroborates
  the sadness unlock (sad, sad, melancholic).
- **Reference-channel joy: 2/3 heard "happy"** (s05✓ s19✓, s07 heard
  surprise) — the strongest human agreement of any joy route.
- **Subtract-tension joy: 0/2 — both heard "sad."** Judge says joy, human
  says sad: the detension route is flagged as possibly judge-idiosyncratic
  (embedding-neighbor artifact). Demoted to "judge-only" pending re-test.
- Overall joy 3/6 = below the pre-registered majority bar even ignoring the
  void.

**Status line updated:** sadness unlock = judge-confirmed + informally
human-corroborated. Joy unlock = judge-confirmed on 3 routes; human
ratification INCOMPLETE (void session; detension route disputed). Writeup
language stays sized accordingly. Re-test path: longer clips on warm long
sentences (listener's stated obstacle), fresh controls, fresh shuffle.

**Listener's product insight, adopted into the record:** context/semantics is
now a MEASURED control channel (the map quantified it: zero-vector baseline
non-neutral on 7/19 sentences). For any conversational product built on this
engine, that is a lever, not a confound — the mouth authors its own
sentences, so sentence-conditional emotion is usable by design: choose the
words and the delivery together.

---

## P4.14 — THE TRAJECTORY: the mouth writes a journey, the ear reads it back (2026-08-10)

**User's directive:** lengthier speech, fixed-interval analysis with the
interval chosen empirically, and a graph of the emotional fluctuation.

**Setup:** four-act journey (neutral → joy → sadness → anger, 27 s total),
each act three sentences using its map-proven config (joy = reference
channel; sadness = mel1.0+calm0.3; anger = the 19/19 config). Read back by
WavLM at four window sizes, 50% overlap; judge (frozen e2v) named every
window at the chosen size. Graph: `docs/p414_trajectory_graph.png`.

**The window question, answered by data:**
| window | corr V | corr A | mean |
|---|---|---|---|
| 1.5 s | +0.80 | +0.85 | +0.83 |
| 2.0 s | +0.86 | +0.94 | +0.90 |
| **3.0 s** | **+0.93** | **+0.90** | **+0.92** |
| 4.0 s | +0.87 | +0.93 | +0.90 |
**≈3 s is the appropriate interval** (2–4 s is a broad plateau; below 2 s the
estimate gets noisy). Independently consistent with the 3 s windows chosen in
the long_1.wav experiment months earlier.

**The result: the commanded emotional staircase reads back at r ≈ 0.92.**
Valence: flat ≈ −0.1 through neutral, rises to +0.45 in the joy act
(EXCEEDING the commanded +0.30 — the first time measured valence beat the joy
centroid), falls to −0.3 in sadness, plunges to −0.7 in anger. Arousal dips
to 0.25 in sadness and spikes to 0.88 in anger. The judge strip narrates the
same story: neutral ×4 → joy ×4 (one boundary window reads surprise) →
sadness ×2 → anger ×2 at the finale. **Sustained, judge-named JOY in flowing
speech** — stronger evidence than any single clip: WavLM's own family read
agrees (joy ×4).

**Honest notes:** boundary windows straddle two acts and read mixed
(surprise/contempt at transitions — expected artifact of windowing, not
drift); two mid-sadness windows relax to neutral (the act's quiet middle);
the journey switches emotion BETWEEN utterances — within-utterance gliding
remains the mouth's frontier (Phase 5).

**Significance for the record:** this is the founding notebook sketch
completed — the repo began with points on the PAD plane; this experiment
commands a PATH through that plane and watches the ear trace it back at
r≈0.92. Ear and mouth demonstrably speak the same geometric language over
time. $0, ~10 min compute.

---

## P5.0 — FEASIBILITY SPIKE: Phase 5 is local (2026-08-11)

Full verdict in `P5_FEASIBILITY.md`. Headline findings from reading the
vendor code line by line:
1. IndexTTS-2 ships NO training code — a LoRA would mean building a training
   stack against undocumented internals.
2. **It doesn't matter: the entire emotion pathway collapses into ONE
   embedding vector (`emovec`)**, assembled by plain arithmetic from tensor
   banks stored in the checkpoint (`emo_matrix`). The 8 sliders are just
   capped, biased weights over those banks.
3. **Phase 5A = optimize `emovec` directly** against the frozen judge:
   10-15-dim search (uncapped, sign-free bank coefficients + reference-derived
   vectors — "subtract tension" becomes a literal operation), CMA-ES,
   ~150 evals/emotion, overnight, local, $0. No GPU. No weights touched.
   Judge frozen. Gate-4 mitigation pre-registered in the doc (held-out
   sentences, dual instruments, hull constraint, eval cap, human final gate).
4. Phase 5B fallback only if 5A saturates: Maya1 LoRA on a cheap A40/4090
   ($5-15). The RTX PRO 6000 is never required.

Next: emovec_worker + smoke, live-voice demo, then the 5A joy run.

---

## CORRECTION to P4.12 discovery 1 (2026-08-11)

The "hidden bias" (`normalize_emo_vec`: per-knob bias + Σ≤0.8 clamp) is
called **only from webui.py — NOT from the inference path our synth_worker
uses.** Our ledger's `emo_vector` control strings were therefore delivered to
the model EXACTLY as recorded — no correction factor applies to any of our
395+ rows. The P4.12 note stands only as a warning for webui users. Found
while building the P5A worker: the vendor infer path takes raw signed
coefficients natively (emovec = Σw·bank + (1−Σw)·speaker_emovec), which means
the Phase 5A search space required NO code changes — only removing our own
harness clamps. Error mine; corrected the same day it mattered.

---

## P5A SMOKE — first signed emovec clips: a record, a confirmation, a warning (2026-08-11)

Four probes on s07, raw signed coefficients through the untouched vendor path.

**The record: `happy +0.60, angry −0.30` → d = 0.096 to the joy centroid —
the closest ANY clip has come to ANY centroid in 399 ledger rows** (previous
all-time best: EL's 0.144). V = +0.34, beyond the +0.30 centroid. The
subtract-tension arithmetic works exactly as P4.13 predicted: removing the
anger bank is worth more valence than adding the happy bank. The blend probe
(happy .45, calm .35, angry −.15) confirms: d = 0.115, V = +0.30. Extrapolation
past 1.0 (happy 1.10) blows up arousal (0.93 → surprise@60%) — the hull
constraint is justified.

**The warning (honest, important): all three warm probes judged NEUTRAL —
including the parity clip (happy 0.35 on s07) that was judged JOY in P4.6.**
Same config, same sentence, different verdict. Cause identified: GPT
generation samples (do_sample=True, temperature 0.8) — `use_random=False`
fixes only the emovec bank row, NOT the autoregressive sampling. Synthesis is
stochastic across runs; near the judge's razor-thin joy boundary, sampling
noise flips verdicts. Two consequences, recorded:
1. All single-clip verdicts in this log carry sampling variance; the P4.6
   map's verdicts are real but each is one draw. Multi-draw confirmation is
   now required for any headline claim.
2. The 5A objective must be noise-aware: ≥2 draws per candidate, optimize
   judge-confidence + V/A/D distance jointly, hull-bounded coefficients.

Phase 5A thesis status: **acoustically confirmed at record level; judge
confirmation pending the optimization run** (which now knows to fight the
noise). Ledger 399 rows.

---

## P5A JOY — first native-space optimization run: held-out 4/6 (2026-08-11)

The first true Phase 5 run: seeded (μ=3, λ=8) evolution strategy over signed,
hull-bounded coefficients on the 8 emotion banks, 10 generations, 2 draws per
candidate (s07 + s05), score = 2·conf·(judge==joy) + max(0, 0.5−dVAD). 160
optimization clips + 8 held-out, every one a ledger row (indextts2-p5a-joy).
Judge untouched. One transient bridge failure at gen 5 (funasr import hiccup;
judge verified healthy, retry-with-cooldown added, run resumed from cache —
zero clips lost).

**Best vector (score 1.224, gen 10):**
`[happy +0.409, angry −0.266, sad +0.022, afraid +0.131, disg +0.058,
mel −0.190, surp −0.230, calm +0.003]`
The optimizer generalized subtract-tension into subtract-EVERYTHING-dark:
negative angry AND melancholic AND surprised. Happy stays moderate (~0.41) —
the search confirmed the smoke lesson that pushing happy up buys arousal, not
valence; valence is bought by removing the other banks' contamination.

**Held-out one-shot (pre-registered, zero iteration, sentences never seen by
the optimizer): 4/6 joy verdicts.**

| clip | draw 0 | draw 1 |
|------|--------|--------|
| h01 "lovely afternoon" | **joy@80%, V=+0.96** | **joy@80%, V=+0.94** |
| h02 "good news about the baby" | **joy@100%, V=+0.09** | **joy@60%, V=+0.13** |
| s06 "garden full of flowers" | neutral@100%, V=+0.35 | neutral@80%, V=+0.32 |
| s03 control "report on the desk" | neutral@100%, V=−0.06 | neutral@100%, V=−0.06 |

What this means, honestly:
1. **Repeatability under sampling noise — first time ever.** h01 and h02 hit
   joy on BOTH independent draws. No prior joy verdict in this log survived a
   second draw (the P5A-smoke parity flip). This is the difference between
   "joy happened once" and "joy is steerable."
2. **V=+0.96 is the highest valence the mouth has ever produced** (previous
   record +0.34). It overshoots the MSP joy centroid (+0.30) — d is large
   (0.67) for h01 precisely because valence is TOO positive. Dual-instrument
   note: judge and valence agree on direction; the distance metric penalizes
   the overshoot. Not hidden.
3. **The control behaved.** Flat administrative content stayed neutral@100%
   at V≈−0.06 both draws. The vector does not paint joy onto everything — it
   amplifies warmth where the text affords it. Sentence-conditionality (P4.6)
   persists but is now an asset: context-appropriate joy, not a joy sticker.
4. **s06 split the instruments**: V=+0.32–0.35 (essentially ON the joy
   centroid, d=0.113) yet judged neutral. The garden sentence is descriptive,
   not relational — consistent with the P4.6 finding that joy needs a warm
   PERSONAL frame. Both relational held-out sentences hit 4/4.

Per-gen joy counts stayed noisy (6,3,6,2,0,4,3,1,3,3 of 16) — the judge
boundary is razor-thin on TRAIN sentences while held-out relational sentences
hit reliably. Sentence choice matters more than another 0.05 on any
coefficient. Ledger 567 rows. Human long-clip gate still owed before any
headline claim (Gate-4 #5).

**Next per sequence:** P5A sadness run (same protocol, mel-route seed), then
exploratory fear; live-voice demo independent.

---

## P5A SADNESS — optimization + hull diagnostic: acoustics generalize, the verdict doesn't (2026-08-12)

Same pre-registered protocol as P5A-JOY (10 gens × 8 cands × 2 draws, s08+s09
train, hull identical, judge frozen, system indextts2-p5a-sad, 168 rows).

**Training saturated immediately.** Gen 1: 14/16 sadness verdicts; best score
2.383 — nearly double joy's best (1.224). The winner IS the seeded mel route
at the hull edge: `[mel +0.784, calm +0.206, surp −0.095, ...]`. No later
generation beat it; the evolution explored subtract-brightness variants
(negative happy/disgusted/surprised, sad up to +0.35) — all close, none
better. On somber carriers the judge boundary is wide open. Per-gen counts:
14,9,9,13,15,10,11,13,13,14 of 16.

**Held-out one-shot: 1/6 by judge verdict — the mirror image of joy.** But
the second instrument disagrees, and both are reported (Gate-4 rule 2):

| clip | judge (d0 / d1) | V (d0/d1) | dVAD (d0/d1) |
|------|-----------------|-----------|--------------|
| s10 photograph | neutral / neutral | −0.10 / −0.19 | 0.200 / 0.130 |
| h03 old clock | **sadness@40%** / neutral@100% | −0.24 / −0.25 | 0.122 / **0.089** |
| h04 nothing the same | neutral / neutral | −0.27 / −0.31 | 0.119 / 0.105 |
| s07 warm CONTROL | neutral / neutral | +0.16 / +0.14 | — |

Every somber held-out clip lands essentially ON the MSP sadness centroid
(V=−0.28): h03-d1 at d=0.089 would be the 2nd-best distance in ledger
history — judged neutral@100%. **Acoustically the vector generalizes to
unseen sentences; categorically it doesn't cross the judge's boundary.** The
control stayed clean both draws — no smearing, unlike the raw P4.6 route.

**Hull diagnostic (indextts2-p5a-sad-diag, +8 rows):** the UNCLAMPED P4.6
recipe (mel 1.0 + calm 0.3, Σ=1.3, outside the hull) on the same held-out
set: 2/6 somber hits (h03-d1@40%, h04-d0@60%) — within sampling noise of the
optimized 1/6 — **but it smeared the control: s07-d1 judged sadness@100%**
("we are going to see them again this weekend" delivered sad). Verdict:
extra intensity does not reliably buy judge crossings on new sentences; it
buys smearing. The in-hull optimized vector is the better product config:
on-centroid V/A/D, clean control.

**What sadness now lacks (honest):** not valence, not arousal, not dominance —
all three sit on the centroid. emotion2vec withholds "sadness" on unseen
sentences unless some draw-level prosodic event occurs (h03 flipped between
draws at the same distance). Whatever that cue is (voice breaks? tempo?
spectral tilt?), it is NOT captured by the V/A/D position — a judge-side
sensitivity, not a mouth-side failure. Options logged: (a) accept the
dimensional signal as the product output for sadness (LoRa-style V/A/D is
what downstream consumes anyway), (b) study which draws flip the verdict and
what distinguishes them acoustically, (c) ear-v2.

Scoreboard after both 5A runs: joy held-out 4/6 (repeatable, control clean),
sadness held-out 1/6 by verdict but ~6/6 by centroid proximity (control
clean). Ledger 743 rows. Human long-clip gate still owed.

---

## P5A FEAR (exploratory) — the blind spot was ours: held-out 4/6, text-INDEPENDENT (2026-08-12)

Protocol identical to joy/sadness (pre-registered AND committed before
results — e72577a). Seed: the native afraid bank, never pushed in any prior
run. MSP fear centroid derived fresh from 794 Train labels: (−0.21, 0.51,
−0.01) — anxiety, not screams. System indextts2-p5a-fear, 168 rows.

**The "judge fear blind spot" dissolved on contact.** Gen 1: 13/16 fear
verdicts. The judge reads fear fine — in 743 prior ledger rows we had simply
never seeded the afraid bank. Per-gen: 13,11,10,11,11,8,10,13,11,10 of 16;
best climbed 1.976 → 2.343.

**Best vector (score 2.343, gen 9):**
`[happy +0.115, angry −0.286, sad +0.319, afraid +0.296, disg +0.069,
mel +0.165, surp +0.071, calm −0.278]`
Third run, third subtraction discovery — this time **subtract anger and
subtract calm**. Dimensionally exact: fear and anger share high arousal but
sit at opposite dominance poles (−0.01 vs +0.41); removing the anger bank
buys powerlessness, removing calm buys unsettledness. The final winner leans
sad+afraid blended, not afraid-pure — natural fear is anxious, not shrieking.

**Held-out one-shot: 4/6 — ties joy. And the control result reframes
everything:**

| clip | draw 0 | draw 1 |
|------|--------|--------|
| s15 "what happened at the station" | **fear@83%** | **fear@83%** |
| h05 "someone is following us" | joy@40% | **fear@80%** |
| h06 "everyone got out in time" | **fear@100%** | sadness@60% |
| s03 flat CONTROL "report on the desk" | **fear@100%** | **fear@83%** |

The control hit fear on BOTH draws. Where joy needed warm text (control
clean) and sadness stayed polite (control clean), the fear vector paints
fear onto flat administrative content. Reported without spin, both readings:
as a P4.6-style single-config-for-everything this is smearing; as STEERING
it is the strongest control yet — command fear and the mouth sounds afraid
regardless of what the words say (a frightened voice reading "the report is
on the desk" is exactly how dread works). 6/8 held-out clips total went fear.
Fear delivery is text-independent; joy is text-conditional; sadness sits
between (acoustics transfer, verdict needs a prosodic event). That
three-way split is now a documented property of the emotion space, not a
guess. h06's draw-1 sadness@60% is honest kinship, not noise.

**Scoreboard after all three 5A runs:** anger 19/19 (P4.6) · fear 4/6 + 2/2
control (text-independent) · joy 4/6 (repeatable, text-conditional) ·
sadness 1/6 verdict but ~6/6 on-centroid · surprise 16/19 (P4.6) · disgust
untouched (no disgusted-bank run yet — the last unexplored bank). Ledger 911
rows. Human long-clip gate still owed before headline claims.

---

## P5A DISGUST — last bank explored: strict 2/6, family 6/6, control clean (2026-08-12)

Final 5A run (pre-registered + committed before results, e42400d). Structural
discovery logged in-protocol: the frozen judge has NO disgust class
(FAMILIES in scripts/retrieval_namer.py) — zero disgust verdicts in 911 rows
was architecture, not acoustics. Judge untouched (the law); scoring moved to
the loop's other frozen instruments: 2·P_wavlm(disgust) + proximity to the
MSP disgust centroid (−0.34, 0.59, +0.24; derived from 1,325 Train labels).
System indextts2-p5a-disgust, 168 rows.

**Training was the hardest of the four:** top-1 counts 0,3,4,7,4,6,5,5,7,6
of 16; best score only 0.900 (joy 1.224, sadness 2.383, fear 2.343 on the
same 2.5-max scale). Disgust is REACHABLE but FAINT for this mouth — peak
P(disgust) ≈ 0.3–0.35, never near-certain.

**Best vector (score 0.900, gen 9):**
`[happy +0.237, angry −0.146, sad −0.442, afraid −0.048, disg +0.360,
mel +0.041, surp −0.258, calm +0.057]`
Fourth run, fourth subtraction signature — the strongest yet: **subtract sad
(−0.44) and subtract surprise (−0.26)**. Disgust = negative-valence
rejection that is energized (anti-sad), closed (anti-surprise), with a
sneer of brightness (happy +0.24). Each 5A emotion now has a documented
subtraction identity: joy −dark, fear −anger/−calm, disgust −sad/−surprise.

**Held-out one-shot: 2/6 strict — but read the whole table:**

| clip | draw 0 | draw 1 |
|------|--------|--------|
| d02 "not touching that" | **disgust** (pdis .26) | **disgust** (pdis .25) |
| d03 "how can you eat this" | contempt | contempt |
| s16 "stop it right now" | contempt | contempt |
| s03 flat CONTROL | neutral | neutral |

Every non-control clip landed in the rejection family — disgust twice
(REPEATABLE, both draws on d02), contempt four times. Contempt is disgust's
blend-sibling (disgust+anger) and sits one WavLM class over; strictly scored
it is a miss, honestly read it is near-field. Control clean both draws. And
the pre-registered prediction confirmed exactly: the 6-way judge read the
disgust clips as anger@38–40% — the nearest name it is allowed to say.

**Verdict:** disgust is UNLOCKED at the dimensional + WavLM level with the
weakest margin of the six — repeatable on one unseen sentence, family-correct
on all, invisible to the 6-way judge by construction. A categorical upgrade
requires judge-v2 (enroll dataset disgust clips in a SECOND namer, original
untouched) — user's decision, not taken.

**PHASE 5A COMPLETE. All 8 banks explored, all Ekman-6 have optimized
vectors + honest held-out numbers:** anger 19/19 (P4.6) · fear 4/6
text-independent · joy 4/6 text-conditional · sadness 1/6 verdict but
on-centroid · surprise 16/19 (P4.6) · disgust 2/6 strict, 6/6 family.
Ledger 1,079 rows. Human long-clip gate owed before headline claims.

---

## THREE-STEP CLOSE-OUT — live demo, human gate pack, judge-v2 (2026-08-12)

**1. LIVE DEMO — the loop closed on a real human voice.** own_voice/001 →
ear: joy@80% (V=+0.41) → mouth replied with the P5A joy vector in the
user's own cloned voice → frozen judge on the reply: joy@80%, V=+0.86.
Heard joy → replied joy → certified joy. First full ear→mouth→ear circle on
live human input. Exchange 2 (007, respond mode): heard fear@60% → neutral
zero-vector reply → drifted sadness@60% — consistent with the P4.6
zero-baseline instability; proven vectors hold, the zero vector wobbles.
System indextts2-live-demo, 4 rows.

**2. HUMAN BLIND GATE PACK built (awaiting the human).** 8 long clips, ONE
fixed neutral paragraph for all (train station text) — voice alone carries
the emotion, no text leakage. Six Ekman best vectors + zero-vector neutral +
duplicate joy (consistency control). Seed-5814 shuffle, key sealed to
out/blind_p5a_key.json with machine verdicts recorded unseen. Fixes the
VOID'd session's design gaps. System indextts2-p5a-blind, 8 rows.

**3. JUDGE-V2 — disgust certified, zero collateral.** Built a SECOND e2v
retrieval DB (models/adaptors_v2/): v1's 175 exemplars copied + 36 dataset
disgust exemplars (24 RAVDESS actors + 12 CREMA-D speakers — never mouth
clips). v1 untouched byte-for-byte; all 1,092 prior rows stay comparable.
Side-by-side re-judge of 14 clips:

- **d02 heldout (both draws): anger@38% → disgust@100%.** The two clips
  WavLM had called disgust are exactly the two v2 certifies — the two
  instruments now agree, through a third instrument neither trained on.
- **12/14 verdicts identical** — joy/fear/sadness sanity set unchanged,
  contempt-flavored clips (d03/s16) NOT over-claimed, controls untouched.

Disgust now has categorical certification: 2/6 held-out strict is
instrument-confirmed, not WavLM's opinion alone. The v2 namer is an
INSTRUMENT UPGRADE, not a judge retrain — the law held. Ledger 1,092 rows.

---

## P5A HUMAN BLIND GATE — RESULTS: human 4/8, machine 3/8, controls PASS (2026-08-12)

Human answered all 8 same-text long clips blind (key sealed until after).
Human's own caveat recorded: "emotions are quite blended, hard to
differentiate — don't blindly believe my judgment." The controls answer
that caveat: the human instrument PASSED its own calibration checks.

| clip | TRUE | human | machine (v1) | notes |
|------|------|-------|--------------|-------|
| 01 | neutral | neutral ✓ | neutral@41% ✓ | control clean both |
| 02 | fear | sadness ✗ | surprise@40% ✗ | both miss, differently |
| 03 | joy (dup B) | neutral ✗ | anger@60% ✗ | |
| 04 | surprise | surprise ✓ | surprise@100% ✓ | |
| 05 | joy (dup A) | neutral ✗ | anger@80% ✗ | |
| 06 | anger | anger ✓ | anger@100% ✓ | V=−0.39 on flat text |
| 07 | disgust | melancholic ✗ | neutral@100% ✗ | |
| 08 | sadness | melancholic ✓ | neutral@100% ✗ | HUMAN BEAT JUDGE |

**Controls (the reason to trust these answers):**
1. Consistency dup: the two identical-vector joy clips (03, 05) got the
   SAME human label — the instrument is repeatable.
2. Neutral control: correct.
3. Human hits align with machine confidence: every machine@100%-correct
   clip, the human also hit.

**Findings, in order of importance:**
1. **Anger + surprise are HUMAN-RATIFIED on neutral text** — both@100%
   machine, both correct human. Text-independent, certified by ears.
2. **Sadness: the human heard what the categorical judge cannot.** Clip 08
   = P5A sadness vector; human said melancholic (sadness family), judge
   said neutral@100%. Direct human confirmation of the P5A-sadness finding
   (acoustics transfer, verdict shy) — the melancholy IS audible.
3. **Joy's text-conditionality is now human-confirmed.** Both joy
   duplicates on flat train-station text → human: neutral, machine: anger.
   Exactly what the held-out map predicted (joy needs warm relational
   text; garden sentence went neutral). Not a new failure — the predicted
   character of joy, observed by a second species of instrument.
4. Fear/disgust don't carry on flat text to human ears (blend into
   sad/melancholic) — consistent with their faint/blended machine margins.
5. Human 4/8 vs machine 3/8 overall: the human is the better judge of this
   mouth. Both instruments confuse the SAME clips — the confusion is in
   the audio, not the judges.

**Gate verdict (honest): PARTIAL PASS.** Ratified: anger, surprise, neutral,
sadness-as-family. Not ratified on flat text: joy (by design of its
character), fear, disgust. No prior session to void; controls clean; this
is the first human gate that COUNTS. Headline claims may now say: "anger
and surprise human-ratified; sadness audible to humans where the machine
judge is silent."

---

## CONGRUENCE GATE — the carrier/resonance law CONFIRMED, accuracy doubles (2026-08-12)

Pre-registered (predictions committed at 846ee7e BEFORE synthesis). Six best
vectors × {flat train text, emotion-congruent text} × 2 draws, long-form.
24 clips, all ledgered (indextts2-congruence). Judges frozen; disgust
additionally scored by WavLM + judge-v2.

| emotion | type (predicted) | FLAT | CONGRUENT | verdict |
|---------|-----------------|------|-----------|---------|
| anger | carrier | 2/2 @100% | 2/2 @100% | carrier CONFIRMED |
| surprise | carrier | 2/2 | 1/2 (d1→anger@80%) | carrier, but text can INTERFERE |
| fear | carrier? (open cell) | 0/2 (both→surprise) | **2/2 fear@100%** | RECLASSIFIED resonance at long-form |
| joy | resonance | 0/2 (→anger) | **2/2 (joy@80%, joy@60%)** | FULL RESCUE as predicted |
| sadness | resonance | 0/2 (→neutral) | 1/2 (sadness@60%) | rescue ≥1/2 as predicted — first long-form sadness verdict ever |
| disgust | resonance | pdis .09/.11, V≈0 | pdis .23/.23, V=−0.37; v2 top-1 disgust d0; WavLM top-1 disgust d1 | gain on ALL instruments as predicted |

**LAW INTERACTION: +0.67** (resonance mean gain +1.00, carrier +0.33).
**Long-form accuracy DOUBLES under congruence: 4/12 flat → 8/12 congruent.**

Refinements the data forced (kept, not smoothed over):
1. **Fear moved cells.** Short-form (P5A held-out, 1-sentence) fear carried
   on flat text — even the flat control hit fear@100%. Long-form flat fear
   reads as surprise; congruent text locks fear@100% both draws. The
   carrier/resonance boundary is DURATION-DEPENDENT: sustained fear needs
   narrative fuel; a short burst carries alone.
2. **Text can hurt a carrier.** Surprise's congruent exclaim text pushed one
   draw to anger@80% — for carriers, vector alone is not just sufficient
   but sometimes SAFER. Congruence is for resonance emotions.
3. Disgust remains the faintest: real gains on every instrument, verdicts
   still marginal (v2@20%, wavlm one draw). Congruence helps; doesn't solve.

**THE CONGRUENCE LAW (codified, joins the binding laws):**
> Steer resonance emotions (joy, sadness, disgust — and fear beyond ~1
> sentence) with VECTOR + CONGRUENT TEXT together; never evaluate them on
> flat text and call the miss a steering failure. Steer carrier emotions
> (anger, surprise — and short-burst fear) by vector alone; congruent text
> is optional and may even interfere. Mechanism: the GPT acts the words —
> text is a steering input, not a semantic leak (judges are acoustic).

Answer to "do we gain accuracy making a law out of it": YES — measured,
long-form emotion delivery goes 33% → 67% when text and vector are chosen
together. Ledger 1,116 rows.

---

## REAL-WORLD TEST — the ear on 'Sorry, Wrong Number' (1943): dimensions track history, categories break (2026-08-12)

First fully-found audio: the original Suspense broadcast (May 25 1943,
public domain, archive.org) — 30 min, telephone-filtered voices, organ
music, AM-broadcast bandwidth. Chosen because its emotional arc is
documented theater history. 1,196 windows (3.0s / 50% overlap, the P4.14
protocol), both instruments, ~8 min of compute. Graph committed:
docs/real_world_trajectory.png.

**What the DIMENSIONAL ear (WavLM V/A/D) got right — at the right clock
positions, with no labels given:**
1. **The shock lands at 4–6 min: V dives +0.04 → −0.21 (raw floor −0.85)
   while A jumps 0.57 → 0.79.** That is where Mrs. Stevenson overhears the
   murder plot. The ear found the inciting incident on its own.
2. **Sustained tension through the middle acts:** A holds 0.74–0.81 from
   minute 4 to 20 — the escalating desperate-calls sequence — with dips to
   ~0.6 exactly at scene/announcer transitions (9.5, 14.5–16 min).
3. **The anger peaks are real scenes:** most-negative window of the whole
   broadcast (V=−0.91, A=1.00, judged anger) at 18.1 min; anger clusters
   6–8 and 16–18 min — her fury at operators/dismissal. Top-15 arousal
   windows cluster at 15.6–18.2, not randomly.
4. **A fear cluster appears at ~28.3–29 min — the climax window** (the
   murder/scream) — visible as the only purple block in the family band.

**What it got wrong / could not do (honest):**
1. **No clean terminal climax in the medians.** The last 5-min slice reads
   V=+0.30 — the cheerful announcer sign-off and closing organ music sit
   right on top of the murder scene and wash it out at median scale. The
   fear cluster survives at window scale, dies at summary scale.
2. **The categorical judge is UNUSABLE out-of-domain: joy = 610/1196
   windows (51%).** The e2v kNN reference DB is enrolled on modern
   close-mic voices; 1943 AM broadcast is a different acoustic planet, and
   the nearest-neighbor default lands on joy. Family colors are mostly
   noise here (the anger/fear clusters at extremes are the exception).
   Same lesson as every cross-domain failure since the RAVDESS days: the
   dimensional signal generalizes, the categorical layer is domain-bound.
3. Music/announcer segments are read as speech (the ear has no
   music/speech gate) — a real product would need VAD + music rejection
   up front.

**Verdict for the product question ("can it track live conversation?"):
the V/A trajectory engine works on found audio** — it located the inciting
shock, the sustained-tension acts, the anger scenes, and the climax
cluster of an 83-year-old drama at the correct timestamps. The family
layer needs domain-matched enrollment (the judge-v2 mechanism exists for
exactly this). Ledger unchanged (windows are analysis, not steering evals).

---

## FULL SPEECH IN THE USER'S CLONED VOICE — 4/7, and the NINTH BANK discovered (2026-08-13)

User provided two fresh recordings; clone_ref_1 (11.3s, ear reads it
sadness@100% A=0.19 — a calm flat read) became the speaker prompt. Seven-act
story, congruent text + proven vectors, concatenated to 60.3s. Every act
judged (indextts2-user-voice-speech, ledger 1,123).

| act | target | verdict | V / A |
|-----|--------|---------|-------|
| 1 | neutral | sadness@100% | +0.00 / 0.17 |
| 2 | joy | **fear@80%** | −0.33 / 0.20 |
| 3 | surprise | surprise@80% ✓ | −0.57 / 0.89 |
| 4 | fear | fear@60% ✓ | −0.28 / 0.10 |
| 5 | anger | anger@100% ✓ | −0.57 / 0.85 |
| 6 | sadness | sadness@80% ✓ | −0.26 / 0.20 |
| 7 | joy | sadness@60% | +0.02 / 0.18 |

**THE DISCOVERY — the speaker prompt is a hidden ninth bank.** The vendor
merge is `emovec = Σw·banks + (1−Σw)·speaker_emovec`. The prompt's own
emotional residue is weighted by (1−Σw) — and Σw varies wildly across our
proven vectors:

- anger Σw=1.00 → speaker emotion fully REPLACED → anger@100% ✓
- surprise Σw=0.80 → mostly replaced → surprise@80% ✓
- sadness Σw=1.10 → replaced (and prompt was sad anyway) → ✓
- fear Σw=0.47 → half speaker, but prompt-calm pulled A to 0.10 → weak ✓
- **joy Σw=−0.06 → (1−Σw)=1.06 → the subtract-heavy P5A joy vector
  AMPLIFIES the prompt's own emotion above unity.** With a melancholic
  prompt, "joy" delivered the prompt's sadness (A stuck at prompt's 0.19,
  both joy acts dark). The P5A vectors were optimized against ONE neutral
  RAVDESS prompt — subtract-heavy solutions are PROMPT-CONDITIONAL.

Law implication (joins congruence): subtraction buys valence only when the
speaker residue is neutral; for arbitrary cloned voices either (a) keep
Σw ≥ ~0.8 so banks dominate, (b) re-optimize per voice, or (c) drive joy
via the emo_audio reference channel (bypasses the Σw arithmetic). The
carriers were IMMUNE precisely because their recipes already replace the
speaker. Also confirmed: zero-vector "neutral" inherits the prompt verbatim
(act 1 = the prompt's own sadness@100%).

Product read: a 60-second six-emotion story in a cloned real voice, with
4/7 machine-certified on first take and a mechanistic account of all three
misses, from an 11-second sample. File: out/user_speech/full_speech.wav.

---

## 3-MIN SINGLE-FILE SPEECH — 5/7, reference-channel joy VINDICATED in the cloned voice (2026-08-13)

Same seven-act story, full paragraphs (~60-70 words/act), 0.25s joins ->
one continuous 2.4-min wav in the user's cloned voice. Ninth-bank fix
applied: joy acts via emo_audio reference (alpha 1.0) instead of the
subtract-heavy vector. System indextts2-user-voice-3min, ledger 1,130.

| act | route | verdict |
|-----|-------|---------|
| 1 neutral | zero vector | fear@60% (A=0.14 — zero-vector wobble again) |
| 2 joy | **reference channel** | **joy@100% ✓** |
| 3 surprise | bank 0.8 | surprise@80% ✓ (A=0.91) |
| 4 fear | P5A vector | fear@80% ✓ |
| 5 anger | bank 0.8+calm | surprise@60% (A=0.88, V=−0.54 — the shared
  anger↔surprise confusion axis; one draw, boundary noise; hit 100% on the
  60s version) |
| 6 sadness | P5A vector | sadness@60% ✓ |
| 7 joy | **reference channel** | **joy@80% ✓** |

**The headline: both joy acts CERTIFIED in the user's cloned voice —
joy@100% and joy@80% — after failing 0/2 with the subtract-heavy vector
yesterday.** The prediction from the ninth-bank analysis held precisely:
bypass the Σw arithmetic via the reference channel and joy survives an
emotionally-dark speaker prompt. First joy verdicts ever in the user's own
voice. Prompt-conditionality: diagnosed yesterday, cured today.

Running scoreboard for cloned-voice delivery: joy(ref) 2/2, surprise 2/2,
fear 2/2, sadness 2/2, anger 1/2 (draw noise), neutral 0/2 (zero-vector
remains the weakest recipe — inherits/wobbles). Product recipe for
arbitrary voices is now: carriers by bank vector, resonance by reference
channel or Σw≥0.8 vector, congruent text throughout.

---

## 2-MIN NARRATION — calm register locked, identity preserved (2026-08-13)

Per the user's register finding: five reflective paragraphs, zero vector
throughout (emovec = pure speaker inheritance), never leaving the prompt's
calm register. Single 1.86-min file. All five paragraphs judged
sadness@60–100% at A=0.14–0.20 — i.e., EXACTLY the profile of the prompt
itself (clone_ref_1: sadness@100%, A=0.19). The uniformity is the point:
the zero vector reproduced the user's own register faithfully across ~350
words. Identity via inheritance, confirmed at length. System
indextts2-user-narration, ledger 1,135.

---

## DEREVERB — the room leaves the voice (2026-08-15)

User heard reverb in the narration. Diagnosis: the cloning reference
carried real room reverb + noise (floor −71 dB), and IndexTTS-2 clones the
ROOM as faithfully as the voice; the vocoder adds slight smear on top.

**Fix at the source, not the output.** WPE dereverberation
(tts_steering/dereverb.py — STFT late-tail prediction, taps=10, delay=3):
- On the REFERENCE: floor −71 → **−98 dB**, speech-to-floor gap +16 dB.
  Re-normalized to peak 0.6, saved as own_voice/clone_ref_1_dry.wav.
- On the synthesized OUTPUT: no improvement (the synth smear is not
  predictable room reverb) — route tested, documented, dropped.

**Re-render from the dry prompt (user_narration_dry.py, system
indextts2-user-narration-dry, ledger 1,140):** pause floor in the output
went from ~−71 dB room tone to DIGITAL SILENCE (speech-to-floor gap ~40 dB
→ ~160 dB). The model cloned the dead room exactly as it had cloned the
live one — proof that prompt hygiene is a first-class steering input.
Judge note (honest): the dry voice reads slightly brighter (para 1/4 joy@
40–80%, A up to 0.28 vs 0.14–0.20 wet) — register shifted subtly; the
user's ear is the final gate on identity.

Law addendum (ninth bank): the prompt contributes not just emotional
residue and register-bound identity, but the ROOM. Standing rule for all
future references: record dry, then WPE + normalize before first use.

---

## MULTI-REGISTER + AUTO-RETRY — ninth bank becomes an ally; surprise teaches headroom (2026-08-15)

First combined run of the two product mechanisms. Register bank: three
user memos (neutral V=−0.01/A=0.25 · excited V=+0.43/A=0.45 — the user's
highest-ever recorded valence · sad V=−0.19/A=0.13/D=−0.52), all WPE'd +
normalized per the room law. 7-act story, register-matched prompts,
certification loop (≤3 rounds, best take assembled). System
indextts2-multi-register, 12 attempts ledgered, total 1,152.

**HEADLINE — joy by VECTOR in the user's voice, both acts, round 1.**
Act 2 joy@60%; act 7 joy@40% at d=0.15 (V=+0.17, A=0.58 — essentially ON
the centroid). Two days ago the same vector delivered the prompt's
melancholy (0/2); with the excited prompt its (1−Σw)=1.06 residue
amplifies +0.43 valence instead. The ninth bank flipped from saboteur to
fuel exactly as the arithmetic predicted. No reference clip needed —
prompt selection IS an emotional steering input.

Scoreboard: joy ✓✓ (vector, r1) · anger ✓@100% (r1, excited prompt) ·
fear ✓ (r1) · sadness ✓ (r2 — the retry loop earning its keep) ·
neutral ✗ (3 rounds, dimensionally fine V=+0.02/A=0.31 but no categorical
verdict — the known weakest recipe) · surprise ✗✗✗.

**The surprise failure is a new law tile — HEADROOM.** With the excited
prompt, surp 0.8 pegged arousal at 1.00 all three rounds and read as
anger/joy (V=−0.65). Surprise was 16–17/19 with CALM prompts: the sudden
pitch spike needs CONTRAST against a quiet baseline. A saturated prompt
leaves no headroom for the spike that IS surprise. Register matching is
per-emotion physics, not "loud emotion ← loud prompt": anger wants a hot
baseline; surprise wants a cold one.

Auto-retry verdict: works as designed (sadness rescued in r2; misses kept
best-effort by centroid distance; every attempt ledgered). Final file
1.53 min, 5/7 certified: out/user_speech/multi_register_speech.wav.
Next fixes queued: surprise → neutral prompt; neutral → needs its own
optimization run.

---

## STATIC FIX + HEADROOM VINDICATED — v2 render (2026-08-15)

User heard old-TV static in the multi-register speech. Diagnosis: the three
register memos were HISSY (speech only 20–29 dB above floor; the neutral
memo worst at 20 dB) and the model cloned the hiss — v1 joy act carried
2.7x the high-frequency quiet-frame noise of the clean-prompt baseline
(0.239 vs 0.088). WPE was the wrong tool (reverb ≠ hiss); spectral
denoising (noisereduce, stationary, 0.92) lifted prompt gaps to 31–40 dB.

**Prompt-hygiene chain now settled: record → DENOISE (hiss) → WPE
(reverb, only if present) → normalize.** Each link was taught by a
different failure the user's ear caught (reverb 08-15a, hiss 08-15b).

v2 render (denoised prompts + surprise→neutral-prompt headroom fix,
indextts2-multi-register-v2, 12 attempts, ledger 1,164):

- **Static measurably gone:** joy-act HF ratio 0.239 → 0.093 (baseline
  0.088). At baseline. The ear should hear a clean voice.
- **Joy: CERTIFIED @100% BOTH acts, round 1** (was 60/40% through the
  hiss) — cleaner prompt, stronger verdict.
- **SURPRISE: CERTIFIED (r2) — first surprise ever in the user's voice.**
  The headroom law works: cold neutral baseline, arousal off the 1.00
  ceiling (0.91–0.93), verdict flipped from anger to surprise.
- fear ✓ r1, sadness ✓ r1.
- **Anger: 0/3 this run** (all draws → surprise, A 0.86–0.94) — with the
  DENOISED excited prompt, anger keeps sliding across the shared
  high-arousal boundary. Noted honestly: v1's noisy prompt certified
  anger@100% — the hiss may have read as vocal grit. Anger vs surprise on
  a hot prompt is now the last unstable pair; candidate fixes: neutral
  prompt for anger too (headroom), or bank 0.9 + calm 0.1.
- neutral: 0/3, as always (dimensionally clean V≈+0.05/A≈0.29).

Certified 5/7. File: out/user_speech/multi_register_speech_v2.wav
(1.96 min). Register-matched identity + clean audio + certified emotions —
the user's three complaints (different-person shouting, reverb, static)
each diagnosed to mechanism and fixed in sequence.

---

## LIVE EAR v1 — real-time parallel emotion mapping, 8x headroom (2026-08-15)

Goal reframe (user): THE EAR IS THE MAIN PRODUCT — map emotions of
parallel-playing audio (interview/movie) in real time: 3s intervals, V/A/D
vectors, named emotions, live graph. Cloning = side gig.

Built scripts/live_ear.py: WavLM-ft RESIDENT in memory (no subprocess per
call), full-file streaming walk (3.0s window / 1.5s stride — the P4.14
protocol), PAD-centroid naming (domain-general; the enrollment-bound kNN
judge is deliberately NOT in this path — the 1943 test showed why), live
matplotlib trajectory while afplay plays the same file in parallel
(--play), JSON + PNG per run. Two loader fixes: predict_wavlm_ft.load_audio
caps at 8s (training mirror) → max_s override; per-WINDOW peak
normalization to mirror per-clip training norm.

**Validation (1943 broadcast, minutes 4:00–5:30 — the overheard-murder
region), fast mode, 59 windows:**
- **Median inference 196 ms/window on MPS → 7.6x real-time headroom** at
  the 1.5s stride. Real-time is proven with margin to spare.
- The trajectory READS: light neutral/joy opening → surprise build →
  spike (V=+0.52, A=0.93) → fear at 30s → two sustained ANGER blocks
  (V −0.5…−0.83, A ~0.85) exactly where the plot turns. Named emotions at
  1.5s resolution track the scene structure without any enrollment.
- Bonus: the MSP namer speaks a richer palette than the 6-family judge
  (contempt and disgust appear natively).

Usage (the parallel-playing demo):
  .venv_diar/bin/python scripts/live_ear.py --input any.wav --play
Known limits, honest: file-based (true system-audio capture of e.g. a
Netflix stream needs a loopback device — BlackHole — queued); no
music/speech gate yet; ambiguity flag shown per window.

---

## SPEECH GATE — the ear learns what not to feel (2026-08-15)

Silero VAD integrated into live_ear.py as a resident per-window gate:
mean speech probability over the 3s window, causal 3-window median
smoothing (a single flicker can't gate mid-sentence), threshold flag
--speech-gate (default 0.5; 0 disables). Gated windows cost 9 ms (WavLM
skipped) vs 216 ms — the gate SPEEDS UP music-heavy audio. Raw per-window
probabilities always recorded in the JSON: gating is auditable, never
silent.

**Full-broadcast validation (1943, 1,196 windows): 443 suppressed (37%),
and the suppression map matches the program structure exactly** — intro
music/announcer (0–4 min) heavily gated, mid-show break (14–16) gated,
closing organ (28–30, p=0.00–0.05) gated, while the continuous-dialogue
stretch at 10–12 min runs 0/80 gated after smoothing (was 3/80 raw).

**Family distribution over speech windows, PAD-centroid naming:**
surprise 246 · anger 202 · joy 112 · contempt 90 · fear 54 · neutral 34 ·
disgust 10 · sadness 5. For a suspense drama about alarm and
confrontation, 79% negative-activated is the RIGHT shape — against the
old ungated e2v run's 51%-joy nonsense. Gate + domain-general naming
together fix what the real-world test exposed. Graph:
docs/live_ear_gated_1943.png.

Live-ear v1 is now: resident WavLM + Silero, 3s/1.5s protocol, 216 ms/
window (7x headroom), speech gate, live graph, parallel playback, JSON
audit trail. Remaining roadmap: system-audio loopback capture (BlackHole)
· emotion-name smoothing · identity judge (side gig).

---

## LIVE CAPTURE MODE — the ear grows a device input (2026-08-15)

live_ear.py refactored around a resident `Ear` class with two sources:
file mode (--input/--play/--fast, unchanged) and STREAM mode — an ffmpeg
f32le pipe read in stride-sized chunks into a rolling window buffer:
  --device N     avfoundation capture (mic, or BlackHole loopback for
                 system audio); Ctrl-C or --duration to stop
  --simulate f   the SAME pipe fed by ffmpeg -re (real-time paced file) —
                 validates the live plumbing with no driver or permission

Validation (--simulate, 1943 shock segment, 30s): chunks flow at wall-
clock pace, gate + emotion + plot all live, verdicts consistent with
file-mode on the same audio, median 262 ms/window -> 6x headroom.
Note: the causal gate median lags ~1 window at speech onsets (documented).

BlackHole status: brew present, cask NOT installed (driver needs the
user's admin password — interactive). Remaining user steps: (1) `brew
install blackhole-2ch`, (2) Audio MIDI Setup -> Multi-Output Device
(speakers + BlackHole) as system output, (3) run `--device <BlackHole
index>`. The engine side is plug-and-play ready.

---

## LOOPBACK COMPLETE — system audio to live emotion graph, zero GUI (2026-08-15)

The full chain the goal reframe asked for now exists:
1. BlackHole 2ch installed (user's password), loaded WITHOUT reboot via
   coreaudiod restart — device [0].
2. **Audio MIDI Setup step automated away**: scripts/ear_multiout.swift
   (CoreAudio, ~120 lines) creates a stacked aggregate "Ear Multi-Output"
   (speakers master + BlackHole with drift compensation) and sets it as
   system default output. `out/ear_multiout revert` restores speakers and
   destroys the aggregate. No GUI touched.
3. End-to-end proof: afplay through the multi-output while live_ear
   captured --device 0 — the playing clip was heard, gated, and mapped
   live (surprise/anger/joy at plausible story points, 277 ms/window, 5x
   headroom). Heavy gating on this test = the 1943 telephone-filtered
   source sits near the speech threshold; modern audio passes cleanly.

THE DEMO: play anything (YouTube, movie, podcast) →
  .venv_diar/bin/python scripts/live_ear.py --device 0
Known macOS quirk: volume keys don't control aggregate devices — set
volume before, or revert when done.

---

## FIRST WILD SESSION — GTO finale, live via loopback: the three-act arc read cross-lingually (2026-08-15)

User played the final episode of GTO (Great Teacher Onizuka) through the
new loopback while live_ear ran unattended: 626 windows / 15.6 min, zero
crashes, latency flat (median 273 ms, max 435), post-video silence 80/80
gated. Forensic audit of the saved JSON against the known content:

**The finale's dramatic structure is IN the trajectory:**
- first third: V −0.17, A 0.83, anger/surprise (setup tension)
- middle: **V −0.28, A 0.90, anger dominant 68 windows** incl. an
  11-window (16.5 s) sustained anger run — the climax confrontation
- final third: anger collapses 68→17, V recovers to −0.06
- **last 90 s of speech: V median +0.21, peak +0.79, joy appears —
  the warm resolution. Valence slope over the final 3 min: +0.156/min.**
Conflict → climax → resolution, measured. No labels, no enrollment.

**Cross-lingual, cross-cultural transfer confirmed:** the model is
trained on English natural speech (MSP); the content is Japanese anime
voice acting. Prosody carried the dimensional signal across the language
boundary. The arousal ceiling (A median 0.86, saturating at 1.00) is now
largely EXPLAINED rather than suspicious: seiyuu delivery is
hyper-activated by design — the ear read anime acting as extreme because
it IS extreme. (Loudness compression + music beds under speech remain a
secondary inflator; calibration item stands.)

**Flaws confirmed by the wild data (queued):** 75% ambiguity in the hot
corner where anger/surprise/fear centroids crowd; 46% name flicker
between adjacent windows (needs majority-vote smoothing); produced-media
loudness normalization; ffmpeg Ctrl-C stderr spam (cosmetic).

Verdict logged: architecture sound, calibration next. The ear watched
television in a language it never learned and read the story right.

---

## CORRECTION to the GTO entry — it was the ENGLISH DUB (2026-08-16)

The user played the English dub, not the Japanese original. The
"cross-lingual transfer confirmed" claim in the previous entry is
WITHDRAWN — the three-act-arc reading stands (that verification was about
dramatic structure, not language), but no cross-language evidence was
collected. Error mine: I assumed the language instead of asking.

**Turned into a pre-registered experiment (user's design): DUB vs SUB.**
The user will replay the SAME episode in Japanese through the same
loopback pipeline. Protocol:
1. Align the two trajectories automatically (cross-correlation of the
   arousal curves finds the time offset — start points differ).
2. Compare over the aligned overlap: V and A correlation, per-third
   medians, family distributions, ambiguity rates.
Predictions (registered before the Japanese run):
- If the ear reads prosody independent of language: aligned V/A
  correlation clearly positive (r >= ~0.5 despite different actors,
  mixes, and takes).
- Known confound to expect: seiyuu delivery tends hotter than dub
  delivery — a constant arousal offset with correlated SHAPE would still
  support language-independence; uncorrelated shapes would refute it.

**User's prediction, registered before the Japanese run:** the patterns
WILL differ — languages place pitch, jitter, and timing differently on
words, so the graph should not match. Formalized as a three-level
scoring, so whatever happens is interpretable:
- MICRO (window-by-window values): both hypotheses expect DIFFERENCE
  (different actors, words, takes) — not diagnostic.
- MESO (scene-by-scene V/A shape, aligned): the battleground. User's
  instinct -> weak correlation; language-independence -> strong.
- MACRO (three-act arc direction): if even THIS diverges, the ear is
  strongly language-bound.
Additional registered question: does Japanese shift the BASELINE (e.g.,
pitch-accent prosody reading as systematically hotter A to an
English-trained model = calibration bias) vs scramble the SHAPE
(= genuine language dependence)? Offset and correlation are scored
separately for exactly this reason.

---

## DUB vs SUB — RESULTS: the shape survives the language (2026-08-17)

Japanese run captured through the same loopback: 575 windows / 14.4 min,
`out/live_ear/device0_1786960341_traj.json`. Speech gate 287/575 (50%)
vs English 306/626 (51%) — Silero gates Japanese identically. Alignment
by arousal cross-correlation: lag +21.0s (JA started earlier in the
episode), aligned overlap 13.5 min, 263 windows where both runs heard
speech.

**Scored against the pre-registered three levels:**

| Level | Metric | Result | Verdict |
|---|---|---|---|
| MICRO (window) | V r=0.364 · A r=0.693 · family agreement 33% | different | as both predicted — not diagnostic |
| MESO (30s scenes) | V r=0.596 · A r=0.735 | above the r≥0.5 bar | shape correlates |
| MESO (60s scenes) | V r=0.667 · A r=0.801 | strongly correlated | shape correlates |
| MACRO (thirds) | act-2 dip + act-3 recovery in BOTH (EN −0.14/−0.28/−0.07 · JA +0.09/−0.15/+0.01) | same arc | language-independent |

Aligned same-content final act: warm resolution present in both — V
median +0.21 (EN) vs +0.17 (JA), peak +0.79 vs +0.69. (Tail slope not
scoreable: only 6 speech windows survive the gate there — ending is
music.)

**OFFSET vs SHAPE (the registered split): both hypotheses were partly
right.**
- SHAPE: scene-level V/A correlation r=0.6–0.8 across different
  languages, different actors, different mixes, different takes. The ear
  tracks the STORY's emotional dynamics, not English phonetics. My
  registered prediction — supported at MESO/MACRO.
- OFFSET: valence baseline shifts +0.13 warmer in Japanese (arousal
  offset −0.04, negligible). Family naming redistributes: anger 38%→24%,
  fear 19%→12%, joy 5%→12%, neutral 2%→7%. The user's registered
  instinct — supported at the calibration level: language DOES move
  where the values sit, exactly as "same words, different pitch/jitter
  placement" predicts. Window-level naming is language-sensitive.
- Direction note: I had guessed Japanese might read HOTTER
  (pitch-accent → higher A). Wrong direction — it reads WARMER-VALENCED
  and equally aroused. The dub's line delivery lands as angrier to the
  ear than the seiyuu original.

**Standing conclusion:** the dimensional ear (V/A/D) is
language-independent at the scene scale and above — the level the
product operates at. The categorical namer inherits a per-language
valence offset (+0.13), which folds into the existing calibration queue
(loudness norm, name smoothing) as a known, measured bias — not a
structural failure. The ear watched the same story twice in two
languages and drew the same arc both times, 0.13 warmer in Japanese.

Ambiguity 75%→72% (unchanged — hot-corner crowding is content-driven,
not language-driven).
