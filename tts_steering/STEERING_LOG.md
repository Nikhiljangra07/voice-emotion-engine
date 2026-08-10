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
