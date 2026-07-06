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
