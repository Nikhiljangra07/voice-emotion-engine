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
3. **Python version pin** — IndexTTS-2 requires ≥3.10 and its stack is happiest
   below 3.13; our main tooling is 3.13. Resolution: vendor env pinned to **3.12**
   via uv, fully separate from `.venv_tts` (3.13).

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

**In progress:** IndexTTS-2 vendor env sync (uv, Py3.12) + 5.9 GB weights download —
running in background (`vendor/setup_uv.log`, `vendor/setup_weights.log`).

**Next (P4.2):** first synthesis — manual emotion-vector sweep on the three target
emotions, listen + measure through the bridge, log the first generated-audio scores.
