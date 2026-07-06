# Emotional TTS Steering & Fair Benchmark

**A closed feedback loop that steers text-to-speech toward target emotions, judged
by a frozen voice-emotion engine — benchmarked fairly against ElevenLabs v3, Hume
Octave, and OpenAI TTS, with every mistake kept and the headline result retracted
when a fairer test dissolved it.**

> **This is Project 2 of this repository** — a separate system *powered by* the
> [Voice Emotion Engine](../README.md) (Project 1), not part of it. The engine is
> the judge; this project is the contestant-wrangler. They touch through exactly
> one file — [`bridge.py`](bridge.py), subprocess → CLI → JSON — with zero code
> imports in either direction and fully separate environments.

*Companion doc: [STEERING_LOG.md](STEERING_LOG.md) — the full chronological record
(every hurdle, every miss). Charter: [README.md](README.md).
Data: [loop_ledger.csv](loop_ledger.csv) — all 130 judged clips.*

---

## 1. The claim (sized honestly)

> A deterministic feedback loop around a frozen perceptual judge finds the best
> emotional operating point of ANY TTS control surface — local or commercial —
> in ≤5 judged attempts, and in doing so characterizes both the mouths (all five
> systems share a high-intensity collapse) and the ear (a sentence-dependent
> synthetic-sadness lock, a joy-attractor on some texts). On this instrument, a
> $0 local stack remains competitive with commercial APIs — winning no category
> cleanly and losing only joy.

This is the *third* version of the claim. Version 1 ("we beat the commercial
APIs") died under our own cross-examination (§6). The retraction is part of the
result.

## 2. The idea

The parent repo reads emotion from voice (fine-tuned WavLM → valence/arousal/
dominance + a frozen emotion2vec family judge). This project points that ear at a
synthetic mouth and closes the loop:

```
IndexTTS-2 (mouth)  →  clip  →  bridge  →  WavLM V/A/D   (steering signal)
     ↑                                      e2v family    (verdict — frozen)
     └────── deterministic proposer ←──── distance to MSP centroid
```

**Anti-circularity design:** steering runs on WavLM distance; a HIT is declared
only by the *independent* e2v judge (different model, different vector space).
**Judge-frozen law:** loop data may improve the mouth's control policy, never the
ear — retraining the judge on its own synthetic verdicts is how a system starts
grading its own homework.

Targets: **sadness, joy, anger** (data-chosen — judge recall 90/86/69%; fear
excluded for the judge's known 47% blind spot). Every generated clip becomes a
row in the ledger, misses included: 130 rows across 5 systems in 2 days.

## 3. What was built

- **`bridge.py`** — the only connection to the parent engine: subprocess → CLI →
  JSON. Zero shared imports; four isolated venvs (project 3.13 / vendor 3.11 /
  chatterbox 3.12 / engine untouched).
- **`optimize_p43.py`** — the autonomous optimizer: warm-start seeds, rule-based
  error-keyed proposals (no RNG), dedup, budget caps, converged emotions stop
  spending.
- **`fair_p44b.py`** — the fair benchmark harness: the same loop applied to
  ElevenLabs (v3 audio tags + meter-picked voice), Hume Octave (acting
  descriptions), OpenAI TTS (instructions + meter-picked voice), Chatterbox
  (reference + exaggeration), plus a generalization round on unseen sentences.
- **Blind listening protocol** — hash-shuffled anonymous clips with real-voice
  controls and a sealed answer key, for the human gate (§5).

## 4. Headline results

**The loop improves itself, and refuses to fake it.** Anger converged
autonomously (`angry=0.7`, distance 0.227→0.207, judge anger@100%); sadness
improved 24% to a measured floor (d=0.134); joy honestly failed from this mouth
(8 attempts, zero positive valence).

**Fair-fight scoreboard** (every system loop-steered, ≤5 attempts/emotion, same
frozen judge; distance to 137k-clip MSP centroids, lower is better):

| emotion | ours (IndexTTS-2) | Chatterbox | ElevenLabs v3 | Hume Octave | OpenAI |
|---|---|---|---|---|---|
| anger | HIT@100% 0.207 | HIT@60% **0.171** | HIT@59% 0.380 | HIT@100% 0.256 | HIT@80% **0.187** |
| joy | ✗ 0.354 | ✗ 0.309 | **HIT@80% 0.144** | ✗ | ✗ 0.495 |
| sadness | **0.134**★ | 0.162 | 0.357 | 0.210 | 0.240 |

★ = judged neutral by the machine but confirmed *sad* by blind human ears (§5);
best sadness acoustics of all five systems.

**Cross-system findings (replicated, not anecdotal):**

1. **The moderation law.** On every system, maximum-intensity emotion collapses:
   over-driven joy reads as fear/anger, over-driven anger as fear, Hume's
   "ecstatic celebration" flew to d=0.956/surprise. Emotional TTS has a sweet
   spot; past it, arousal swamps valence.
2. **The synthetic-sadness lock.** The e2v judge called synthetic sadness
   "neutral" in ~30 attempts across all five mouths — yet blind human ears heard
   sadness, and on one held-out sentence the lock finally broke (first-ever
   synthetic sadness verdicts: ours@40%, Chatterbox@60%). The blind spot is
   real and *sentence-dependent*.
3. **Joy is the hardest direction.** Ten one-shot attempts across four systems
   produced zero positive WavLM valence; only loop-steering with an expressive
   voice cracked it (ElevenLabs `[excited][laughs]`, d=0.144).
4. **Sentence instability dominates.** Configs tuned on one sentence transfer
   poorly (best systems: 3/6 family hits on unseen sentences). The n=1 caveat
   was not pedantry.

## 5. The human gate (blind protocol)

11 clips, names hash-shuffled, answer key sealed until scoring: 8 synthetic + 3
real-voice controls. The listener got **all 3 controls right** (calibrated ears),
scored 9/11 against intent, and — decisively — heard **sadness** in both clips
the machine judge had called neutral@100%, and **joy** in a clip the judge
called fear@100%. Verdicts: sadness was mouth-delivered and ear-blocked; joy is
split by intensity (moderate = human-real, high = genuinely broken); anger closed
with triple agreement (machine + human + intent).

Consequence, kept binding: the judge stays frozen; its blind spots are
*documented* instead of trained away.

## 6. The catch (what this does NOT show)

Raised from inside the project before any external review:

1. We won on our own scoreboard — our judge, our centroids. Iterating against a
   fixed metric beats one-shot competitors on that metric (Goodhart in our
   favor). The fair rematch (§4) was built to attack this, and it dissolved most
   of our original "win": anger became a four-way tie, ElevenLabs took joy.
2. We never measured naturalness — the thing commercial TTS is actually selling.
   No MOS test; their clips likely sound more human.
3. OpenAI's emotional flatness is plausibly a product choice, not incapacity.
4. Probe-scale: n=1–2 per cell, one steering sentence, one voice per system.
5. The judge has documented synthetic blind spots (§4.2), so family-hit counts
   undercount sadness for every system symmetrically.

## 7. What's next (documented, not promised)

- **P4.6 transfer map:** loop over ~20 varied sentences to map sentence-
  dependence — also generates a training-grade ledger.
- **Phase 5 (gated):** rejection-sampling fine-tuning of the mouth on its own
  judge-approved, human-ratified best clips — success metric fixed in advance
  (one-shot family-hit rate on held-out sentences, before vs after). Blocked on
  a time-boxed feasibility spike (IndexTTS-2 has no published training code).
  Fallback: a sentence-aware control policy distilled from the ledger.

## 8. Artifacts

| artifact | what it is |
|---|---|
| [loop_ledger.csv](loop_ledger.csv) | all 130 judged clips: control params → V/A/D → verdict |
| [STEERING_LOG.md](STEERING_LOG.md) | chronological record: adoptions, rejections, 12+ hurdles, retractions |
| `bridge.py` / `optimize_p43.py` / `fair_p44b.py` | the loop, reusable |
| `out/listen_check/` | blind-protocol response sheet + sealed key (local) |

**Systems:** IndexTTS-2 (Bilibili; weights non-commercial without authorization —
this is non-commercial research) · Chatterbox (Resemble AI, MIT) · ElevenLabs v3 ·
Hume Octave · OpenAI gpt-4o-mini-tts. **Judge:** the parent Voice Emotion Engine
(WavLM-ft + emotion2vec+, frozen).

**References:** IndexTTS2 (arXiv:2506.21619) · emotion2vec (arXiv:2312.15185) ·
WavLM (arXiv:2110.13900) · MSP-Podcast corpus (centroids fit on 137k clips) ·
parent engine writeup: [../WRITEUP.md](../WRITEUP.md).
