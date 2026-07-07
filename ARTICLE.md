# The Axis Everyone Said Was Unrecoverable: Valence from Voice, 0.06 → 0.705

*How a solo, self-taught builder measured the ceiling of handcrafted acoustic features — and then broke it.*

---

There's a piece of folk wisdom in speech-emotion research: **you can hear *how activated* a speaker is, but not *how they feel about it*.** Arousal — loud, fast, tense — leaks into pitch and energy and is easy to recover from audio. Valence — whether the feeling is pleasant or unpleasant — hides. The handcrafted-feature literature keeps finding the same thing: valence correlations from acoustic features alone are weak to useless.

I spent months finding that out the hard way, on purpose. Then I broke it. This is the record of the arc: **CCC 0.06 → 0.35 → 0.705** on valence, all on naturalistic speech, all speaker-independent.

## Why I cared about voice at all

I build [LoRa](https://github.com/Nikhiljangra07/LoRa-EmotionalEngine-v1), an analytical reasoning partner that ran as a public beta in spring 2026. Its first emotional-signal pipeline was text-based, and text hit a ceiling I documented at the time: people mask their emotions in text. What they can't easily mask is prosody. So the question became concrete: **how much emotional signal actually lives in the voice channel — measured honestly, on natural speech, on speakers the model has never heard?**

The measurement rules were fixed before anything was trained:

- **Naturalistic data**, not acted corpora. Actors project emotions; real people leak them. (I later measured this directly: performance on acted datasets did not transfer to natural speech in either direction.)
- **Speaker-independent evaluation.** The test speakers never appear in training. No memorizing voices.
- **CCC (Concordance Correlation Coefficient)** as the metric — it punishes both bad correlation *and* bad calibration, and it's what the field's benchmark papers use.

Training and evaluation used **MSP-Podcast v2.0** (Lotfian & Busso; UT-Dallas), the largest naturalistic speech-emotion corpus available — with a **~46,000-clip held-out test set from unseen speakers**.

## Phase 1: handcrafted features — 0.06, then 0.35

The first system was the classic recipe, built from scratch so I would understand every stage: pitch statistics, energy, speech rate, pause structure, spectral features — regression head on top.

First honest run on valence: **CCC 0.06**. Statistically indistinguishable from guessing.

Diagnosis, iteration, better features, better pooling, cleaner labels: **0.35**. And there it stopped. Every further intervention — more features, different heads, different aggregation — moved decimals, not the number. 0.35 on valence is roughly where the handcrafted literature sits, and now I had reproduced the ceiling myself instead of taking the literature's word for it.

That failure was the finding. **The information wasn't in my features.** Either valence genuinely isn't in the audio — or it's in the audio in a form that pitch-and-pause statistics can't see.

## Phase 2: change the representation, not the features

The bet: valence lives in *how* things are said at a resolution handcrafted features destroy — micro-prosody, voice quality, the texture self-supervised speech models learn from thousands of hours of raw audio.

So: **WavLM-large, fine-tuned end-to-end** — not used as a frozen feature extractor, but with the whole backbone learning the task — mean-pooled, with a small regression head predicting Valence, Arousal, and Dominance jointly.

Result on the same 46k held-out unseen-speaker test set:

| Dimension | CCC |
|---|---|
| **Valence** | **0.705** |
| Arousal | 0.714 |
| Dominance | 0.626 |

Valence went from "barely above noise" to **on par with arousal** — the axis that was supposed to be the easy one. The full training log ships with the weights.

**The lever was the representation, not the features.** Twice the feature-engineering effort would have bought me 0.40. Changing what the model *sees* bought 0.705.

## What I'd tell someone attempting this

1. **Build the baseline you intend to kill.** Reproducing the handcrafted ceiling myself — instead of citing it — is what made the fine-tune result meaningful. 0.06 → 0.35 is the part of the arc that gives 0.705 its weight.
2. **Acted corpora will lie to you.** My phase-1 numbers on acted datasets looked respectable and transferred to nothing. Every headline number here is naturalistic speech or real phone-mic audio.
3. **Speaker independence is non-negotiable.** Valence models are excellent at memorizing *people*. If your test speakers appear in training, you have measured nothing.
4. **Fine-tune the backbone.** Frozen self-supervised features are a half-measure; the valence signal only fully appeared when the representation itself adapted to the task.

## The artifacts

Everything is public and reproducible:

- **Model weights + training log:** [Nikhil0097/wavlm-large-emotion-vad](https://huggingface.co/Nikhil0097/wavlm-large-emotion-vad)
- **Full project record** (both phases, including the failure): [github.com/Nikhiljangra07/voice-emotion-engine](https://github.com/Nikhiljangra07/voice-emotion-engine)
- The repo also contains a second, deliberately isolated project: emotional TTS steering with a fair benchmark against ElevenLabs — honest mixed results, documented the same way.

*(MSP-Podcast itself is research-licensed and not redistributable — request access from UT-Dallas. The model is released CC-BY-NC-SA 4.0 for that reason.)*

---

*I design and build AI systems end-to-end, working solo, with AI coding tools as the hands and the design, evaluation methodology, and judgment as my job. More of the record: [github.com/Nikhiljangra07](https://github.com/Nikhiljangra07).*
