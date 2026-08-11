# Phase 5 Feasibility Spike — verdict (2026-08-11)

## What the dig found (vendor code, read line by line)

IndexTTS-2 ships **inference only** — no training code, no Trainer, no loss
anywhere in the repo. A LoRA on its 1.5B GPT would mean writing the training
stack from scratch against undocumented internals. That was the plan's risk —
and it turns out we don't need it.

**The architecture reduces all emotion control to ONE vector.** The complete
conditioning path (`infer_v2.py`):

```
emovec = gpt.merge_emovec(spk_emb, emo_ref_emb, alpha)      # audio-derived
if emo_vector:                                              # slider path
    emovec_mat = Σ_i  w_i · emo_matrix[i][row]              # bank arithmetic
    emovec     = emovec_mat + (1 − Σw) · emovec
gpt.inference_speech(..., emo_vec=emovec)                   # that's it
```

- `emo_matrix` is a **plain tensor file in the checkpoint**: 8 emotion banks,
  row selected by cosine similarity to the speaker style.
- The 8 sliders are nothing but *weights over stored embeddings* — with the
  library silently applying bias (surprised ×0.69, calm ×0.56) and a Σ≤0.8
  clamp (P4.12 discovery).
- The final `emovec` — a single embedding — is the entire emotional will of
  the mouth for that utterance.

## VERDICT — Phase 5A: optimize the embedding, not the network

"Fine-tuning" the emotion pathway = **searching for better `emovec` vectors
directly**, with the frozen judge as the objective. No training code, no GPU
rental, no weights touched:

- **Search space (10–15 dims, not 512):** coefficients over the 8 native
  banks (UNCAPPED and allowed NEGATIVE — P4.13's "subtract tension" becomes a
  literal operation: `joy_candidate = happy_bank − β·angry_bank + …`) plus
  audio-derived emovecs from chosen reference clips (EmoKnob arithmetic in
  the mouth's own native space).
- **Optimizer:** CMA-ES / coordinate search, ~100–150 synth+judge evals per
  target emotion ≈ one overnight local MPS run per emotion. $0.
- **Implementation:** a small `emovec_worker.py` subclassing the vendor class
  to accept raw coefficients (vendor stays unmodified; engine repo stays
  additive-only).
- **Targets, in order:** joy (judge-confirmed reachable — push confidence and
  sentence-independence), sadness (widen beyond the mel route), then
  exploratory fear (the afraid bank exists; the judge's 47% blind spot noted).

**Phase 5B (fallback, only if 5A saturates):** Maya1 (Apache-2.0, public
fine-tuning toolkit, SNAC/Llama) LoRA-trained on our judge-approved corpus.
GPU class: A40/4090 (~$0.30–0.70/hr, $5–15 total). The RTX PRO 6000 is not
required at any point.

## Gate 4 — reward-hacking (Goodhart) mitigation, pre-registered

1. **Train/held-out split:** optimization runs on a fixed sentence set;
   the reported metric is one-shot family-hit rate on UNSEEN sentences.
2. **Two instruments:** judge family verdict AND WavLM V/A/D distance must
   both improve; a candidate that games one metric fails the other.
3. **Hull constraint:** coefficient norms bounded (candidates stay within a
   documented neighborhood of the native banks) — no adversarial-noise
   embeddings.
4. **Eval budget cap:** 150 evals/emotion, logged; every eval a ledger row.
5. **Human final gate:** long-clip blind ratification (P4.14-style journeys,
   not 2-second fragments) before any claim.
6. **The judge is never retrained. Ever.**

## Sequencing

1. `emovec_worker.py` + smoke test (custom emovec → clip → judge) — 1 session
2. Live-voice demo (independent of 5A): ear reads live audio → mouth replies
   with proven configs — the end-to-end product moment, $0
3. 5A joy run (overnight) → held-out eval → human gate
4. 5A sadness → same · exploratory fear
5. Only then, if needed: 5B on a cheap GPU

*Spike cost: $0. Verdict: Phase 5 is local. The mouth's emotions live in one
vector, and we own the arithmetic around it.*
