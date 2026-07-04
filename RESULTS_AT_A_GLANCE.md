# Results at a Glance — Voice Emotion Engine

*Reading emotion from **how** a voice sounds. One page. Full report: `WRITEUP.md`.*

---

### The headline

> Voice alone reveals **arousal** and **dominance** reliably — and, once you
> fine-tune a self-supervised model, **valence too** (the axis everyone calls
> language-bound). Naming an unseen speaker's emotion family reaches **≈ 69%**
> (chance 17%) by fusing a frozen emotion model with a fine-tuned one.

---

### Dimensional prediction (PAD) — fine-tuned WavLM on MSP-Podcast

*Metric: CCC (concordance correlation), speaker-independent. ~0.6 is strong in this field.*

| Split | Valence | Arousal | Dominance | Mean |
|---|---|---|---|---|
| Dev (34k) | 0.728 | 0.675 | 0.598 | 0.667 |
| **Test1 (46k, held-out speakers)** | **0.705** | **0.714** | **0.626** | **0.681** |

**The valence wall, cleared:** `0.059` (naive probe) → `0.347` (handcrafted SVR
ceiling) → **`0.705`** (fine-tuned WavLM). The lever was the *representation*, not
the transcript.

---

### Emotion naming — cross-speaker, leave-one-speaker-out (175 own-voice clips)

| Scorer | Accuracy |
|---|---|
| emotion2vec-kNN (frozen) | 63.4% |
| WavLM-ft + head | 68.0% |
| **Hybrid (equal weight)** | **68.6%** |

Per-family recall (hybrid): sadness 90% · joy 86% · surprise 70% · anger 69% ·
neutral 57% · **fear 47% (the weak point)**. Chance = 17%.

**Counter-intuitive win:** the *frozen* emotion model out-generalizes our own
*fine-tuned* one for retrieval (63.4% vs 52.6% kNN) — so we score WavLM with a head
and fuse the two at the decision level (never mixing vector spaces).

---

### Speaker separation (diarization, ECAPA-TDNN)

- Speaker count known → **~100%** per-turn attribution (2-speaker demo).
- Auto count → 78–90% (2 speakers), ~60% (3). Identity solid; **counting is the weak link.**

---

### Honest negatives (the short list — full version in `WRITEUP.md` §6)

- **Fear → sadness**, often confidently (family fear recall 47%).
- **Per-speaker gap:** 78.7% vs 63.0% — 2-speaker eval is *indicative*, not population-level.
- **Fine-tuning narrowed retrieval** (WavLM-kNN 52.6% < frozen emotion2vec 63.4%).
- **Weight-tuned hybrid 72.6% is directional only**; the defensible number is 68.6%.
- **Auto speaker-count is unreliable**; supply k when possible.

---

### What's in the box

- `models/wavlm_vad_ft/` — fine-tuned WavLM V/A/D backbone + head.
- `models/adaptors/{wavlm_ft,emotion2vec_plus_large}/` — tagged, non-mixable retrieval DBs.
- One-command hybrid + speaker enrollment:
  `.venv_diar/bin/python -m scripts.adaptors predict --use-case hybrid clip.wav`
- 21 primary-source-verified citations (`REFERENCES.md`); 160 tests passing.

---

*Data: MSP-Podcast v2.0 (264k segments) + 175 own-voice clips. Cite MSP-Podcast:
Busso et al. 2025 (arXiv:2509.09791) + Lotfian & Busso 2019.*
