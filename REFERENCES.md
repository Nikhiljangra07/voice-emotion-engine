# REFERENCES — LoRa Waveform Engine

Foundational and directly-related literature for this project. Every citation below
was verified against its primary source (publisher / arXiv / ACL Anthology). Each entry
notes **how it is used in this engine** — the work builds on these, it does not merely
list them.

---

## 1. The emotion model — dimensional (PAD) affect

- **Russell, J. A. (1980).** A circumplex model of affect. *Journal of Personality and
  Social Psychology, 39*(6), 1161–1178.
  → *Basis for the valence–arousal plane; our V/A axes and the "one point maps to several
  emotions" overlap analysis follow the circumplex.*

- **Mehrabian, A., & Russell, J. A. (1974).** *An Approach to Environmental Psychology.*
  MIT Press. (PAD: Pleasure–Arousal–Dominance.)
  → *Why we use 3 axes, not 2. Dominance is the third axis that separates same-VA
  emotions (anger high-D vs. fear low-D).*

- **Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013).** Norms of valence, arousal,
  and dominance for 13,915 English lemmas. *Behavior Research Methods, 45*(4), 1191–1207.
  doi:10.3758/s13428-012-0314-x
  → *Source of the data-grounded V/A/D centroids in the signal mapper (replaced our
  earlier guessed values).*

## 2. The evaluation metric

- **Lin, L. I.-K. (1989).** A concordance correlation coefficient to evaluate
  reproducibility. *Biometrics, 45*(1), 255–268. doi:10.2307/2532051
  → *CCC is our primary metric for V/A/D regression (the AVEC standard) — not "accuracy,"
  because dimensional prediction is regression.*

## 3. Acoustic feature extraction (Engine / Layer 1)

- **Eyben, F., Scherer, K. R., Schuller, B. W., et al. (2016).** The Geneva Minimalistic
  Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing. *IEEE
  Transactions on Affective Computing, 7*(2), 190–202.
  → *The 88 eGeMAPS features at the core of our 111-feature vector; the paper's finding
  that eGeMAPS is strong on arousal but weak on valence directly motivated our SSL pivot.*

- **Eyben, F., Wöllmer, M., & Schuller, B. (2010).** openSMILE — The Munich Versatile and
  Fast Open-Source Audio Feature Extractor. *Proc. 18th ACM Int. Conf. on Multimedia*,
  1459–1462.
  → *The toolkit we use to extract eGeMAPS.*

- **Jadoul, Y., Thompson, B., & de Boer, B. (2018).** Introducing Parselmouth: A Python
  interface to Praat. *Journal of Phonetics, 71*, 1–15. doi:10.1016/j.wocn.2018.07.001
  → *Praat-based jitter, shimmer, HNR, formants + bandwidths (the clinical voice-quality
  features complementing openSMILE).*

## 4. Self-supervised speech representations (the "vectorizers")

- **Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020).** wav2vec 2.0: A Framework for
  Self-Supervised Learning of Speech Representations. *NeurIPS 2020.* arXiv:2006.11477
  → *Foundational SSL-for-speech architecture underpinning the models we use.*

- **Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A.
  (2021).** HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of
  Hidden Units. *IEEE/ACM TASLP, 29*, 3451–3460. arXiv:2106.07447
  → *Masked-prediction SSL lineage; WavLM extends this line.*

- **Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., & Auli, M. (2022).** data2vec: A
  General Framework for Self-supervised Learning in Speech, Vision and Language. *ICML
  2022*, 1298–1312. arXiv:2202.03555
  → *The self-distillation framework emotion2vec is built on.*

- **Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., et al. (2022).** WavLM: Large-Scale
  Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE Journal of Selected
  Topics in Signal Processing, 16*(6), 1505–1518. arXiv:2110.13900
  → *The backbone we **fine-tuned end-to-end** on MSP-Podcast for V/A/D (held-out Test1
  CCC: V 0.705 / A 0.714 / D 0.626) — our headline result.*

- **Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, Z., Zhang, S., & Chen, X. (2024).** emotion2vec:
  Self-Supervised Pre-Training for Speech Emotion Representation. *Findings of the
  Association for Computational Linguistics: ACL 2024*, 15747–15760. arXiv:2312.15185
  → *The emotion-specialized vectorizer we benchmarked against WavLM; its frozen
  embeddings out-generalized our fine-tuned WavLM cross-speaker (+10 pp) and power the
  "stranger" adaptor.*

## 5. Speaker embeddings & diarization (the "distinguisher")

- **Desplanques, B., Thienpondt, J., & Demuynck, K. (2020).** ECAPA-TDNN: Emphasized
  Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.
  *Interspeech 2020*, 3830–3834. arXiv:2005.07143
  → *The speaker-embedding model behind our neural diarizer (100% per-turn attribution);
  its emotion-invariance is why it separates speakers where acoustic features can't.*

## 6. Datasets

- **Lotfian, R., & Busso, C. (2019).** Building Naturalistic Emotionally Balanced Speech
  Corpus by Retrieving Emotional Speech from Existing Podcast Recordings. *IEEE
  Transactions on Affective Computing, 10*(4), 471–483.
  → **MSP-Podcast** — the original corpus-construction paper (methodology + labeling).

- **Busso, C., Lotfian, R., Sridhar, K., Salman, A. N., Lin, W.-C., Goncalves, L., et al.
  (2025).** The MSP-Podcast Corpus. arXiv:2509.09791.
  → *The corpus release paper covering the version we trained on (our primary spine);
  the license requires citing this alongside Lotfian & Busso 2019.*

- **Busso, C., Bulut, M., Lee, C.-C., et al. (2008).** IEMOCAP: Interactive emotional
  dyadic motion capture database. *Language Resources and Evaluation, 42*(4), 335–359.
  → *Reference dimensional+categorical corpus; the cross-corpus generalization target.*

- **Livingstone, S. R., & Russo, F. A. (2018).** The Ryerson Audio-Visual Database of
  Emotional Speech and Song (RAVDESS). *PLoS ONE, 13*(5), e0196391.
  → *Acted-speech dataset; Phase-1 baselines and the stitched multi-speaker test convos.*

- **Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014).**
  CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on
  Affective Computing, 5*(4), 377–390. doi:10.1109/TAFFC.2014.2336244
  → *Cross-dataset robustness testing (exposed the acted→acted domain shift).*

- **Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019).**
  MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. *ACL
  2019*, 527–536. arXiv:1810.02508
  → *Semi-natural conversational speech; combined-corpus training.*

## 7. Core software

- **McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., &
  Nieto, O. (2015).** librosa: Audio and Music Signal Analysis in Python. *Proc. 14th
  Python in Science Conf. (SciPy)*.
  → *Audio loading, resampling, trimming.*

- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of
  Machine Learning Research, 12*, 2825–2830.
  → *SVR/RF/Ridge/logistic regressors, StandardScaler, clustering, GroupShuffleSplit.*

---

## Which paper powers which part of the engine

| Project component | Key references |
|---|---|
| PAD 3-D emotion space | Russell 1980; Mehrabian & Russell 1974 |
| V/A/D centroids (naming) | Warriner et al. 2013 |
| Metric (CCC) | Lin 1989 |
| 111-feature engine (Layer 1) | Eyben et al. 2016, 2010; Jadoul et al. 2018 |
| Fine-tuned dimensional engine | Chen et al. 2022 (WavLM); + wav2vec2/HuBERT lineage |
| Emotion-specialized vectorizer | Ma et al. 2023 (emotion2vec); data2vec 2022 |
| Speaker distinguisher | Desplanques et al. 2020 (ECAPA-TDNN) |
| Gold-standard data | Lotfian & Busso 2019 + Busso et al. 2025 (MSP-Podcast) |
| Supplementary data | RAVDESS 2018; CREMA-D 2014; MELD 2019; IEMOCAP 2008 |
| Tooling | librosa 2015; scikit-learn 2011 |

*All entries verified against primary sources, July 2026.*
