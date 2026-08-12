"""JUDGE-V2 — a SECOND e2v retrieval namer that knows disgust. V1 untouched.

The frozen judge cannot say "disgust" (FAMILIES hard-codes 6 classes). The
law says never retrain the judge — so we don't. This builds a SEPARATE
reference database:

    v2 = v1's exemplars (copied, unmodified)
       + ~36 DATASET disgust exemplars (RAVDESS code-07 + CREMA-D _DIS_,
         one per actor — never mouth clips: anti-circularity preserved)

stored under models/adaptors_v2/emotion2vec_plus_large/. The backbone is the
same frozen emotion2vec (feature extraction only, no gradients anywhere).
All 1,000+ prior ledger rows remain comparable — they were judged by v1,
which still exists byte-for-byte.

Then re-judges with BOTH namers, side by side:
  * the P5A disgust held-out clips (does v2 certify what v1 couldn't name?)
  * the P5A joy/fear/sadness held-out clips (sanity: no misfire regression)

Run (engine env):  .venv_diar/bin/python tts_steering/judge_v2.py
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from scripts.retrieval_namer import RetrievalNamer  # noqa: E402

V1_DB = ROOT / "models/adaptors/emotion2vec_plus_large/db"
V2_DIR = ROOT / "models/adaptors_v2/emotion2vec_plus_large"
CACHE = HERE / "out/judge_v2_disgust_emb.npz"


def disgust_clips():
    clips = []
    for actor in sorted((ROOT / "data/ravdess").glob("Actor_*")):
        f = sorted(actor.glob("03-01-07-01-01-01-*.wav"))
        if f:
            clips.append((f[0], f"rav_{actor.name}"))
    crema = sorted((ROOT / "data/crema_d/audios").glob("*_DIS_XX.wav"))
    seen = set()
    for f in crema:
        spk = f.name.split("_")[0]
        if spk not in seen:
            seen.add(spk)
            clips.append((f, f"crema_{spk}"))
        if len(seen) >= 12:
            break
    return clips


def main():
    v1 = RetrievalNamer.load(V1_DB)
    labs = list(v1.labels)
    print(f"v1 DB: {v1.size} exemplars, families "
          f"{sorted(set(labs))}")

    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        X, speakers = z["X"], list(z["speakers"])
        print(f"disgust embeddings from cache: {X.shape}")
    else:
        import logging
        logging.disable(logging.WARNING)
        import setuptools  # noqa: F401
        from funasr import AutoModel
        m = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True,
                      disable_pbar=True)
        clips = disgust_clips()
        print(f"embedding {len(clips)} dataset disgust clips ...")
        rows, speakers = [], []
        for f, spk in clips:
            rec = m.generate(str(f), granularity="utterance",
                             extract_embedding=True)
            rows.append(np.asarray(rec[0]["feats"], dtype=np.float32).ravel())
            speakers.append(spk)
        X = np.vstack(rows)
        np.savez(CACHE, X=X, speakers=np.array(speakers))
        print(f"embedded: {X.shape} (cached)")

    v2 = RetrievalNamer()
    v2.add(v1.vectors, list(v1.labels), list(v1.speakers))
    v2.add(X, ["disgust"] * len(X), speakers)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    v2.save(V2_DIR / "db")
    (V2_DIR / "config.json").write_text(json.dumps(
        {"backbone": "emotion2vec_plus_large", "use_case": "stranger_v2",
         "dim": int(v2.vectors.shape[1]), "size": v2.size,
         "note": "v1 + dataset disgust exemplars; v1 untouched"}))
    print(f"v2 DB: {v2.size} exemplars, families {sorted(set(v2.labels))}")
    print(f"saved -> {V2_DIR}\n")

    # side-by-side re-judge
    import logging
    logging.disable(logging.WARNING)
    import setuptools  # noqa: F401
    from funasr import AutoModel
    m = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True,
                  disable_pbar=True)

    def embed(path):
        rec = m.generate(str(path), granularity="utterance",
                         extract_embedding=True)
        return np.asarray(rec[0]["feats"], dtype=np.float32).ravel()

    targets = []
    for sid in ("d02", "d03", "s16", "s03"):
        for dr in (0, 1):
            targets.append(("DISGUST-HELDOUT",
                            HERE / f"out/p5a_disgust/heldout_c0_{sid}_d{dr}.wav"))
    for run, sids in (("p5a_joy", ("h01", "h02")),
                      ("p5a_fear", ("h05", "h06")),
                      ("p5a_sad", ("h03", "h04"))):
        for sid in sids:
            targets.append((f"SANITY-{run}",
                            HERE / f"out/{run}/heldout_c0_{sid}_d0.wav"))

    print(f"{'clip':34s} {'v1 (frozen)':22s} v2 (+disgust)")
    flips = 0
    for group, wav in targets:
        if not wav.exists():
            print(f"{wav.name:34s} MISSING")
            continue
        vec = embed(wav)
        r1 = v1.predict(vec, k=5)
        r2 = v2.predict(vec, k=5)
        f1 = f"{r1['emotion']}@{r1['confidence']:.0%}"
        f2 = f"{r2['emotion']}@{r2['confidence']:.0%}"
        mark = ""
        if r2["emotion"] != r1["emotion"]:
            flips += 1
            mark = " <- FLIP"
        if r2["emotion"] == "disgust":
            mark += " DISGUST"
        print(f"[{group}] {wav.name:22s} {f1:22s} {f2}{mark}")
    print(f"\nverdict flips v1->v2: {flips}/{len(targets)}")
    print("JUDGE_V2_DONE")


if __name__ == "__main__":
    main()
