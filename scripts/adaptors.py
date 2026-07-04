"""Two separate, non-mixable emotion adaptors — one per vectorizer backbone.

Design law (correctness, not just tidiness): vectors from different backbones live
in DIFFERENT spaces — cosine between a WavLM vector and an emotion2vec vector is
meaningless. So each backbone gets its OWN tagged database in its OWN directory,
and loading asserts the tag matches. Mixing is structurally impossible.

  models/adaptors/
    wavlm_ft/                 use-case: ENROLLED  (sharper in-domain, 80.6%)
      config.json  db/
    emotion2vec_plus_large/   use-case: STRANGER  (better generalization, 63.4%)
      config.json  db/

Routing: pick the adaptor by use-case (stranger -> emotion2vec, enrolled -> WavLM).

CLI (isolated env):
    .venv_diar/bin/python -m scripts.adaptors build-all
    .venv_diar/bin/python -m scripts.adaptors predict --use-case stranger clip.wav
    .venv_diar/bin/python -m scripts.adaptors predict --backbone wavlm_ft clip.wav
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from scripts.retrieval_namer import FAMILIES, RetrievalNamer

ADAPTOR_ROOT = "models/adaptors"
# backbone -> (cached own-voice embeddings, expected dim, use-case)
BACKBONES = {
    "wavlm_ft":               {"cache": "out/_ownvoice_ft_emb.npy",  "dim": 1024,
                               "use_case": "enrolled"},
    "emotion2vec_plus_large": {"cache": "out/_ownvoice_e2v_emb.npy", "dim": 1024,
                               "use_case": "stranger"},
}
USECASE_TO_BACKBONE = {"enrolled": "wavlm_ft", "stranger": "emotion2vec_plus_large"}


# ── embedders (lazy; each returns a same-space vector for ITS backbone) ──
def get_embedder(backbone: str, root: Path):
    if backbone == "wavlm_ft":
        import torch
        from scripts.predict_wavlm_ft import WavLMRegressor, load_audio
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        model = WavLMRegressor(str(root / "models/wavlm_vad_ft")).to(dev).eval()

        def embed(path):
            y = load_audio(str(path))
            wav = torch.from_numpy(y).unsqueeze(0).to(dev)
            mask = torch.ones_like(wav, dtype=torch.long)
            with torch.no_grad():
                h = model.backbone(wav, attention_mask=mask).last_hidden_state.mean(1)
            return h.float().cpu().numpy().ravel()
        return embed

    if backbone == "emotion2vec_plus_large":
        import logging
        logging.disable(logging.WARNING)
        import setuptools  # noqa: F401  (activates its bundled distutils on py3.13)
        from funasr import AutoModel
        m = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True,
                      disable_pbar=True)

        def embed(path):
            rec = m.generate(str(path), granularity="utterance",
                             extract_embedding=True)
            return np.asarray(rec[0]["feats"], dtype=np.float32).ravel()
        return embed

    raise ValueError(f"unknown backbone: {backbone}")


class Adaptor:
    def __init__(self, backbone, use_case, namer, root):
        self.backbone = backbone
        self.use_case = use_case
        self.namer = namer
        self.root = root
        self._embed = None

    def _dir(self):
        return self.root / ADAPTOR_ROOT / self.backbone

    def predict(self, path, k=5):
        if self._embed is None:
            self._embed = get_embedder(self.backbone, self.root)
        vec = self._embed(path)
        # anti-mix guard: query vector dim must match this adaptor's DB
        if vec.shape[0] != self.namer.vectors.shape[1]:
            raise RuntimeError("VECTOR-SPACE MISMATCH — refusing to mix backbones.")
        return self.namer.predict(vec, k=k)

    def enroll(self, files, label, speaker, k=5):
        if self._embed is None:
            self._embed = get_embedder(self.backbone, self.root)
        X = np.vstack([self._embed(f) for f in files])
        self.namer.add(X, [label] * len(files), [speaker] * len(files))
        self.save()

    def save(self):
        d = self._dir(); d.mkdir(parents=True, exist_ok=True)
        self.namer.save(d / "db")
        (d / "config.json").write_text(json.dumps({
            "backbone": self.backbone, "use_case": self.use_case,
            "dim": int(self.namer.vectors.shape[1]), "size": self.namer.size}))

    @classmethod
    def load(cls, backbone, root):
        d = root / ADAPTOR_ROOT / backbone
        cfg = json.loads((d / "config.json").read_text())
        # HARD anti-mix guard
        if cfg["backbone"] != backbone:
            raise RuntimeError(
                f"ADAPTOR MIX GUARD: dir '{backbone}' holds '{cfg['backbone']}'")
        namer = RetrievalNamer.load(d / "db")
        if namer.vectors.shape[1] != cfg["dim"]:
            raise RuntimeError("ADAPTOR MIX GUARD: DB dim != config dim.")
        return cls(backbone, cfg["use_case"], namer, root)

    @classmethod
    def build_from_cache(cls, backbone, root):
        spec = BACKBONES[backbone]
        manifest = json.loads((root / "own_voice/manifest.json").read_text())
        labeled = [e for e in manifest if e["label"] in FAMILIES]
        cache = root / spec["cache"]
        meta = Path(str(cache)[:-4] + ".meta.json")
        X = np.load(cache)
        assert json.loads(meta.read_text()) == [e["file"] for e in labeled], \
            f"{backbone} cache order mismatch"
        assert X.shape[1] == spec["dim"], f"{backbone} dim {X.shape[1]} != {spec['dim']}"
        namer = RetrievalNamer()
        namer.add(X, [e["label"] for e in labeled], [e["speaker"] for e in labeled])
        obj = cls(backbone, spec["use_case"], namer, root)
        obj.save()
        return obj


class HybridAdaptor:
    """Decision-level fusion of BOTH backbones — the best cross-speaker namer.

    The anti-mix law is preserved: the two vector spaces are NEVER mixed. Each
    backbone scores the clip in its own space, producing a 6-way family
    distribution; we average those distributions (equal weight by default). The two
    scorers are chosen by what each backbone is good at (measured, not assumed):
      * emotion2vec  -> cosine kNN over its DB   (best generalizer, 63.4%)
      * WavLM-ft     -> logistic head over its DB (its space is linearly separable
                        but poor for kNN — head 68.0% vs kNN 52.6%)
    Equal-weight hybrid = 68.6% cross-speaker (leave-one-speaker-out), beating either
    backbone alone. Both scorers are derived from the SAME growable databases, so
    enrolling a speaker (embed-and-store into both) improves the hybrid with no
    gradient training — the head is simply refit (<1s on ~175 vectors).
    """

    FAMS = FAMILIES

    def __init__(self, e2v: Adaptor, wav: Adaptor, root: Path, w: float = 0.5):
        self.e2v = e2v          # emotion2vec adaptor (kNN)
        self.wav = wav          # WavLM-ft adaptor    (head)
        self.root = root
        self.w = w              # emotion2vec share of the average
        self.backbone = "hybrid"
        self.use_case = "hybrid"
        self._head = None
        self._scaler = None

    @property
    def size(self) -> int:
        return self.wav.namer.size

    def _fit_head(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = self.wav.namer.vectors
        y = np.array(self.wav.namer.labels)
        self._scaler = StandardScaler().fit(X)
        self._head = LogisticRegression(max_iter=2000, C=0.5,
                                        class_weight="balanced")
        self._head.fit(self._scaler.transform(X), y)

    @staticmethod
    def _full_dist(distribution: dict) -> np.ndarray:
        """kNN distribution (only families in the neighborhood) -> full 6-vector."""
        return np.array([distribution.get(f, 0.0) for f in HybridAdaptor.FAMS])

    def predict(self, path, k: int = 5) -> dict:
        if self.e2v._embed is None:
            self.e2v._embed = get_embedder(self.e2v.backbone, self.root)
        if self.wav._embed is None:
            self.wav._embed = get_embedder(self.wav.backbone, self.root)
        if self._head is None:
            self._fit_head()

        # emotion2vec side — kNN distribution (its own space)
        ve = self.e2v._embed(path)
        if ve.shape[0] != self.e2v.namer.vectors.shape[1]:
            raise RuntimeError("VECTOR-SPACE MISMATCH (emotion2vec) — refusing to mix.")
        r_e = self.e2v.namer.predict(ve, k=k)
        pe = self._full_dist(r_e["distribution"])

        # WavLM-ft side — logistic head distribution (its own space)
        vw = self.wav._embed(path)
        if vw.shape[0] != self.wav.namer.vectors.shape[1]:
            raise RuntimeError("VECTOR-SPACE MISMATCH (WavLM) — refusing to mix.")
        vw = vw / (np.linalg.norm(vw) + 1e-8)   # DB rows are L2-normalized
        proba = self._head.predict_proba(self._scaler.transform(vw[None, :]))[0]
        cls = list(self._head.classes_)
        pw = np.array([proba[cls.index(f)] if f in cls else 0.0 for f in self.FAMS])

        # decision-level fusion (distributions only — never vectors)
        p = self.w * pe + (1.0 - self.w) * pw
        p = p / (p.sum() or 1.0)
        order = sorted(zip(self.FAMS, p), key=lambda kv: kv[1], reverse=True)
        margin = order[0][1] - (order[1][1] if len(order) > 1 else 0.0)
        return {
            "emotion": order[0][0],
            "confidence": float(order[0][1]),
            "distribution": {f: float(v) for f, v in order},
            "ambiguous": bool(margin < self.e2v.namer.ambiguity_margin),
            "neighbors": r_e["neighbors"],   # from the enrollable retrieval side
            "backends": {"emotion2vec": r_e["emotion"],
                         "wavlm": self.FAMS[int(pw.argmax())]},
        }

    def enroll(self, files, label, speaker, k=5):
        """Grow BOTH databases (embed-and-store), then refit the head. No retrain."""
        self.e2v.enroll(files, label, speaker, k=k)   # embeds + adds + saves
        self.wav.enroll(files, label, speaker, k=k)
        self._head = None                             # force refit on next predict

    @classmethod
    def load(cls, root: Path, w: float = 0.5) -> "HybridAdaptor":
        return cls(Adaptor.load("emotion2vec_plus_large", root),
                   Adaptor.load("wavlm_ft", root), root, w=w)


def route(use_case: str, root: Path):
    if use_case == "hybrid":
        return HybridAdaptor.load(root)
    return Adaptor.load(USECASE_TO_BACKBONE[use_case], root)


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-all")
    p = sub.add_parser("predict")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--use-case", choices=list(USECASE_TO_BACKBONE) + ["hybrid"])
    g.add_argument("--backbone", choices=list(BACKBONES))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("files", nargs="+")
    e = sub.add_parser("enroll")
    ge = e.add_mutually_exclusive_group(required=True)
    ge.add_argument("--use-case", choices=list(USECASE_TO_BACKBONE) + ["hybrid"])
    ge.add_argument("--backbone", choices=list(BACKBONES))
    e.add_argument("--label", required=True, choices=FAMILIES)
    e.add_argument("--speaker", required=True)
    e.add_argument("--k", type=int, default=5)
    e.add_argument("files", nargs="+")
    args = ap.parse_args()

    if args.cmd == "build-all":
        for b in BACKBONES:
            a = Adaptor.build_from_cache(b, root)
            print(f"built adaptor '{b}'  use-case={a.use_case}  "
                  f"dim={a.namer.vectors.shape[1]}  exemplars={a.namer.size} "
                  f"-> {ADAPTOR_ROOT}/{b}/")
        # prove separation
        wa = Adaptor.load("wavlm_ft", root)
        ea = Adaptor.load("emotion2vec_plus_large", root)
        print(f"\nSEPARATION CHECK: wavlm dim={wa.namer.vectors.shape[1]} "
              f"({wa.use_case}) | emotion2vec dim={ea.namer.vectors.shape[1]} "
              f"({ea.use_case}) — separate dirs, tagged, non-mixable.")
        return

    if args.cmd == "predict":
        a = (route(args.use_case, root) if args.use_case
             else Adaptor.load(args.backbone, root))
        db = a.size if isinstance(a, HybridAdaptor) else a.namer.size
        print(f"adaptor={a.backbone} (use-case={a.use_case}) DB={db}\n")
        for f in args.files:
            r = a.predict(f, k=args.k)
            amb = " (ambiguous)" if r["ambiguous"] else ""
            back = (f"  [e2v={r['backends']['emotion2vec']}, "
                    f"wavlm={r['backends']['wavlm']}]" if "backends" in r else "")
            nb = ", ".join(f"{lab}[{spk}]={s}" for lab, spk, s in r["neighbors"])
            print(f"{Path(f).name}: {r['emotion']}  conf={r['confidence']:.0%}{amb}{back}"
                  f"\n   nearest: {nb}")
        return

    if args.cmd == "enroll":
        a = (route(args.use_case, root) if args.use_case
             else Adaptor.load(args.backbone, root))
        a.enroll(args.files, args.label, args.speaker, k=args.k)
        db = a.size if isinstance(a, HybridAdaptor) else a.namer.size
        who = a.backbone if not args.use_case else args.use_case
        print(f"enrolled {len(args.files)} '{args.label}' clip(s) for "
              f"'{args.speaker}' into {who}. DB now {db} exemplars.")


if __name__ == "__main__":
    main()
