"""Score the PAD-centroid Namer against MSP Test1 ground truth.

The dimensional ear was validated on MSP Test1 (CCC V .705/A .714/D .626)
but the NAMER — the layer that turns V/A/D into a family name — was
never scored against labeled data. This closes that box, and decomposes
the error:

  N1  name from GROUND-TRUTH V/A/D  -> the ceiling of centroid naming
                                       (geometry error only)
  N2  name from WavLM-ft PREDICTED V/A/D -> the number the live ear
                                       actually runs at (full system)
  N3  distance variants on both: current Mahalanobis · V-A only (drop
      D) · D-boosted Mahalanobis (scale the D axis) · Euclidean.
      Data decides whether dominance is underused in the hot corner.

MSP data use: statistics only (license). Audio never leaves the machine,
nothing here trains or feeds the mouth.

Run:  .venv_diar/bin/python scripts/eval_namer_msp.py --infer   (~35 min)
      venv/bin/python scripts/eval_namer_msp.py --analyze
Outputs: out/wavlm_test1_pred.npy (cache), out/namer_eval/*.csv + report
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_DIR = ROOT / "data/msp_podcast/Audios"
META = ROOT / "out/meta_msp_test1.csv"
PRED = ROOT / "out/wavlm_test1_pred.npy"
OUT = ROOT / "out/namer_eval"
NAMER_DIR = ROOT / "models/namer_msp_final"


def load_meta():
    with open(META) as fh:
        return list(csv.DictReader(fh))


def norm_targets(rows):
    """SAM 1-7 -> PAD, same transform as training."""
    t = np.array([[float(r["valence"]), float(r["arousal"]),
                   float(r["dominance"])] for r in rows])
    return np.stack([(t[:, 0] - 4) / 3, (t[:, 1] - 1) / 6,
                     (t[:, 2] - 4) / 3], axis=1)


def infer():
    import torch
    from scripts.predict_wavlm_ft import WavLMRegressor, load_audio

    rows = load_meta()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading WavLM-ft on {device} ...", flush=True)
    model = WavLMRegressor(str(ROOT / "models/wavlm_vad_ft"))
    head_sd = torch.load(ROOT / "models/wavlm_vad_ft/head.pt",
                         map_location="cpu")
    model.head.load_state_dict(head_sd)
    model.to(device).eval()

    preds = np.full((len(rows), 3), np.nan)
    if PRED.exists():  # resume
        cached = np.load(PRED)
        preds[:len(cached)] = cached
    import time
    t0 = time.time()
    for i, r in enumerate(rows):
        if not np.isnan(preds[i]).any():
            continue
        y = load_audio(str(AUDIO_DIR / r["filename"]))
        wav = torch.from_numpy(y).unsqueeze(0).to(device)
        mask = torch.ones_like(wav, dtype=torch.long)
        with torch.no_grad():
            out = model(wav, mask).float().cpu().numpy().ravel()
        raw = np.clip(out, 0.0, 1.0) * 6.0 + 1.0
        preds[i] = [(raw[0] - 4) / 3, (raw[1] - 1) / 6, (raw[2] - 4) / 3]
        if (i + 1) % 250 == 0:
            np.save(PRED, preds)
            el = time.time() - t0
            print(f"{i + 1}/{len(rows)}  {el / (i + 1):.2f}s/file  "
                  f"eta {(len(rows) - i - 1) * el / (i + 1) / 60:.0f} min",
                  flush=True)
    np.save(PRED, preds)
    print(f"saved {PRED}", flush=True)
    print("INFER_DONE", flush=True)


# ── distance variants (N3) ──────────────────────────────────────────

def load_namer_cfg():
    meta = json.loads((NAMER_DIR / "namer.json").read_text())
    labels = meta["labels"]
    cent = {e: np.array(v) for e, v in meta["centroids"].items()}
    inv = {e: np.array(v) for e, v in meta["inv_cov"].items()}
    return labels, cent, inv, meta["ambiguity_margin"]


def name_batch(P, labels, cent, inv, mode="maha", d_scale=1.0, margin=0.15):
    """Vectorized naming. Returns (names, ambiguous) for N points.

    mode: maha (current) · euclid · va (drop D axis)
    d_scale: multiply the D axis before distance (D-boost test); applied
    to point and centroid alike so geometry stretches, not shifts.
    """
    S = np.diag([1.0, 1.0, d_scale])
    D = np.zeros((len(P), len(labels)))
    for j, e in enumerate(labels):
        delta = P - cent[e]
        if mode == "maha":
            # stretch the metric consistently with the axis scaling
            D[:, j] = np.sqrt(np.maximum(np.einsum(
                "ni,ij,nj->n", delta, S @ inv[e] @ S, delta), 0.0))
        elif mode == "euclid":
            D[:, j] = np.linalg.norm(delta @ S, axis=1)
        elif mode == "va":
            D[:, j] = np.linalg.norm(delta[:, :2], axis=1)
    logits = -D
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    probs = w / w.sum(axis=1, keepdims=True)
    top = np.argsort(-probs, axis=1)
    names = [labels[i] for i in top[:, 0]]
    marg = probs[np.arange(len(P)), top[:, 0]] - \
        probs[np.arange(len(P)), top[:, 1]]
    return names, marg < margin


def score(names, truth, labels):
    ok = np.array([n == t for n, t in zip(names, truth)])
    acc = ok.mean()
    per = {}
    for e in labels:
        m = np.array([t == e for t in truth])
        if m.sum():
            per[e] = float(ok[m].mean())
    bal = float(np.mean(list(per.values())))
    return acc, bal, per


def confusion_csv(names, truth, labels, path):
    idx = {e: i for i, e in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, n in zip(truth, names):
        M[idx[t], idx[n]] += 1
    with open(path, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + labels)
        for i, e in enumerate(labels):
            w.writerow([e] + list(M[i]))
    return M


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_meta()
    labels, cent, inv, margin = load_namer_cfg()
    keep = [i for i, r in enumerate(rows) if r["emotion"] in labels]
    truth = [rows[i]["emotion"] for i in keep]
    gt = norm_targets(rows)[keep]
    pred = np.load(PRED)[keep]
    have_pred = not np.isnan(pred).any()
    print(f"Test1 usable: {len(keep)}/{len(rows)} "
          f"(dropped no_agreement/other)")
    print("label counts:", Counter(truth).most_common(), "\n")

    variants = [("maha (current)", dict(mode="maha", d_scale=1.0)),
                ("maha D x1.5", dict(mode="maha", d_scale=1.5)),
                ("maha D x2.0", dict(mode="maha", d_scale=2.0)),
                ("maha D x3.0", dict(mode="maha", d_scale=3.0)),
                ("maha D x0.5", dict(mode="maha", d_scale=0.5)),
                ("euclid", dict(mode="euclid")),
                ("V-A only (no D)", dict(mode="va"))]

    report = []
    for src, P in [("N1 ground-truth PAD", gt)] + \
                  ([("N2 predicted PAD", pred)] if have_pred else []):
        report.append(f"== {src} ==")
        for vn, kw in variants:
            names, amb = name_batch(P, labels, cent, inv,
                                    margin=margin, **kw)
            acc, bal, per = score(names, truth, labels)
            report.append(f"{vn:18s} acc {acc * 100:5.1f}%  "
                          f"balanced {bal * 100:5.1f}%  "
                          f"ambiguous {amb.mean() * 100:4.0f}%")
            if vn == "maha (current)":
                tag = "gt" if src.startswith("N1") else "pred"
                confusion_csv(names, truth, labels,
                              OUT / f"confusion_{tag}.csv")
                report.append("  per-class recall: " + "  ".join(
                    f"{e} {per.get(e, 0) * 100:.0f}%" for e in labels))
        report.append("")
    txt = "\n".join(report)
    print(txt)
    (OUT / "report.txt").write_text(txt + "\n")
    print(f"saved: {OUT}/report.txt + confusion CSVs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.infer:
        infer()
    if a.analyze:
        analyze()
    if not (a.infer or a.analyze):
        print(__doc__)


if __name__ == "__main__":
    main()
