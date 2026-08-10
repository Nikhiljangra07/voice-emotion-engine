"""P4.9c — THE EXPANDED VIEW: Fourier fingerprints across three human corpora.

User's directive: don't judge the fingerprint equations on one corpus — expand
to CREMA-D and MSP-Podcast audio, then work from the results. The mathematical
reason expansion matters: averaging emotion directions over many speakers,
sentences and corpora cancels text-specific and generic-expressivity confounds
(the two walls of P4.9b); what survives the average is the emotion-specific
component.

Corpora (all local, statistics only):
  RAVDESS  acted, 2 fixed sentences, 24 actors  (anger/sadness/joy/surprise + neutral)
  CREMA-D  acted, 12 sentences, 91 actors       (anger/sadness/joy + neutral)
  MSP      natural conversation, ~10k speakers  (sampled 1000/family, seed 42)

Analyses:
  1. One canonical z-space (pooled mu/sd over all human clips); per-corpus
     neutral anchors remove channel/style offsets; directions are within-corpus.
  2. UNIVERSALITY: same-family direction similarity across corpora (W_deriv).
  3. Cross-corpus transfer matrix (fit on X, classify Y, common families).
  4. Pooled directions (unit-normalized per corpus, averaged).
  5. Our judged clips (P4.7 report) scored against pooled human directions.
  6. PRE-REGISTERED FINAL GATE (declared now, before the run): matched-text
     knob effects (kids_* calibration clips) ranked by CONTRAST scoring with
     POOLED directions under W_deriv. PASS = >=3/4 targets rank a sane knob
     first, anger mandatory. One shot. No tuning after seeing the result.

Run:  venv/bin/python tts_steering/fourier_expand.py
"""

import csv
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from fourier_gate import WEIGHTS, fingerprint, wcos  # noqa: E402

OUT = HERE / "out/fourier_expand"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ["anger", "sadness", "joy", "surprise"]
DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
SANE = {"anger": {"angry"}, "surprise": {"surprised"},
        "joy": {"happy"}, "sadness": {"sad", "melancholic"}}
W = WEIGHTS["deriv"]
MSP_PER_FAM = 1000
CREMA_MAP = {"ANG": "anger", "SAD": "sadness", "HAP": "joy", "NEU": "neutral"}


def extract_corpus(name: str, items: list) -> tuple:
    """items: (path, label). Cached to npz. Returns (X, y, n_failed)."""
    cache = OUT / f"fp_{name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"].astype(str), int(d["failed"])
    fps, labs, failed = [], [], 0
    for i, (p, lab) in enumerate(items):
        fp = fingerprint(str(p))
        if fp is None:
            failed += 1
            continue
        fps.append(fp); labs.append(lab)
        if (i + 1) % 500 == 0:
            print(f"  {name}: {i+1}/{len(items)}", flush=True)
    X, y = np.stack(fps), np.array(labs)
    np.savez(cache, X=X, y=y, failed=failed)
    return X, y, failed


def main() -> None:
    # ---------------- gather corpora ----------------
    from src.utils.dataset_loader import load_ravdess
    rav = [(str(s.path), s.label) for s in load_ravdess(ROOT / "data/ravdess")
           if s.label in TARGETS]
    rav += [(p, "neutral") for p in
            sorted(glob.glob(str(ROOT / "data/ravdess/Actor_*/*.wav")))
            if p.split("/")[-1].split("-")[2] in ("01", "02")]

    crema = []
    for p in sorted(glob.glob(str(ROOT / "data/crema_d/audios/*.wav"))):
        code = Path(p).name.split("_")[2]
        if code in CREMA_MAP:
            crema.append((p, CREMA_MAP[code]))

    rng = random.Random(42)
    by_fam: dict = {}
    with open(ROOT / "out/meta_msp_train.csv") as f:
        for r in csv.DictReader(f):
            if r["emotion"] in TARGETS + ["neutral"]:
                by_fam.setdefault(r["emotion"], []).append(r["filename"])
    msp = []
    for fam, files in by_fam.items():
        for fn in rng.sample(files, min(MSP_PER_FAM, len(files))):
            msp.append((str(ROOT / "data/msp_podcast/Audios" / fn), fam))

    corpora = {}
    for name, items in [("ravdess", rav), ("crema", crema), ("msp", msp)]:
        print(f"extracting {name}: {len(items)} clips ...", flush=True)
        X, y, failed = extract_corpus(name, items)
        corpora[name] = (X, y)
        print(f"  {name}: {len(y)} ok, {failed} failed "
              f"({dict(zip(*np.unique(y, return_counts=True)))})", flush=True)

    # ---------------- canonical z-space ----------------
    ALL = np.vstack([X for X, _ in corpora.values()])
    mu, sd = ALL.mean(axis=0), ALL.std(axis=0)
    sd[sd < 1e-9] = 1e-9

    dirs, cons = {}, {}
    for name, (X, y) in corpora.items():
        Z = (X - mu) / sd
        anchor = Z[y == "neutral"].mean(axis=0)
        fams = {f: Z[y == f].mean(axis=0) for f in TARGETS if (y == f).any()}
        dirs[name] = {f: m - anchor for f, m in fams.items()}
        cons[name] = {f: {g: fams[f] - fams[g] for g in fams if g != f}
                      for f in fams}

    # ---------------- 2. universality of directions ----------------
    print("\nUNIVERSALITY — same-family direction similarity across corpora "
          "(W_deriv):")
    names = list(corpora)
    for f in TARGETS:
        row = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if f in dirs[a] and f in dirs[b]:
                    row.append(f"{a[:3]}~{b[:3]} {wcos(dirs[a][f], dirs[b][f], W):+.2f}")
        print(f"  {f:8s} " + "   ".join(row))

    # ---------------- 3. cross-corpus transfer matrix ----------------
    print("\nTRANSFER — fit on X, classify Y (nearest direction, W_deriv, "
          "common families):")
    for a in names:
        for b in names:
            if a == b:
                continue
            common = [f for f in TARGETS if f in dirs[a] and f in dirs[b]]
            Xb, yb = corpora[b]
            Zb = (Xb - mu) / sd
            anchor_b = Zb[yb == "neutral"].mean(axis=0)
            m = np.isin(yb, common)
            ach = Zb[m] - anchor_b
            S = np.stack([[wcos(v, dirs[a][f], W) for f in common]
                          for v in ach])
            pred = np.array(common)[S.argmax(axis=1)]
            acc = float((pred == yb[m]).mean())
            print(f"  {a:8s} -> {b:8s} {len(common)}-way {acc:.1%} "
                  f"(chance {1/len(common):.0%})")

    # ---------------- 4. pooled directions + contrasts ----------------
    def unit(v):
        n = np.sqrt(float((W * v * v).sum()))
        return v / n if n > 1e-9 else v

    pooled_dir = {f: np.mean([unit(dirs[n][f]) for n in names
                              if f in dirs[n]], axis=0) for f in TARGETS}
    pooled_con = {f: {g: np.mean([unit(cons[n][f][g]) for n in names
                                  if f in cons[n] and g in cons[n][f]], axis=0)
                      for g in TARGETS if g != f} for f in TARGETS}

    # ---------------- 5. our judged clips vs pooled human directions --------
    print("\nOUR JUDGED CLIPS (P4.7) vs pooled human directions (W_deriv):")
    rep = json.loads((HERE / "out/abc_p47/report.json").read_text())
    base_fp = fingerprint(str(HERE / "out/sweep_p42/baseline_zero.wav"))
    base_z = (base_fp - mu) / sd
    for key, r in rep.items():
        tag, tgt = key.split("|")
        for c in r["clips"]:
            if c["family"] == tgt or (tgt == "sadness"
                                      and c["distance"] < 0.21):
                fp = fingerprint(c["wav"])
                if fp is None:
                    continue
                ach = (fp - mu) / sd - base_z
                scores = {f: wcos(ach, pooled_dir[f], W) for f in TARGETS}
                best = max(scores, key=scores.get)
                mark = "✓" if best == tgt else f"✗({best})"
                print(f"  {Path(c['wav']).name:26s} judge={c['family']:9s} "
                      + " ".join(f"{f[:4]}={scores[f]:+.2f}" for f in TARGETS)
                      + f"  fingerprint-best: {mark}")

    # ---------------- 6. THE PRE-REGISTERED GATE ----------------
    print("\nFINAL GATE (pre-registered: pooled directions, contrast scoring, "
          "W_deriv, matched-text effects; >=3/4 sane, anger mandatory):\n")
    calib = HERE / "out/fourier_calib"
    kb = fingerprint(str(calib / "kids_baseline.wav"))
    kb_z = (kb - mu) / sd
    eff = {}
    for k in DIMS:
        fp = fingerprint(str(calib / f"kids_{k}_08.wav"))
        eff[k] = ((fp - mu) / sd - kb_z) / 0.8
    passed = {}
    for tgt in TARGETS:
        ranked = sorted(
            ((min(wcos(eff[k], c, W) for c in pooled_con[tgt].values()), k)
             for k in DIMS), reverse=True)
        top = ranked[0][1]
        ok = top in SANE[tgt]
        passed[tgt] = ok
        print(f"  {tgt:8s} -> " + " | ".join(
            f"{k} {s:+.2f}" for s, k in ranked[:4])
            + f"   top: {top} {'✓' if ok else '✗'}")
    n_ok = sum(passed.values())
    verdict = n_ok >= 3 and passed["anger"]
    print(f"\n  -> {n_ok}/4 sane (bar 3/4, anger mandatory): "
          f"{'PASS' if verdict else 'FAIL'}")
    print("EXPAND_GATE_" + ("PASS" if verdict else "FAIL"))


if __name__ == "__main__":
    main()
