"""P4.8 — SCAFFOLD v2: the higher harmonics (contrast equations).

C1's equation was one harmonic per emotion: direction(family - neutral). It found
the right acoustic REGION (best distances of P4.7) but was valence-blind between
the hot emotions (steered anger with the happy knob). v2 adds the missing
harmonics — pairwise CONTRAST directions (anger-joy, joy-surprise, ...) — so the
score for target f becomes:

    S(clip, f) = cos_w(achieved, dir_f)                     <- the region term
               + lam * min over rivals g of
                     cos_w(achieved, mean_f - mean_g)       <- the edge term

The blend weight `lam` is NOT chosen by hand: it is fit by grid search to
maximize held-out 4-way classification accuracy on the MSP dev split
(no-magic-numbers law). Validation reported before any steering use.

Run:  venv/bin/python tts_steering/scaffold2_msp.py
Out:  tts_steering/out/scaffold_msp_v2.json  (superset of v1's contents)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scaffold_msp import load_split, FAMILIES, EXCLUDE_SUBSTR, TOP_K  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from src.features.feature_vector import feature_names  # noqa: E402

TARGET_FAMS = ["anger", "sadness", "joy", "surprise"]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def main() -> None:
    v1 = json.loads((HERE / "out/scaffold_msp.json").read_text())
    idx = v1["feature_index_in_111"]
    w = np.array(v1["weights"])
    med = np.array(v1["global_median"])
    iqr = np.array(v1["global_iqr"])
    means = {f: np.array(v1["templates_z"][f]) for f in FAMILIES}

    def z_sel(X):
        return (X[:, idx] - med) / iqr

    def wcos(A, b):  # rows of A vs vector b, weighted cosine
        num = (A * w * b).sum(axis=1)
        den = np.sqrt((A * A * w).sum(axis=1) * (b * b * w).sum())
        return np.where(den > 1e-9, num / den, 0.0)

    dirs = {f: means[f] - means["neutral"] for f in TARGET_FAMS}
    contrasts = {f: {g: means[f] - means[g]
                     for g in TARGET_FAMS if g != f} for f in TARGET_FAMS}

    # ---- fit lam on dev: 4-way nearest-by-S classification ----
    Xd, yd = load_split("dev")
    m = np.isin(yd, TARGET_FAMS)
    Zd, yd4 = z_sel(Xd[m]), yd[m]
    ach = Zd - means["neutral"]          # achieved-from-neutral, population ref

    region = {f: wcos(ach, dirs[f]) for f in TARGET_FAMS}
    edge = {f: np.min(np.stack([wcos(ach, contrasts[f][g])
                                for g in contrasts[f]]), axis=0)
            for f in TARGET_FAMS}

    print(f"dev 4-way fit set: {len(yd4)} clips "
          f"({', '.join(f'{f}:{int((yd4==f).sum())}' for f in TARGET_FAMS)})")
    best = None
    for lam in LAM_GRID:
        S = np.stack([region[f] + lam * edge[f] for f in TARGET_FAMS], axis=1)
        pred = np.array(TARGET_FAMS)[S.argmax(axis=1)]
        acc = float((pred == yd4).mean())
        hot = np.isin(yd4, ["anger", "joy", "surprise"])
        Sh = np.stack([region[f] + lam * edge[f]
                       for f in ["anger", "joy", "surprise"]], axis=1)
        hot_pred = np.array(["anger", "joy", "surprise"])[Sh[hot].argmax(axis=1)]
        hot_acc = float((hot_pred == yd4[hot]).mean())
        print(f"  lam={lam:4.2f}  4-way={acc:.1%}  hot-3-way={hot_acc:.1%}")
        if best is None or acc > best[1]:
            best = (lam, acc, hot_acc)
    lam, acc4, acc3 = best
    print(f"\nchosen lam={lam} (dev 4-way {acc4:.1%}, hot-3-way {acc3:.1%})")

    # per-family recall at chosen lam, for the record
    S = np.stack([region[f] + lam * edge[f] for f in TARGET_FAMS], axis=1)
    pred = np.array(TARGET_FAMS)[S.argmax(axis=1)]
    recalls = {f: float((pred[yd4 == f] == f).mean()) for f in TARGET_FAMS}
    for f, r in recalls.items():
        print(f"  {f:9s} recall {r:.1%}")

    out = dict(v1)
    out["v2"] = {
        "lam": lam,
        "lam_grid": LAM_GRID,
        "contrasts_z": {f: {g: c.tolist() for g, c in contrasts[f].items()}
                        for f in TARGET_FAMS},
        "validation": {"dev_4way_acc": acc4, "chance_4way": 0.25,
                       "dev_hot3way_acc": acc3, "chance_3way": 1 / 3,
                       "per_family_recall": recalls,
                       "v1_5way_for_reference": v1["validation"]},
    }
    dest = HERE / "out/scaffold_msp_v2.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nSCAFFOLD_V2_SAVED {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
