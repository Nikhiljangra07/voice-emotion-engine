"""P4.8 — SCAFFOLD-R: same equations, PROMINENT foundation (RAVDESS acted speech).

Gate 1 failed on the MSP foundation: natural conversational speech carries the
hot emotions too subtly — their recipes are nearly collinear in classical
features, so the compass confuses them (user's hypothesis: "their emotional
wavelength is much higher... we need somewhere more prominent for emotional
calibration"). RAVDESS is that somewhere: professional actors, validated labels,
deliberately exaggerated delivery — and structurally closer to what the mouth
IS (a performer, not a conversationalist).

Same z-space as v1/v2 (MSP robust scaler, same 32 Fisher-selected dims) so the
knob-effect calibration carries over unchanged; only the direction/contrast
vectors are refit from RAVDESS. Neutral anchor = 288 RAVDESS neutral+calm clips
(same recording channel as the family clips — no channel leak into directions).

Run:  venv/bin/python tts_steering/scaffold_ravdess.py
Out:  tts_steering/out/scaffold_ravdess.json
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET_FAMS = ["anger", "sadness", "joy", "surprise"]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def main() -> None:
    v1 = json.loads((HERE / "out/scaffold_msp.json").read_text())
    idx = v1["feature_index_in_111"]
    w = np.array(v1["weights"])
    med, iqr = np.array(v1["global_median"]), np.array(v1["global_iqr"])

    def z_sel(X):
        return (X[:, idx] - med) / iqr

    def wcos(A, b):
        num = (A * w * b).sum(axis=1)
        den = np.sqrt((A * A * w).sum(axis=1) * (b * b * w).sum())
        return np.where(den > 1e-9, num / den, 0.0)

    X = np.load(ROOT / "out/features_ravdess.npy")
    y = np.load(ROOT / "out/labels_ravdess.npy", allow_pickle=True).astype(str)
    Xn = np.load(ROOT / "out/features_ravdess_neutral288.npy")
    Z, Zn = z_sel(X), z_sel(Xn)
    neutral = Zn.mean(axis=0)
    means = {f: Z[y == f].mean(axis=0) for f in TARGET_FAMS}

    dirs = {f: means[f] - neutral for f in TARGET_FAMS}
    contrasts = {f: {g: means[f] - means[g]
                     for g in TARGET_FAMS if g != f} for f in TARGET_FAMS}

    # ---- validation 1: RAVDESS itself (resubstitution — optimistic, labeled so)
    m = np.isin(y, TARGET_FAMS)
    ach = Z[m] - neutral
    y4 = y[m]
    region = {f: wcos(ach, dirs[f]) for f in TARGET_FAMS}
    edge = {f: np.min(np.stack([wcos(ach, contrasts[f][g])
                                for g in contrasts[f]]), axis=0)
            for f in TARGET_FAMS}
    print(f"RAVDESS fit set: {len(y4)} clips (4x192), neutral anchor n=288")
    best = None
    for lam in LAM_GRID:
        S = np.stack([region[f] + lam * edge[f] for f in TARGET_FAMS], axis=1)
        pred = np.array(TARGET_FAMS)[S.argmax(axis=1)]
        acc = float((pred == y4).mean())
        hot = np.isin(y4, ["anger", "joy", "surprise"])
        Sh = np.stack([region[f] + lam * edge[f]
                       for f in ["anger", "joy", "surprise"]], axis=1)
        hp = np.array(["anger", "joy", "surprise"])[Sh[hot].argmax(axis=1)]
        hacc = float((hp == y4[hot]).mean())
        print(f"  lam={lam:4.2f}  4-way={acc:.1%}  hot-3-way={hacc:.1%}  "
              f"(resubstitution)")
        if best is None or acc > best[1]:
            best = (lam, acc, hacc)
    lam, acc4, acc3 = best
    S = np.stack([region[f] + lam * edge[f] for f in TARGET_FAMS], axis=1)
    pred = np.array(TARGET_FAMS)[S.argmax(axis=1)]
    recalls = {f: float((pred[y4 == f] == f).mean()) for f in TARGET_FAMS}
    print(f"\nchosen lam={lam}  per-family recall: " +
          ", ".join(f"{f} {r:.0%}" for f, r in recalls.items()))

    # ---- validation 2: transfer to MSP dev (acted -> natural, honest check)
    import sys
    sys.path.insert(0, str(HERE))
    from scaffold_msp import load_split  # noqa: E402
    Xd, yd = load_split("dev")
    md = np.isin(yd, TARGET_FAMS)
    achd = z_sel(Xd[md]) - neutral
    yd4 = yd[md]
    Sd = np.stack([wcos(achd, dirs[f]) + lam * np.min(
        np.stack([wcos(achd, contrasts[f][g]) for g in contrasts[f]]), axis=0)
        for f in TARGET_FAMS], axis=1)
    predd = np.array(TARGET_FAMS)[Sd.argmax(axis=1)]
    accd = float((predd == yd4).mean())
    print(f"transfer to MSP dev (acted->natural): 4-way {accd:.1%} "
          f"(chance 25%) — expected weak, the mouth is a performer not a "
          f"conversationalist")

    out = {
        "provenance": {
            "foundation": "RAVDESS acted (1,152 Ekman + 288 neutral/calm anchor)",
            "z_space": "MSP robust scaler + 32 Fisher dims from scaffold v1 "
                       "(kept so knob calibration carries over)",
            "lam": lam,
        },
        "feature_index_in_111": idx,
        "weights": w.tolist(),
        "global_median": med.tolist(),
        "global_iqr": iqr.tolist(),
        "directions_from_neutral_z": {f: d.tolist() for f, d in dirs.items()},
        "v2": {"lam": lam,
               "contrasts_z": {f: {g: c.tolist() for g, c in contrasts[f].items()}
                               for f in TARGET_FAMS}},
        "validation": {
            "ravdess_resub_4way": acc4, "ravdess_resub_hot3way": acc3,
            "per_family_recall": recalls, "msp_dev_transfer_4way": accd,
        },
    }
    dest = HERE / "out/scaffold_ravdess.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nSCAFFOLD_R_SAVED {dest}")


if __name__ == "__main__":
    main()
