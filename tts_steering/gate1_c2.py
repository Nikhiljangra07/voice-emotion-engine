"""P4.8 GATE 1 (free, no synthesis) — does scaffold v2 pick sane knobs?

Pre-registered pass criterion: for each target, hill-climbing the v2 score over
the mouth's measured knob responses must rank an emotion-appropriate knob first
(anger -> angry; surprise -> surprised; joy -> happy; sadness -> sad/melancholic).
C1 failed exactly this: it ranked `happy` first for the anger target.

Uses only cached artifacts: scaffold_msp_v2.json + out/abc_p47/features.json
(calibration clips already extracted in P4.7).

Run:  .venv_tts/bin/python tts_steering/gate1_c2.py
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "out/sweep_p42"
P47 = HERE / "out/abc_p47"

CALIB = {  # clip -> (knob, level)   [same set as P4.7 calibration]
    str(SWEEP / "angry_08.wav"): ("angry", 0.8),
    str(SWEEP / "angry_12.wav"): ("angry", 1.2),
    str(SWEEP / "joy_08.wav"): ("happy", 0.8),
    str(SWEEP / "joy_12.wav"): ("happy", 1.2),
    str(SWEEP / "sad_04.wav"): ("sad", 0.4),
    str(SWEEP / "sad_12.wav"): ("sad", 1.2),
    str(SWEEP / "mel_08.wav"): ("melancholic", 0.8),
    str(SWEEP / "calm_08.wav"): ("calm", 0.8),
    str(P47 / "probe_afraid_08.wav"): ("afraid", 0.8),
    str(P47 / "probe_disgusted_08.wav"): ("disgusted", 0.8),
    str(P47 / "probe_surprised_08.wav"): ("surprised", 0.8),
}
BASELINE = str(SWEEP / "baseline_zero.wav")
TARGETS = ["anger", "sadness", "joy", "surprise"]
SANE = {"anger": {"angry"}, "surprise": {"surprised"},
        "joy": {"happy"}, "sadness": {"sad", "melancholic"}}


import sys

SCAFFOLD_FILE = sys.argv[1] if len(sys.argv) > 1 else "out/scaffold_msp_v2.json"


class V2:
    def __init__(self) -> None:
        s = json.loads((HERE / SCAFFOLD_FILE).read_text())
        self.idx, self.w = s["feature_index_in_111"], s["weights"]
        self.med, self.iqr = s["global_median"], s["global_iqr"]
        self.dir = s["directions_from_neutral_z"]
        self.lam = s["v2"]["lam"]
        self.con = s["v2"]["contrasts_z"]

    def z(self, f111):
        return [(f111[j] - self.med[i]) / self.iqr[i]
                for i, j in enumerate(self.idx)]

    def wdot(self, a, b):
        return sum(w * x * y for w, x, y in zip(self.w, a, b))

    def cos(self, a, b):
        na, nb = math.sqrt(self.wdot(a, a)), math.sqrt(self.wdot(b, b))
        return self.wdot(a, b) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0

    def score(self, achieved, fam):
        region = self.cos(achieved, self.dir[fam])
        edge = min(self.cos(achieved, c) for c in self.con[fam].values())
        return region + self.lam * edge, region, edge


def main() -> None:
    feats = json.loads((P47 / "features.json").read_text())
    v2 = V2()
    base = v2.z(feats[BASELINE])
    effects: dict[str, list] = {}
    for clip, (knob, lvl) in CALIB.items():
        zz = v2.z(feats[clip])
        eff = [(zz[i] - base[i]) / lvl for i in range(len(zz))]
        effects.setdefault(knob, []).append(eff)
    effects = {k: [sum(c) / len(c) for c in zip(*v)] for k, v in effects.items()}

    print("GATE 1 — knob ranking by scaffold-v2 score (per-unit slider effect)\n")
    passed = {}
    for tgt in TARGETS:
        ranked = []
        for k, eff in effects.items():
            s, region, edge = v2.score(eff, tgt)
            ranked.append((s, region, edge, k))
        ranked.sort(reverse=True)
        top = ranked[0][3]
        ok = top in SANE[tgt]
        passed[tgt] = ok
        print(f"  {tgt:8s} -> " + " | ".join(
            f"{k} S={s:+.2f} (region {r:+.2f}, edge {e:+.2f})"
            for s, r, e, k in ranked[:3]))
        print(f"           top knob: {top}  "
              f"{'SANE ✓' if ok else 'INSANE ✗ (v1 failure mode persists)'}\n")

    n_ok = sum(passed.values())
    print(f"GATE 1: {n_ok}/4 targets pick a sane knob "
          f"({'PASS' if n_ok >= 3 else 'FAIL'} — pre-registered bar: 3/4, "
          f"anger mandatory)")
    print("GATE1_" + ("PASS" if n_ok >= 3 and passed["anger"] else "FAIL"))


if __name__ == "__main__":
    main()
