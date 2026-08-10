"""P4.7 — THE A/B/C TEST: three proposers, one mouth, one frozen judge.

  ARM A  indextts2-abcA-det       the incumbent deterministic rules (P4.3 logic)
  ARM B  indextts2-abcB-r1        DeepSeek R1 reasons over the arm's own history
  ARM C  indextts2-abcC-scaffold  the MSP equation scaffold ("training wheels"):
                                  steer toward the certified acoustic RECIPE of
                                  each emotion, fit from 169k real MSP clips

Design laws (pre-registered before any result):
  * Equal budgets: 3 rounds x 2 candidates = 6 scoreable clips per target per arm.
    Stop-on-HIT (the incumbent's convention) — attempts-to-hit is part of the score.
  * Each arm sees ONLY its own history. No cross-contamination.
  * Internal steering compass differs by design (A/B: V/A/D distance; C: scaffold
    match). FINAL scoring is identical for all: the frozen e2v judge names the
    family (HIT), WavLM V/A/D distance ranks quality. The judge is never retrained.
  * Every clip -> ledger row, misses kept, rows appended per round (outage law).
  * Arm C extras, disclosed: reuses P4.2 single-knob clips + 3 new probe clips
    (afraid/disgusted/surprised @0.8) as instrument calibration (knob->feature
    response map). Probes are unscored. The neutral reference is baseline_zero.wav.
  * Targets: anger, sadness, joy, surprise. Surprise centroid (+0.05,0.64,+0.26)
    derived by the SAME label mapping as the original three (reproduced exactly).
  * Fixed sentence S1, fixed neutral voice prompt (like-for-like limitation, noted).
  * R1 spend hard-capped at $2.50 (balance $3.60). R1 parse failure = honest
    forfeit for that round, never a silent fallback to rules.

Run:  .venv_tts/bin/python tts_steering/abc_p47.py
Resumable: wavs, judged results, features and R1 proposals are all cached.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402
import r1_propose  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "out/abc_p47"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"
SWEEP = HERE / "out/sweep_p42"
ENGINE_PY = ROOT / "venv/bin/python"          # classical-feature env (opensmile)

TEXT = "The table is in the room, and the door is closed."
PROMPT = str(ROOT / "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav")
DIMS = ["happy", "angry", "sad", "afraid", "disgusted",
        "melancholic", "surprised", "calm"]
CALM = DIMS.index("calm")

CENTROIDS = {
    "sadness":  (-0.28, 0.39, -0.07),
    "joy":      (+0.30, 0.58, +0.19),
    "anger":    (-0.42, 0.70, +0.41),
    "surprise": (+0.05, 0.64, +0.26),   # same MSP-label mapping, n=3220
}
TARGETS = list(CENTROIDS)
ROUNDS, CANDS = 3, 2

ARMS = {"A": "indextts2-abcA-det",
        "B": "indextts2-abcB-r1",
        "C": "indextts2-abcC-scaffold"}

# P4.2 single-knob clips reused for calibration: file -> (knob, level)
CALIB_EXISTING = {
    "angry_08.wav": ("angry", 0.8),  "angry_12.wav": ("angry", 1.2),
    "joy_08.wav":   ("happy", 0.8),  "joy_12.wav":   ("happy", 1.2),
    "sad_04.wav":   ("sad", 0.4),    "sad_12.wav":   ("sad", 1.2),
    "mel_08.wav":   ("melancholic", 0.8),
    "calm_08.wav":  ("calm", 0.8),
}
CALIB_NEW = {"probe_afraid_08.wav": ("afraid", 0.8),
             "probe_disgusted_08.wav": ("disgusted", 0.8),
             "probe_surprised_08.wav": ("surprised", 0.8)}
BASELINE = SWEEP / "baseline_zero.wav"

SEEDS_A = {  # the incumbent's warm starts (P4.3), trimmed to the 2-cand budget
    "anger":    [{"angry": 0.7}, {"angry": 0.8, "calm": 0.2}],
    "sadness":  [{"melancholic": 1.0}, {"melancholic": 1.2}],
    "joy":      [{"happy": 0.4}, {"happy": 0.4, "calm": 0.4}],
    "surprise": [{"surprised": 0.8}, {"surprised": 1.2}],
}


def vec(d: dict) -> list[float]:
    return [round(float(d.get(k, 0.0)), 2) for k in DIMS]


def clamp(v: list[float]) -> list[float]:
    v = [min(max(x, 0.0), 1.4) for x in v]
    s = sum(v)
    if s > 1.5:
        v = [x * 1.5 / s for x in v]
    return [round(x, 2) for x in v]


def dist(m: dict, target: str) -> float:
    c = CENTROIDS[target]
    return math.sqrt((m["V"] - c[0]) ** 2 + (m["A"] - c[1]) ** 2
                     + (m["D"] - c[2]) ** 2)


def synthesize(jobs: list[dict]) -> None:
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if not todo:
        return
    jf = OUT / "_jobs.json"
    jf.write_text(json.dumps(todo))
    print(f"  synthesizing {len(todo)} clips ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [str(VENDOR / ".venv/bin/python"), str(HERE / "synth_worker.py"), str(jf)],
        cwd=str(VENDOR), env={"PYTHONPATH": str(VENDOR), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=7200)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"synth worker failed:\n{proc.stderr[-600:]}")
    print(f"  ... done in {time.time()-t0:.0f}s", flush=True)


def extract111(wavs: list[str], cache: dict) -> dict:
    """Classical features via the engine venv; cached in features.json."""
    todo = [w for w in wavs if w not in cache]
    if todo:
        tmp = OUT / "_feat.json"
        proc = subprocess.run(
            [str(ENGINE_PY), str(HERE / "extract111.py"), str(tmp)] + todo,
            capture_output=True, text=True, timeout=1800)
        if "EXTRACT111_DONE" not in proc.stdout:
            raise RuntimeError(f"extract111 failed:\n{proc.stderr[-600:]}")
        cache.update(json.loads(tmp.read_text()))
        (OUT / "features.json").write_text(json.dumps(cache))
    return cache


# ----------------------------------------------------------------- the scaffold
class Scaffold:
    def __init__(self) -> None:
        s = json.loads((HERE / "out/scaffold_msp.json").read_text())
        self.idx = s["feature_index_in_111"]
        self.w = s["weights"]
        self.med = s["global_median"]
        self.iqr = s["global_iqr"]
        self.dir = {f: d for f, d in s["directions_from_neutral_z"].items()}
        self.base_z: list[float] | None = None  # set after baseline extraction

    def z(self, feat111: list[float]) -> list[float]:
        return [(feat111[j] - self.med[i]) / self.iqr[i]
                for i, j in enumerate(self.idx)]

    def achieved(self, feat111: list[float]) -> list[float]:
        assert self.base_z is not None
        zz = self.z(feat111)
        return [zz[i] - self.base_z[i] for i in range(len(zz))]

    def _wdot(self, a, b) -> float:
        return sum(w * x * y for w, x, y in zip(self.w, a, b))

    def cos(self, a, b) -> float:
        na, nb = math.sqrt(self._wdot(a, a)), math.sqrt(self._wdot(b, b))
        return self._wdot(a, b) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0

    def score(self, feat111: list[float], family: str) -> float:
        """Weighted cosine between achieved delta and the certified direction."""
        return self.cos(self.achieved(feat111), self.dir[family])


def build_knob_effects(scaf: Scaffold, feats: dict) -> dict[str, list[float]]:
    """Per-unit-slider feature response, averaged over available levels."""
    acc: dict[str, list] = {}
    base = scaf.z(feats[str(BASELINE)])
    scaf.base_z = base
    for fname, (knob, level) in {**{str(SWEEP / k): v for k, v in
                                    CALIB_EXISTING.items()},
                                 **{str(OUT / k): v for k, v in
                                    CALIB_NEW.items()}}.items():
        zz = scaf.z(feats[fname])
        eff = [(zz[i] - base[i]) / level for i in range(len(zz))]
        acc.setdefault(knob, []).append(eff)
    return {k: [sum(col) / len(col) for col in zip(*effs)]
            for k, effs in acc.items()}


# ----------------------------------------------------------------- arm proposers
def propose_A(target: str, st: dict) -> list[list[float]]:
    if not st["history"]:
        return [clamp(vec(d)) for d in SEEDS_A[target]][:CANDS]
    best = min(st["history"], key=lambda h: h["distance"])
    bv, c = best["vector"], CENTROIDS[target]
    cands = []
    dom = max(range(8), key=lambda i: bv[i])
    for scale in (1.25, 0.75):
        w = list(bv); w[dom] = round(w[dom] * scale, 2)
        cands.append(clamp(w))
    a_err = best["A"] - c[1]
    if a_err > 0.08:
        w = list(bv); w[CALM] = round(min(w[CALM] + 0.3, 0.6), 2)
        cands.append(clamp(w))
    elif a_err < -0.08 and bv[CALM] > 0:
        w = list(bv); w[CALM] = round(max(w[CALM] - 0.3, 0.0), 2)
        cands.append(clamp(w))
    fresh = [w for w in cands if tuple(w) not in st["tried"]]
    return fresh[:CANDS]


def propose_B(target: str, st: dict) -> list[list[float]]:
    hist = [{"vector": h["vector"], "V": h["V"], "A": h["A"], "D": h["D"],
             "distance": h["distance"], "judge_family": h["family"],
             "judge_confidence": h["conf"]} for h in st["history"]]
    vecs = r1_propose.propose(target, CENTROIDS[target], hist, CANDS)
    return [w for w in vecs if tuple(w) not in st["tried"]][:CANDS]


def propose_C(target: str, st: dict, scaf: Scaffold,
              effects: dict) -> list[list[float]]:
    fam = target
    direction = scaf.dir[fam]
    if not st["history"]:
        ranked = sorted(effects, key=lambda k: -scaf.cos(effects[k], direction))
        k1, k2 = ranked[0], ranked[1]
        mag_d = math.sqrt(scaf._wdot(direction, direction))
        mag_e = math.sqrt(scaf._wdot(effects[k1], effects[k1]))
        lvl = min(max(round(mag_d / max(mag_e, 1e-6), 1), 0.4), 1.2)
        seeds = [clamp(vec({k1: lvl})), clamp(vec({k1: 0.6, k2: 0.6}))]
        print(f"    [C] {target}: derived seeds — best-aligned knobs "
              f"{k1} (cos={scaf.cos(effects[k1], direction):+.2f}, lvl={lvl}) "
              f"+ pair with {k2}")
        return seeds[:CANDS]
    best = max(st["history"], key=lambda h: h["scaffold"])
    bv = best["vector"]
    achieved = scaf.achieved(best["feat"])
    residual = [direction[i] - achieved[i] for i in range(len(direction))]
    cands = []
    ranked = sorted(effects, key=lambda k: -abs(scaf.cos(effects[k], residual)))
    for k in ranked[:2]:
        eff = effects[k]
        coef = scaf._wdot(residual, eff) / max(scaf._wdot(eff, eff), 1e-9)
        coef = min(max(coef, -0.6), 0.6)
        w = list(bv); ki = DIMS.index(k)
        w[ki] = round(w[ki] + coef, 2)
        cands.append(clamp(w))
    fresh = [w for w in cands if tuple(w) not in st["tried"]]
    return fresh[:CANDS]


# ------------------------------------------------------------------------ main
def main() -> None:
    feats: dict = (json.loads((OUT / "features.json").read_text())
                   if (OUT / "features.json").exists() else {})
    props: dict = (json.loads((OUT / "proposals.json").read_text())
                   if (OUT / "proposals.json").exists() else {})
    results: dict = (json.loads((OUT / "results.json").read_text())
                     if (OUT / "results.json").exists() else {})

    print("=" * 74 + "\nPHASE 0 — CALIBRATION (disclosed, unscored)\n" + "=" * 74)
    synthesize([{"prompt": PROMPT, "text": TEXT,
                 "vector": vec({k: lvl}), "out": str(OUT / f)}
                for f, (k, lvl) in CALIB_NEW.items()])
    calib_wavs = [str(BASELINE)] + \
        [str(SWEEP / k) for k in CALIB_EXISTING] + \
        [str(OUT / k) for k in CALIB_NEW]
    for wv in calib_wavs:
        assert Path(wv).exists(), f"missing calibration clip: {wv}"
    extract111(calib_wavs, feats)
    scaf = Scaffold()
    effects = build_knob_effects(scaf, feats)
    print(f"  knob->feature response map built from {len(calib_wavs)} clips "
          f"({len(CALIB_NEW)} new probes). Knobs: {sorted(effects)}")
    for tgt in TARGETS:
        ranked = sorted(effects, key=lambda k: -scaf.cos(effects[k],
                                                         scaf.dir[tgt]))
        top = ", ".join(f"{k} {scaf.cos(effects[k], scaf.dir[tgt]):+.2f}"
                        for k in ranked[:3])
        print(f"    {tgt:8s} best-aligned knobs: {top}")
    if "--calib-only" in sys.argv:
        print("\nCALIBRATION_ONLY_DONE")
        return

    state = {a: {t: {"history": [], "tried": set(), "hit": None}
                 for t in TARGETS} for a in ARMS}
    # rebuild state from cached results (resume)
    ledgered = set(json.loads((OUT / "ledgered.json").read_text())) \
        if (OUT / "ledgered.json").exists() else set()

    for rnd in range(1, ROUNDS + 1):
        print("=" * 74 + f"\nROUND {rnd}\n" + "=" * 74, flush=True)
        jobs, meta = [], []
        for arm in ARMS:
            for tgt in TARGETS:
                st = state[arm][tgt]
                if st["hit"]:
                    continue
                pk = f"{arm}|{tgt}|{rnd}"
                if pk in props:
                    cands = props[pk]
                else:
                    if arm == "A":
                        cands = propose_A(tgt, st)
                    elif arm == "B":
                        cands = propose_B(tgt, st)
                    else:
                        cands = propose_C(tgt, st, scaf, effects)
                    props[pk] = cands
                    (OUT / "proposals.json").write_text(json.dumps(props))
                for i, v in enumerate(cands):
                    st["tried"].add(tuple(v))
                    wav = OUT / f"abc{arm}_{tgt}_r{rnd}_{i}.wav"
                    jobs.append({"prompt": PROMPT, "text": TEXT, "vector": v,
                                 "out": str(wav)})
                    meta.append((arm, tgt, v, str(wav)))
        if not meta:
            print("  all arms satisfied or exhausted — stopping early")
            break

        synthesize(jobs)
        new_wavs = [w for _, _, _, w in meta if w not in results]
        if new_wavs:
            print(f"  judging {len(new_wavs)} clips (frozen bridge) ...", flush=True)
            for wav, m in zip(new_wavs, judge(new_wavs)):  # order-preserving
                results[wav] = {"V": m["V"], "A": m["A"], "D": m["D"],
                                "family": m["judge_family"],
                                "conf": m["judge_confidence"]}
            (OUT / "results.json").write_text(json.dumps(results))
        extract111([w for _, _, _, w in meta], feats)

        rows = []
        n = sum(1 for _ in open(LEDGER)) - 1
        for arm, tgt, v, wav in meta:
            m = results[wav]
            d = dist(m, tgt)
            sc = scaf.score(feats[wav], tgt)
            hit = m["family"] == tgt
            h = {"vector": v, "V": m["V"], "A": m["A"], "D": m["D"],
                 "distance": d, "family": m["family"], "conf": m["conf"],
                 "scaffold": sc, "feat": feats[wav], "wav": wav, "round": rnd}
            st = state[arm][tgt]
            st["history"].append(h)
            if hit and not st["hit"]:
                st["hit"] = h
            flag = "HIT ✓" if hit else "miss"
            print(f"  [{arm}] {tgt:8s} {v} -> d={d:.3f} scaffold={sc:+.2f} "
                  f"judge={m['family']}@{m['conf']:.0%} {flag}")
            if wav not in ledgered:
                n += 1
                rows.append([n, ARMS[arm], tgt,
                             f"emo_vector={v};spk=ravdess_neutral_A01;p47_r{rnd}",
                             round(m["V"], 3), round(m["A"], 3), round(m["D"], 3),
                             m["family"], round(m["conf"], 2), round(d, 3),
                             int(hit)])
                ledgered.add(wav)
        if rows:
            with open(LEDGER, "a", newline="") as f:
                csv.writer(f).writerows(rows)
            (OUT / "ledgered.json").write_text(json.dumps(sorted(ledgered)))
            print(f"  ledger: +{len(rows)} rows (total {n})")

    print("=" * 74 + "\nSCOREBOARD (frozen judge is the only law)\n" + "=" * 74)
    report = {}
    for tgt in TARGETS:
        print(f"\n  {tgt.upper()}  (centroid {CENTROIDS[tgt]})")
        for arm, tag in ARMS.items():
            st = state[arm][tgt]
            if not st["history"]:
                print(f"    {tag:24s} no attempts (forfeit)")
                continue
            best = min(st["history"], key=lambda h: h["distance"])
            hit = st["hit"]
            hs = (f"HIT on attempt {st['history'].index(hit)+1} "
                  f"(d={hit['distance']:.3f})" if hit else "no HIT")
            print(f"    {tag:24s} best d={best['distance']:.3f} "
                  f"scaffold={best['scaffold']:+.2f} "
                  f"attempts={len(st['history'])} {hs}")
            report[f"{tag}|{tgt}"] = {
                "best_distance": best["distance"], "best_vector": best["vector"],
                "best_scaffold": best["scaffold"],
                "attempts": len(st["history"]),
                "hit": bool(hit),
                "hit_attempt": st["history"].index(hit) + 1 if hit else None,
                "clips": [{k: h[k] for k in
                           ("wav", "vector", "V", "A", "D", "distance",
                            "family", "conf", "scaffold", "round")}
                          for h in st["history"]],
            }
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\n  report: {OUT/'report.json'}")
    print(f"  R1 spend: ${r1_propose._spent():.2f}")
    print("\nABC_P47_DONE")


if __name__ == "__main__":
    main()
