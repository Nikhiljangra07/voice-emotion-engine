"""P4.4b — THE FAIR REMATCH: every system gets the loop.

Fixes the fixable catches from THE CATCH (STEERING_LOG 2026-07-05):
  * CATCH 1 (iteration asymmetry): every rival now runs the SAME closed loop —
    3 seeds -> judge -> up to 3 deterministic error-keyed refinements. Budget
    <= 6 judged steering attempts per emotion per system (+3 voice-test clips
    on anger for ElevenLabs/OpenAI, disclosed).
  * CATCH 4 (voice handicap): ElevenLabs and OpenAI get a voice-SELECTION round —
    3 expressive candidate voices tested on the anger seed, winner picked by the
    meter, not by hand.
  * CATCH 5 (n=1): a GENERALIZATION round — each system's best config per emotion
    re-synthesized on 2 unseen sentences and re-judged.
  * CATCH 2 (naturalness) cannot be fixed by machine — needs blind human ears.
  * CATCH 3 (product-choice flatness) is not ours to fix; we measure what we get.

Constants: same neutral sentence for steering (S1), same frozen judge, same MSP
centroids, semantics NEVER changed (delivery-only control). Deterministic — no RNG.
Refinement rule: dominant error axis of best attempt picks the next candidate from
a pre-authored pool (arousal_down / arousal_up / valence_up / intensify).

Ours (IndexTTS-2) enters with its ALREADY-LOGGED steering history (29 attempts,
S1) — no new steering; it joins the generalization round only. Per-emotion attempt
counts for every system are disclosed in the output.

Run:  .venv_tts/bin/python tts_steering/fair_p44b.py
"""

import base64
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import judge  # noqa: E402

ROOT = HERE.parent
OUT = HERE / "out/p44b"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = HERE / "out/loop_ledger.csv"
VENDOR = HERE / "vendor/index-tts"

S1 = "The table is in the room, and the door is closed."
GEN_SENTENCES = {"s2": "She opened the letter and read it slowly.",
                 "s3": "The meeting starts at nine tomorrow morning."}

CENTROIDS = {
    "sadness": (-0.28, 0.39, -0.07),
    "joy":     (+0.30, 0.58, +0.19),
    "anger":   (-0.42, 0.70, +0.41),
}
EMOTIONS = list(CENTROIDS)
MAX_REFINES = 3  # after 3 seeds -> <=6 steering attempts per emotion

KEYS = {}
for line in (HERE / ".keys.env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, _, v = line.partition("=")
        KEYS[k.strip()] = v.strip()

RAV = ROOT / "data/ravdess/Actor_01"
CBX_REFS = {"joy": str(RAV / "03-01-03-02-01-01-01.wav"),
            "sadness": str(RAV / "03-01-04-02-01-01-01.wav"),
            "anger": str(RAV / "03-01-05-02-01-01-01.wav")}

# ---------------------------------------------------------------- control spaces
# Each candidate is a dict; "desc" is the ledger control string.
# Voice-test candidates (anger only, EL + OAI): expressive premades, meter decides.
EL_VOICES = {"charlie": "IKne3meq5aSn9XLyUdCD",   # Deep, Confident, Energetic
             "harry":   "SOYHLrjzK2X1ezoPC6cr",   # Fierce Warrior
             "laura":   "FGY2WhTYpPnrIDTdsKH5"}   # Enthusiast, Quirky Attitude
OAI_VOICES = ["ash", "coral", "ballad"]

SEEDS = {
    "elevenlabs": {
        "anger":   [{"tags": "[angry] ", "stability": 0.0, "voice": v}
                    for v in EL_VOICES.values()],           # doubles as voice test
        "sadness": [{"tags": "[sad] ", "stability": 0.0},
                    {"tags": "[crying] ", "stability": 0.0},
                    {"tags": "[sad][sighs] ", "stability": 0.5}],
        "joy":     [{"tags": "[happily] ", "stability": 0.0},
                    {"tags": "[laughs] ", "stability": 0.0},
                    {"tags": "[excited] ", "stability": 0.5}],
    },
    "hume-octave": {
        "anger":   [{"desc": "furious, harsh, clipped, seething with rage"},
                    {"desc": "shouting in outrage, aggressive, confrontational"},
                    {"desc": "cold fury, menacing, speaking through gritted teeth"}],
        "sadness": [{"desc": "quietly devastated, voice breaking, holding back tears, slow"},
                    {"desc": "grieving at a funeral, hollow and drained, almost whispering"},
                    {"desc": "heartbroken, trembling voice, long heavy pauses"}],
        "joy":     [{"desc": "genuinely delighted, warm bright smile in the voice, light"},
                    {"desc": "laughing with joy, just heard wonderful news, elated"},
                    {"desc": "cheerful and playful, sunny warmth, giggling between words"}],
    },
    "openai-tts": {
        "anger":   [{"instr": "Speak with furious, harsh, clipped, seething rage — "
                              "raised voice, aggressive, like a screaming match.",
                     "voice": v} for v in OAI_VOICES],       # doubles as voice test
        "sadness": [{"instr": "Speak as if quietly devastated: slow, heavy, voice "
                              "breaking, on the verge of tears."},
                    {"instr": "Speak like someone grieving at a funeral: hollow, "
                              "drained, almost whispering, long pauses."},
                    {"instr": "Speak with deep melancholy: low pitch, trembling, "
                              "exhausted by sorrow."}],
        "joy":     [{"instr": "Speak with genuine delight: warm, bright, smiling "
                              "while speaking, light and bouncy."},
                    {"instr": "Speak as if you just heard wonderful news: elated, "
                              "laughing warmth in every word."},
                    {"instr": "Speak cheerfully and playfully: sunny, melodious, "
                              "upbeat but natural."}],
    },
    "chatterbox": {
        "anger":   [{"ex": 0.4, "cfg": 0.5}, {"ex": 0.6, "cfg": 0.5}, {"ex": 0.8, "cfg": 0.5}],
        "sadness": [{"ex": 0.3, "cfg": 0.5}, {"ex": 0.5, "cfg": 0.5}, {"ex": 0.7, "cfg": 0.5}],
        "joy":     [{"ex": 0.4, "cfg": 0.5}, {"ex": 0.5, "cfg": 0.5}, {"ex": 0.6, "cfg": 0.5}],
    },
}

# Error-keyed refinement pools. Key = dominant error axis of the best attempt.
REFINE = {
    "elevenlabs": {
        "anger":   {"arousal_down": [{"tags": "[angry] ", "stability": 0.5}],
                    "intensify":    [{"tags": "[angry][shouting] ", "stability": 0.0}],
                    "valence_up":   [{"tags": "[angry] ", "stability": 0.5}]},
        "sadness": {"arousal_down": [{"tags": "[sad][whispers] ", "stability": 0.0}],
                    "intensify":    [{"tags": "[crying][sighs] ", "stability": 0.0}],
                    "valence_up":   [{"tags": "[sad] ", "stability": 0.5}]},
        "joy":     {"valence_up":   [{"tags": "[laughs][happily] ", "stability": 0.0}],
                    "arousal_down": [{"tags": "[happily] ", "stability": 0.5}],
                    "intensify":    [{"tags": "[excited][laughs] ", "stability": 0.0}]},
    },
    "hume-octave": {
        "anger":   {"arousal_down": [{"desc": "cold controlled fury, low menacing voice"}],
                    "intensify":    [{"desc": "explosive rage, yelling at the top of the lungs"}],
                    "valence_up":   [{"desc": "bitter contempt, sharp and dismissive"}]},
        "sadness": {"arousal_down": [{"desc": "numb, exhausted by grief, flat, barely audible"}],
                    "intensify":    [{"desc": "openly crying while speaking, sobbing between words"}],
                    "valence_up":   [{"desc": "wistful, mourning something lost, soft and heavy"}]},
        "joy":     {"valence_up":   [{"desc": "overflowing warm happiness, melodious laughter in every word"}],
                    "arousal_down": [{"desc": "softly content, serene warm smile, gentle happiness"}],
                    "intensify":    [{"desc": "ecstatic celebration, whooping with delight"}]},
    },
    "openai-tts": {
        "anger":   {"arousal_down": [{"instr": "Speak with cold, controlled fury: low, "
                                               "menacing, through gritted teeth."}],
                    "intensify":    [{"instr": "You are a method actor playing explosive "
                                               "rage: yell, snarl, maximum aggression."}],
                    "valence_up":   [{"instr": "Speak with bitter contempt: sharp, "
                                               "dismissive, sneering."}]},
        "sadness": {"arousal_down": [{"instr": "Speak numb with grief: flat, lifeless, "
                                               "barely audible, drained of all energy."}],
                    "intensify":    [{"instr": "You are a method actor: openly crying "
                                               "while speaking, voice cracking, sobbing."}],
                    "valence_up":   [{"instr": "Speak wistfully, mourning something "
                                               "lost: soft, heavy, slow."}]},
        "joy":     {"valence_up":   [{"instr": "You are a method actor overflowing with "
                                               "warm happiness: melodious, laughing, radiant."}],
                    "arousal_down": [{"instr": "Speak softly content: serene, warm "
                                               "smile, gentle calm happiness."}],
                    "intensify":    [{"instr": "Speak ecstatically: celebrating, "
                                               "whooping with delight, maximum joy."}]},
    },
    "chatterbox": {
        "anger":   {"arousal_down": [{"ex": 0.5, "cfg": 0.7}],
                    "intensify":    [{"ex": 1.0, "cfg": 0.5}],
                    "valence_up":   [{"ex": 0.6, "cfg": 0.3}]},
        "sadness": {"arousal_down": [{"ex": 0.4, "cfg": 0.7}],
                    "intensify":    [{"ex": 0.9, "cfg": 0.5}],
                    "valence_up":   [{"ex": 0.5, "cfg": 0.3}]},
        "joy":     {"valence_up":   [{"ex": 0.5, "cfg": 0.3}],
                    "arousal_down": [{"ex": 0.45, "cfg": 0.7}],
                    "intensify":    [{"ex": 0.7, "cfg": 0.5}]},
    },
}

# Ours: best steered configs from the logged history (S1, 29 attempts) — enters
# the generalization round only.
OURS_BEST = {
    "anger":   [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sadness": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3],
    "joy":     [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


# ------------------------------------------------------------------ synthesizers
def synth_el(cand: dict, text: str, wav: Path) -> str:
    voice = cand.get("voice", EL_STATE.get("voice", "IKne3meq5aSn9XLyUdCD"))
    vn = [k for k, v in EL_VOICES.items() if v == voice]
    ctrl = (f"v3;tags={cand['tags'].strip()};stab={cand['stability']};"
            f"voice={vn[0] if vn else voice}")
    if wav.exists():
        return ctrl
    mp3 = wav.with_suffix(".mp3")
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
        headers={"xi-api-key": KEYS["ELEVENLABS_API_KEY"]},
        json={"text": cand["tags"] + text, "model_id": "eleven_v3",
              "voice_settings": {"stability": cand["stability"]}},
        timeout=120)
    r.raise_for_status()
    mp3.write_bytes(r.content)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ar", "16000", "-ac", "1", str(wav)], check=True)
    return ctrl


def synth_hume(cand: dict, text: str, wav: Path) -> str:
    ctrl = f"octave;desc={cand['desc'][:60]}"
    if wav.exists():
        return ctrl
    r = requests.post(
        "https://api.hume.ai/v0/tts",
        headers={"X-Hume-Api-Key": KEYS["HUME_API_KEY"]},
        json={"utterances": [{"text": text, "description": cand["desc"]}],
              "format": {"type": "wav"}, "num_generations": 1},
        timeout=180)
    r.raise_for_status()
    raw = wav.with_suffix(".raw.wav")
    raw.write_bytes(base64.b64decode(r.json()["generations"][0]["audio"]))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                    "-ar", "16000", "-ac", "1", str(wav)], check=True)
    return ctrl


def synth_oai(cand: dict, text: str, wav: Path) -> str:
    voice = cand.get("voice", OAI_STATE.get("voice", "coral"))
    ctrl = f"4o-mini-tts;voice={voice};instr={cand['instr'][:60]}"
    if wav.exists():
        return ctrl
    raw = wav.with_suffix(".raw.wav")
    r = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {KEYS['OPENAI_API_KEY']}"},
        json={"model": "gpt-4o-mini-tts", "voice": voice, "input": text,
              "instructions": cand["instr"], "response_format": "wav"},
        timeout=120)
    r.raise_for_status()
    raw.write_bytes(r.content)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                    "-ar", "16000", "-ac", "1", str(wav)], check=True)
    return ctrl


CBX_PENDING: list = []  # batched, synthesized once per round via cbx_worker


def synth_cbx(cand: dict, text: str, wav: Path, emotion: str) -> str:
    CBX_PENDING.append({"ref": CBX_REFS[emotion], "exaggeration": cand["ex"],
                        "cfg_weight": cand["cfg"], "text": text, "out": str(wav)})
    return f"ref={emotion};ex={cand['ex']};cfg={cand['cfg']}"


def flush_cbx() -> None:
    if not CBX_PENDING:
        return
    jf = OUT / "_cbx_jobs.json"
    jf.write_text(json.dumps(CBX_PENDING))
    proc = subprocess.run([str(ROOT / ".venv_cbx/bin/python"),
                           str(HERE / "cbx_worker.py"), str(jf)],
                          capture_output=True, text=True, timeout=3600)
    if "CBX_WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"cbx worker failed:\n{proc.stderr[-600:]}")
    CBX_PENDING.clear()


SYNTH = {"elevenlabs": synth_el, "hume-octave": synth_hume,
         "openai-tts": synth_oai, "chatterbox": synth_cbx}
EL_STATE: dict = {}
OAI_STATE: dict = {}


# ------------------------------------------------------------------ loop helpers
def dist(m: dict, target: str) -> float:
    c = CENTROIDS[target]
    return math.sqrt((m["V"] - c[0]) ** 2 + (m["A"] - c[1]) ** 2 + (m["D"] - c[2]) ** 2)


def error_key(m: dict, target: str) -> str:
    c = CENTROIDS[target]
    v_err, a_err = m["V"] - c[0], m["A"] - c[1]
    if abs(a_err) >= abs(v_err):
        return "arousal_down" if a_err > 0 else "intensify"
    return "valence_up" if v_err < 0 else "intensify"


def append_ledger(rows: list) -> int:
    with open(LEDGER, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    return sum(1 for _ in open(LEDGER)) - 1


def main() -> None:
    n = sum(1 for _ in open(LEDGER)) - 1
    state = {s: {e: {"best": None, "hit": None, "attempts": 0, "used": set()}
                 for e in EMOTIONS} for s in SEEDS}
    ledger_rows: list = []

    for rnd in range(1, 2 + MAX_REFINES):
        batch: list = []  # (system, emotion, cand, wav_path, control_str)
        for system in SEEDS:
            for emo in EMOTIONS:
                st = state[system][emo]
                if rnd == 1:
                    cands = SEEDS[system][emo]
                else:
                    if st["best"] is None:
                        continue
                    key = error_key(st["best"], emo)
                    pool = REFINE[system][emo].get(key, []) \
                        + REFINE[system][emo].get("intensify", [])
                    cands = [c for c in pool if json.dumps(c, sort_keys=True)
                             not in st["used"]][:1]
                for cand in cands:
                    st["used"].add(json.dumps(cand, sort_keys=True))
                    st["attempts"] += 1
                    name = f"{system.split('-')[0]}_{emo}_r{rnd}_{st['attempts']}"
                    wav = OUT / f"{name}.wav"
                    if system == "chatterbox":
                        ctrl = synth_cbx(cand, S1, wav, emo)
                        batch.append((system, emo, cand, wav, ctrl))
                    else:
                        try:
                            ctrl = SYNTH[system](cand, S1, wav)
                            batch.append((system, emo, cand, wav, ctrl))
                        except Exception as exc:
                            print(f"  {name} SYNTH FAILED: {exc}", flush=True)
        if not batch:
            break
        flush_cbx()
        print(f"\n=== ROUND {rnd}: judging {len(batch)} clips ===", flush=True)
        results = judge([str(b[3]) for b in batch])
        for (system, emo, cand, wav, ctrl), r in zip(batch, results):
            d = dist(r, emo)
            hit = r["judge_family"] == emo
            n += 1
            st = state[system][emo]
            rec = {"V": r["V"], "A": r["A"], "D": r["D"], "dist": d, "cand": cand,
                   "family": r["judge_family"], "conf": r["judge_confidence"],
                   "clip": wav.name, "ctrl": ctrl, "hit": hit}
            if st["best"] is None or d < st["best"]["dist"]:
                st["best"] = rec
            if hit and (st["hit"] is None or d < st["hit"]["dist"]):
                st["hit"] = rec
            # voice selection: lock EL/OAI voice from the anger seed round
            if rnd == 1 and emo == "anger":
                if system == "elevenlabs" and st["best"] is rec:
                    EL_STATE["voice"] = cand["voice"]
                if system == "openai-tts" and st["best"] is rec:
                    OAI_STATE["voice"] = cand["voice"]
            ledger_rows.append([n, f"{system}-fair", emo, ctrl,
                                round(r["V"], 3), round(r["A"], 3), round(r["D"], 3),
                                r["judge_family"], round(r["judge_confidence"], 2),
                                round(d, 3), int(hit)])
            mark = "HIT" if hit else "   "
            print(f"  {wav.name:30s} d={d:.3f} judge={r['judge_family']}"
                  f"@{r['judge_confidence']:.0%} {mark} [{ctrl[:60]}]", flush=True)
        append_ledger(ledger_rows)   # persist per round — a crash loses nothing
        ledger_rows = []

    # ---------------------------------------------------------- generalization
    print("\n=== GENERALIZATION: best config per system-emotion on 2 unseen "
          "sentences ===", flush=True)
    gen_batch: list = []
    # ours: synthesize best vectors on S2/S3 via vendor worker
    jobs = []
    for emo, vec in OURS_BEST.items():
        for sid, text in GEN_SENTENCES.items():
            wav = OUT / f"ours_{emo}_{sid}.wav"
            jobs.append({"prompt": str(RAV / "03-01-01-01-01-01-01.wav"),
                         "text": text, "vector": vec, "out": str(wav)})
            gen_batch.append(("indextts2", emo, sid, wav, f"emo_vector={vec}"))
    jf = OUT / "_gen_jobs.json"
    jf.write_text(json.dumps(jobs))
    proc = subprocess.run([str(VENDOR / ".venv/bin/python"),
                           str(HERE / "synth_worker.py"), str(jf)],
                          cwd=str(VENDOR),
                          env={"PYTHONPATH": str(VENDOR), "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=3600)
    if "WORKER_DONE" not in proc.stdout:
        raise RuntimeError(f"vendor worker failed:\n{proc.stderr[-600:]}")
    # rivals: best candidate per emotion on S2/S3
    for system in SEEDS:
        for emo in EMOTIONS:
            best = state[system][emo]["hit"] or state[system][emo]["best"]
            if best is None:
                continue
            for sid, text in GEN_SENTENCES.items():
                wav = OUT / f"{system.split('-')[0]}_{emo}_{sid}.wav"
                if system == "chatterbox":
                    ctrl = synth_cbx(best["cand"], text, wav, emo)
                    gen_batch.append((system, emo, sid, wav, ctrl))
                else:
                    try:
                        ctrl = SYNTH[system](best["cand"], text, wav)
                        gen_batch.append((system, emo, sid, wav, ctrl))
                    except Exception as exc:
                        print(f"  {wav.name} SYNTH FAILED: {exc}", flush=True)
    flush_cbx()
    results = judge([str(b[3]) for b in gen_batch])
    gen: dict = {}
    for (system, emo, sid, wav, ctrl), r in zip(gen_batch, results):
        d = dist(r, emo)
        hit = r["judge_family"] == emo
        n += 1
        ledger_rows.append([n, f"{system}-fair-gen", emo, f"{ctrl};sent={sid}",
                            round(r["V"], 3), round(r["A"], 3), round(r["D"], 3),
                            r["judge_family"], round(r["judge_confidence"], 2),
                            round(d, 3), int(hit)])
        gen.setdefault(system, {}).setdefault(emo, []).append(
            {"sent": sid, "dist": round(d, 3), "family": r["judge_family"],
             "conf": r["judge_confidence"], "hit": hit})
        mark = "HIT" if hit else "   "
        print(f"  {wav.name:30s} d={d:.3f} judge={r['judge_family']}"
              f"@{r['judge_confidence']:.0%} {mark}", flush=True)

    total = append_ledger(ledger_rows)

    # -------------------------------------------------------------- summary
    print("\n" + "=" * 76)
    summary = {"steering": {}, "generalization": gen,
               "voice_selected": {"elevenlabs": EL_STATE.get("voice"),
                                  "openai": OAI_STATE.get("voice")}}
    for system in SEEDS:
        summary["steering"][system] = {}
        for emo in EMOTIONS:
            st = state[system][emo]
            b, h = st["best"], st["hit"]
            summary["steering"][system][emo] = {
                "attempts": st["attempts"],
                "best_dist": None if b is None else round(b["dist"], 3),
                "best_family": None if b is None else b["family"],
                "hit": None if h is None else
                {"clip": h["clip"], "dist": round(h["dist"], 3),
                 "conf": h["conf"], "ctrl": h["ctrl"]}}
            status = (f"HIT {h['clip']} d={h['dist']:.3f} conf={h['conf']:.0%}"
                      if h else
                      f"no hit; best d={b['dist']:.3f} ({b['family']})" if b
                      else "all failed")
            print(f"{system:12s} {emo:8s} attempts={st['attempts']}  {status}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nledger rows added: {len(ledger_rows)} (total {total})")
    print("FAIR_P44B_DONE")


if __name__ == "__main__":
    main()
