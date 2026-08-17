"""AFFECTOGRAM — the full-detail session report of the live ear.

A spectrogram shows the frequency content of sound; an Affectogram shows
the emotional content of speech. One figure, everything the ear measured:

  A. Emotion ribbon    smoothed family band over time + raw-name ticks
                       (flicker visible as tick density vs band calm)
  B. Valence trace     per-window + 30s rolling median, ambiguous windows
                       hollow, gated stretches greyed
  C. Arousal trace     same treatment
  D. Dominance trace   same treatment (computed since day 1, never drawn)
  E. Speech strip      Silero speech probability area + gate threshold
  F. Circumplex path   the session as a time-colored trajectory through
                       Russell's V-A plane, MSP centroids starred
                       (FEELtrace's design, driven by the model)
  G. Family share      horizontal bars, smoothed names
  H. Fact box          session metadata: duration, speech %, medians,
                       ambiguity, flicker raw->smoothed, latency, and the
                       exact protocol (window/stride/gate/smooth-k) so
                       every Affectogram is reproducible from its footer

Standalone: numpy + matplotlib only (no torch), so old sessions re-render.

Run:  venv/bin/python scripts/affectogram.py out/live_ear/X_traj.json
      (writes X_affectogram.png next to it)
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# same palette as live_ear.py (kept in sync by hand — this module must
# stay importable without torch)
COLORS = {"anger": "#d62728", "fear": "#9467bd", "joy": "#e6b800",
          "sadness": "#1f77b4", "surprise": "#ff7f0e", "neutral": "#909090",
          "disgust": "#2ca02c", "contempt": "#8c564b"}
GATE_GREY = "#b9b9b9"

# MSP-Podcast derived PAD centroids (statistics only) — mirrors the Namer
CENTROIDS = {"sadness": (-0.28, 0.39), "joy": (0.30, 0.58),
             "anger": (-0.42, 0.70), "surprise": (0.05, 0.64),
             "neutral": (-0.04, 0.45), "fear": (-0.21, 0.51),
             "disgust": (-0.34, 0.59)}


class NameSmoother:
    """Causal majority vote over the last k speech-window names.

    Sticky tie-break: the current smoothed name wins ties (hysteresis),
    else the most recent raw name among the tied winners. A gap of more
    than max_gap gated windows resets the buffer — a name should not
    survive across a scene break. k=5 at 1.5s stride = 7.5s memory;
    a genuine emotion change takes ~3 windows (4.5s) to flip the vote.
    Lives here (torch-free) so old sessions can be re-scored offline;
    live_ear imports it for the real-time path.
    """

    def __init__(self, k=5, max_gap=3):
        self.k, self.max_gap = k, max_gap
        self.buf, self.gap, self.state = [], 0, None

    def gate(self):
        self.gap += 1
        if self.gap > self.max_gap:
            self.buf, self.state = [], None

    def update(self, raw):
        self.gap = 0
        self.buf = (self.buf + [raw])[-self.k:]
        counts = Counter(self.buf)
        top = max(counts.values())
        winners = {n for n, v in counts.items() if v == top}
        if self.state not in winners:
            self.state = next(n for n in reversed(self.buf) if n in winners)
        return self.state


def flicker_rate(names):
    """Name changes between strictly adjacent speech windows."""
    pairs = trans = 0
    for a, b in zip(names, names[1:]):
        if a and b:
            pairs += 1
            trans += a != b
    return trans / pairs if pairs else 0.0


def _mmss(sec):
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def _rolling_median(x, k=21):
    """NaN-aware rolling median (k windows ~ 30s at 1.5s stride).
    Stays NaN where the source is NaN — the trend line must go dark
    over gated stretches, not float across silence."""
    out = np.full_like(x, np.nan)
    h = k // 2
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        seg = x[max(0, i - h):i + h + 1]
        if np.any(~np.isnan(seg)):
            out[i] = np.nanmedian(seg)
    return out


def render(rows, stem, out_path, params=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    p = params or {}
    stride = float(p.get("stride", 1.5))
    gate_th = float(p.get("speech_gate", 0.5))

    t = np.array([r["t"] for r in rows])
    speech = np.array([r["emotion"] != "no-speech" for r in rows])
    V = np.array([r["V"] if s else np.nan for r, s in zip(rows, speech)])
    A = np.array([r["A"] if s else np.nan for r, s in zip(rows, speech)])
    D = np.array([r.get("D") if s and r.get("D") is not None else np.nan
                  for r, s in zip(rows, speech)])
    sp = np.array([r.get("speech_prob", np.nan) for r in rows], dtype=float)
    sp[sp < 0] = np.nan
    amb = np.array([bool(r.get("ambiguous", False)) for r in rows])
    if any("emotion_raw" in r for r in rows):
        emo = [r["emotion"] if s else None for r, s in zip(rows, speech)]
        raw = [r.get("emotion_raw", r["emotion"]) if s else None
               for r, s in zip(rows, speech)]
    else:
        # pre-smoothing session: stored names ARE raw; smooth offline
        raw = [r["emotion"] if s else None for r, s in zip(rows, speech)]
        sm = NameSmoother(int((params or {}).get("smooth_k", 5)))
        emo = []
        for x in raw:
            if x is None:
                sm.gate(); emo.append(None)
            else:
                emo.append(sm.update(x))
    lat = [r["ms"] for r in rows if r["emotion"] != "no-speech"]

    dur = t[-1] + stride if len(t) else 0.0
    n_speech = int(speech.sum())

    fig = plt.figure(figsize=(17, 11), facecolor="white")
    gs = fig.add_gridspec(
        5, 2, width_ratios=[2.6, 1.0], height_ratios=[0.55, 1, 1, 1, 0.5],
        hspace=0.14, wspace=0.16, left=0.055, right=0.975,
        top=0.90, bottom=0.075)

    ax_rib = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_rib)
    ax_a = fig.add_subplot(gs[2, 0], sharex=ax_rib)
    ax_d = fig.add_subplot(gs[3, 0], sharex=ax_rib)
    ax_sp = fig.add_subplot(gs[4, 0], sharex=ax_rib)
    ax_cx = fig.add_subplot(gs[0:3, 1])
    ax_fam = fig.add_subplot(gs[3, 1])
    ax_txt = fig.add_subplot(gs[4, 1]); ax_txt.axis("off")

    # ---- A. emotion ribbon: smoothed band + raw ticks ----
    for i, e in enumerate(emo):
        c = COLORS.get(e, GATE_GREY) if e else GATE_GREY
        ax_rib.axvspan(t[i], t[i] + stride, ymin=0.35, ymax=1.0,
                       color=c, alpha=0.95 if e else 0.35, lw=0)
    for i, e in enumerate(raw):
        if e:
            ax_rib.axvspan(t[i], t[i] + stride, ymin=0.0, ymax=0.28,
                           color=COLORS.get(e, "#cccccc"), alpha=0.9, lw=0)
    ax_rib.set_ylim(0, 1); ax_rib.set_yticks([0.14, 0.675])
    ax_rib.set_yticklabels(["raw", "smoothed"], fontsize=8)
    ax_rib.tick_params(labelbottom=False, length=0)
    ax_rib.set_title("emotion timeline (top: majority-5 smoothed · "
                     "bottom: raw per-window)", fontsize=9, loc="left")

    # ---- B/C/D. dimension traces ----
    for ax, y, name, lim, color in [
            (ax_v, V, "Valence", (-1, 1), "#0b3d91"),
            (ax_a, A, "Arousal", (0, 1), "#8b0000"),
            (ax_d, D, "Dominance", (-1, 1), "#2e6f40")]:
        # gated stretches
        on = False
        for i, s in enumerate(speech):
            if not s and not on:
                x0, on = t[i], True
            elif s and on:
                ax.axvspan(x0, t[i], color="k", alpha=0.05, lw=0); on = False
        if on:
            ax.axvspan(x0, t[-1] + stride, color="k", alpha=0.05, lw=0)
        m_conf = ~np.isnan(y) & ~amb
        m_amb = ~np.isnan(y) & amb
        ax.scatter(t[m_conf], y[m_conf], s=8, color=color, alpha=0.55,
                   lw=0, label="confident")
        ax.scatter(t[m_amb], y[m_amb], s=10, facecolors="none",
                   edgecolors=color, alpha=0.45, lw=0.7, label="ambiguous")
        ax.plot(t, _rolling_median(y), color=color, lw=2.2, alpha=0.95,
                label="30s median")
        if lim[0] < 0:
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylim(*lim); ax.set_ylabel(name, fontsize=10)
        ax.tick_params(labelbottom=False)
    ax_v.legend(loc="upper right", fontsize=7, ncols=3, framealpha=0.6)

    # ---- E. speech probability strip ----
    ax_sp.fill_between(t, 0, np.nan_to_num(sp), color="#444444",
                       alpha=0.55, lw=0, step="post")
    ax_sp.axhline(gate_th, color="#d62728", lw=0.9, ls="--", alpha=0.8)
    ax_sp.set_ylim(0, 1); ax_sp.set_ylabel("speech\nprob", fontsize=8)
    ax_sp.set_xlabel("time", fontsize=10)
    ax_sp.set_xlim(0, max(30, dur))
    xt = np.arange(0, dur + 1, 60 if dur <= 900 else 120)
    ax_sp.set_xticks(xt)
    ax_sp.set_xticklabels([_mmss(x) for x in xt], fontsize=8)

    # ---- F. circumplex path (Russell plane, time-colored) ----
    ax_cx.axhline(0.5, color="k", lw=0.5, alpha=0.25)
    ax_cx.axvline(0, color="k", lw=0.5, alpha=0.25)
    for q, (x, y) in {"tense": (-0.93, 0.97), "excited": (0.93, 0.97),
                      "gloomy": (-0.93, 0.04), "serene": (0.93, 0.04)}.items():
        ax_cx.text(x, y, q, fontsize=8, alpha=0.45, style="italic",
                   ha="left" if x < 0 else "right",
                   va="top" if y > 0.5 else "bottom")
    for fam, (cv, ca) in CENTROIDS.items():
        ax_cx.plot(cv, ca, marker="*", ms=11, color=COLORS[fam],
                   mec="k", mew=0.4, zorder=5)
        ax_cx.annotate(fam, (cv, ca), textcoords="offset points",
                       xytext=(5, 4), fontsize=7.5, color=COLORS[fam])
    m = ~np.isnan(V)
    if m.sum() > 1:
        vv, aa, tt = V[m], A[m], t[m]
        ax_cx.plot(vv, aa, color="#999999", lw=0.5, alpha=0.35, zorder=2)
        sc = ax_cx.scatter(vv, aa, c=tt / 60, cmap="viridis", s=13,
                           alpha=0.8, lw=0, zorder=3)
        cb = plt.colorbar(sc, ax=ax_cx, fraction=0.045, pad=0.02)
        cb.set_label("minutes", fontsize=8); cb.ax.tick_params(labelsize=7)
        ax_cx.plot(vv[0], aa[0], "o", ms=9, mfc="none", mec="k", mew=1.4,
                   zorder=6)
        ax_cx.plot(vv[-1], aa[-1], "s", ms=9, mfc="none", mec="k", mew=1.4,
                   zorder=6)
        ax_cx.annotate("start", (vv[0], aa[0]), textcoords="offset points",
                       xytext=(6, -10), fontsize=7)
        ax_cx.annotate("end", (vv[-1], aa[-1]), textcoords="offset points",
                       xytext=(6, -10), fontsize=7)
    ax_cx.set_xlim(-1, 1); ax_cx.set_ylim(0, 1)
    ax_cx.set_xlabel("valence", fontsize=9)
    ax_cx.set_ylabel("arousal", fontsize=9)
    ax_cx.set_title("circumplex path (V-A plane, colored by time)",
                    fontsize=9, loc="left")

    # ---- G. family share (smoothed) ----
    fams = [e for e in emo if e]
    order = sorted(set(fams), key=fams.count)
    counts = [fams.count(f) for f in order]
    ax_fam.barh(order, [c / max(1, len(fams)) * 100 for c in counts],
                color=[COLORS.get(f, "#cccccc") for f in order],
                alpha=0.9)
    for i, (f, c) in enumerate(zip(order, counts)):
        ax_fam.text(c / max(1, len(fams)) * 100 + 0.8, i,
                    f"{c / max(1, len(fams)) * 100:.0f}%",
                    va="center", fontsize=7.5)
    ax_fam.set_xlabel("family share — % of speech windows (smoothed)",
                      fontsize=8, labelpad=2)
    ax_fam.tick_params(labelsize=8)
    ax_fam.set_xlim(0, 60)

    # ---- H. fact box ----
    amb_rate = amb[speech].mean() if n_speech else 0.0
    facts = [
        f"duration {_mmss(dur)} · {len(rows)} windows · "
        f"speech {n_speech} ({n_speech / max(1, len(rows)) * 100:.0f}%)",
        f"V median {np.nanmedian(V):+.2f} · A median {np.nanmedian(A):.2f}"
        f" · D median {np.nanmedian(D):+.2f}",
        f"ambiguous {amb_rate * 100:.0f}% · flicker raw "
        f"{flicker_rate(raw) * 100:.0f}% → smoothed "
        f"{flicker_rate(emo) * 100:.0f}%",
        f"median latency {np.median(lat):.0f} ms/window" if lat else "",
        f"protocol: window {p.get('window', 3.0)}s · stride {stride}s · "
        f"gate {gate_th} · smooth-k {p.get('smooth_k', 5)}",
        "model: wavlm_vad_ft + namer_msp_final (PAD centroids)",
    ]
    ax_txt.text(0, 0.88, "\n".join(f for f in facts if f), fontsize=8.2,
                va="top", family="monospace", linespacing=1.55)

    fig.suptitle(f"AFFECTOGRAM — {stem}", fontsize=14, x=0.055,
                 ha="left", fontweight="bold")
    legend = [Patch(color=COLORS[f], label=f) for f in COLORS
              if f in set(fams) | set(CENTROIDS)]
    fig.legend(handles=legend, loc="upper right", ncols=len(legend),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.975, 0.965))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for arg in sys.argv[1:]:
        path = Path(arg)
        rows = json.loads(path.read_text())
        stem = path.stem.replace("_traj", "")
        out = path.with_name(f"{stem}_affectogram.png")
        render(rows, stem, out)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
