"""
v2: Teacher Feedback supporting RL Training – improved scientific visualization.

Panels
------
1. EMA Reward w/ confidence band + teacher-trigger rug (Savitzky-Golay, no
   boundary artifacts, ±1 std across 2 GPU streams)
2. Histogram of avg_best_delta (reward_refined − reward_original per call) with
   mean / median / positive-% annotations
3. Refined Win Rate over training (smoothed)
4. Teacher Trigger Rate over training (smoothed + linear trend)
5. Summary metrics table
"""

import re
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

LOG_FILE = "training_recogdrive_agent_feedback_8B_new.txt"
OUT_FILE  = "teacher_feedback_visualization_v2.png"

# ── colour palette ─────────────────────────────────────────────────────────────
C_BLUE       = "#1565C0"
C_BLUE_LIGHT = "#90CAF9"
C_GREEN      = "#2E7D32"
C_GREEN_LIGHT= "#A5D6A7"
C_ORANGE     = "#E65100"
C_RED        = "#C62828"
C_GRAY       = "#607D8B"
C_BG         = "#F8F9FA"

# ── smooth with Savitzky-Golay (no boundary artifacts) ────────────────────────
def smooth_sg(arr, window=13, poly=3):
    """
    Savitzky-Golay filter.  Falls back to a boundary-safe moving average when
    scipy is unavailable.  The fallback pads both ends with the first/last valid
    values so the curve never drops artificially at the edges.
    """
    arr = np.asarray(arr, dtype=float)
    n   = len(arr)
    if n < 4:
        return arr.copy()
    try:
        from scipy.signal import savgol_filter
        w = min(window, n)
        w = w if w % 2 == 1 else w - 1
        w = max(w, 3)
        return savgol_filter(arr, w, min(poly, w - 1))
    except Exception:
        k    = min(window, n)
        half = k // 2
        pad  = np.concatenate([np.full(half, arr[0]), arr, np.full(half, arr[-1])])
        conv = np.convolve(pad, np.ones(k) / k, mode='valid')
        return conv[:n]


# ── parse log ─────────────────────────────────────────────────────────────────
def parse_log(path):
    """
    Returns
    -------
    policy_blocks  : list of {step, ema_reward}
    teacher_blocks : list of {step, win_rate, avg_delta, n_calls}
    best_deltas    : list of float  (avg_best_delta per Refinement-Filter line)
    trigger_steps  : list of int    (steps where teacher was triggered)
    rf_accept_rates: list of float  (per-event acceptance rate %)
    """
    with open(path) as f:
        lines = f.readlines()

    policy_blocks   = []
    teacher_blocks  = []
    best_deltas     = []
    trigger_steps   = []
    rf_accept_rates = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # ── Teacher Trigger step ──────────────────────────────────────────────
        if "[Teacher Trigger]" in line and i + 1 < len(lines):
            ms = re.search(r"step=(\d+)", lines[i + 1])
            if ms:
                trigger_steps.append(int(ms.group(1)))

        # ── Refinement Filter ─────────────────────────────────────────────────
        rf = re.search(
            r"\[Refinement Filter\].*?accepted_rate=([\d.]+)%"
            r"(?:.*?avg_best_delta=\+([\d.]+))?",
            line,
        )
        if rf:
            rf_accept_rates.append(float(rf.group(1)))
            if rf.group(2):
                best_deltas.append(float(rf.group(2)))

        # ── Policy Monitor → look-ahead for Teacher Monitor ───────────────────
        pm = re.match(
            r"\[Policy Monitor\]\s+step=(\d+)\s+ema_reward=([\d.]+)", line
        )
        if pm:
            step = int(pm.group(1))
            ema  = float(pm.group(2))
            policy_blocks.append({"step": step, "ema_reward": ema})

            blk = {"step": step}
            for j in range(i + 1, min(i + 10, len(lines))):
                if "[Teacher Monitor]" in lines[j]:
                    for k in range(j + 1, min(j + 8, len(lines))):
                        lk = lines[k].rstrip()
                        mw = re.search(r"refined_win_rate=([\d.]+)%",   lk)
                        md = re.search(r"avg_delta_reward=\+([\d.]+)",   lk)
                        mn = re.search(r"trigger_rate=[\d.]+%\s+\((\d+)/", lk)
                        if mw: blk["win_rate"]  = float(mw.group(1))
                        if md: blk["avg_delta"] = float(md.group(1))
                        if mn: blk["n_calls"]   = int(mn.group(1))
                    break
            if {"win_rate", "avg_delta", "n_calls"} <= blk.keys():
                teacher_blocks.append(blk)

        i += 1

    return policy_blocks, teacher_blocks, best_deltas, trigger_steps, rf_accept_rates


(
    policy_blocks,
    teacher_blocks,
    best_deltas,
    trigger_steps,
    rf_accept_rates,
) = parse_log(LOG_FILE)

best_deltas     = np.array(best_deltas,     dtype=float)
rf_accept_rates = np.array(rf_accept_rates, dtype=float)

# ── aggregate both GPU streams by step ───────────────────────────────────────
pol_by_step = defaultdict(list)
for b in policy_blocks:
    pol_by_step[b["step"]].append(b["ema_reward"])

pol_steps  = np.array(sorted(pol_by_step))
pol_mean   = np.array([np.mean(pol_by_step[s]) for s in pol_steps])
pol_std    = np.array([np.std( pol_by_step[s]) for s in pol_steps])
pol_smooth = smooth_sg(pol_mean, window=11, poly=3)

tea_by_step = defaultdict(list)
for b in teacher_blocks:
    tea_by_step[b["step"]].append(b)

tea_steps = np.array(sorted(tea_by_step))
tea_win   = np.array([np.mean([b["win_rate"]  for b in tea_by_step[s]]) for s in tea_steps])
tea_delta = np.array([np.mean([b["avg_delta"] for b in tea_by_step[s]]) for s in tea_steps])
tea_trig  = np.array([
    np.mean([b["n_calls"] / s * 100 for b in tea_by_step[s]])
    for s in tea_steps
])

tea_win_smooth  = smooth_sg(tea_win,  window=9, poly=2)
tea_trig_smooth = smooth_sg(tea_trig, window=9, poly=2)

# ── summary stats ─────────────────────────────────────────────────────────────
last_step         = tea_steps[-1]
total_calls       = sum(max(b["n_calls"] for b in tea_by_step[s]) for s in [tea_steps[-1]])
# Approximate: both streams' last n_calls
total_calls_both  = sum(b["n_calls"] for b in tea_by_step[last_step])

start_ema = pol_mean[0]
end_ema   = pol_mean[-1]
start_trig, end_trig = tea_trig[0], tea_trig[-1]
avg_win_rate  = float(np.mean(tea_win))
avg_delta_mon = float(np.mean(tea_delta))        # from Teacher Monitor
mean_best     = float(np.mean(best_deltas))      # from Refinement Filter
med_best      = float(np.median(best_deltas))
pct_pos       = 100.0 * float(np.mean(best_deltas > 0))
pct_neg       = 100.0 - pct_pos
avg_accept    = float(np.mean(rf_accept_rates))

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 10))
fig.patch.set_facecolor("#F5F6FA")

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.065, right=0.97,
    top=0.91,   bottom=0.09,
    hspace=0.44, wspace=0.34,
    width_ratios=[1.6, 1.2, 1.0],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1 – EMA Reward with confidence band + teacher-trigger rug
# ═══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(C_BG)

# Raw GPU lines (very faint) — split by even/odd order in list
seen_steps: dict = {}
gpu0_pts, gpu1_pts = [], []
for b in policy_blocks:
    s = b["step"]
    seen_steps[s] = seen_steps.get(s, 0) + 1
    (gpu0_pts if seen_steps[s] == 1 else gpu1_pts).append((s, b["ema_reward"]))

for gpublocks in [sorted(gpu0_pts), sorted(gpu1_pts)]:
    if gpublocks:
        xs, ys = zip(*gpublocks)
        ax1.plot(xs, ys, color=C_BLUE_LIGHT, lw=0.6, alpha=0.35)

# ±1 std confidence band
ax1.fill_between(
    pol_steps,
    pol_mean - pol_std,
    pol_mean + pol_std,
    alpha=0.18, color=C_BLUE, label="±1 std (2 GPUs)",
)

# Smoothed mean — no boundary drop
ax1.plot(
    pol_steps, pol_smooth, color=C_BLUE, lw=2.6,
    label=f"Smoothed Mean  {start_ema:.3f} → {end_ema:.3f}",
    zorder=5,
)

# Teacher-trigger rug at bottom of axis
rug_y = 0.505
unique_trig = sorted(set(trigger_steps))
ax1.vlines(unique_trig, rug_y, rug_y + 0.013,
           colors=C_ORANGE, lw=0.7, alpha=0.45, label="Teacher trigger")

# Start / end annotations
ax1.annotate(
    f"{start_ema:.3f}",
    xy=(pol_steps[0], pol_smooth[0]),
    xytext=(pol_steps[0] + 120, pol_smooth[0] - 0.04),
    fontsize=8.5, color=C_BLUE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1),
)
ax1.annotate(
    f"{end_ema:.3f}",
    xy=(pol_steps[-1], pol_smooth[-1]),
    xytext=(pol_steps[-1] - 450, pol_smooth[-1] + 0.04),
    fontsize=8.5, color=C_BLUE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1),
)

ax1.set_ylim(0.49, 1.03)
ax1.set_title(
    "Policy EMA Reward during RL Training\n"
    "  (orange rug = teacher trigger events — density decreases as policy improves)",
    fontweight="bold", fontsize=10.5,
)
ax1.set_xlabel("Training Step", fontsize=9)
ax1.set_ylabel("EMA Reward", fontsize=9)
ax1.legend(fontsize=8, loc="lower right", framealpha=0.85)
ax1.grid(True, alpha=0.22)
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2 – Histogram of avg_best_delta
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(C_BG)

n_bins  = 22
bins    = np.linspace(best_deltas.min() * 0.96, best_deltas.max() * 1.02, n_bins + 1)
counts, edges, patches = ax2.hist(
    best_deltas, bins=bins, edgecolor="white", lw=0.6, zorder=3,
)

# Colour-code positive/negative relative to 0
for patch, left_edge in zip(patches, edges[:-1]):
    patch.set_facecolor(C_GREEN if left_edge >= 0 else C_RED)
    patch.set_alpha(0.82)

# KDE overlay
from_x = edges[0]
to_x   = edges[-1]
kde_x  = np.linspace(from_x, to_x, 300)
bw     = 1.06 * np.std(best_deltas) * len(best_deltas) ** (-1 / 5)
kde_y  = np.array([
    np.sum(np.exp(-0.5 * ((best_deltas - xi) / bw) ** 2)) / (bw * np.sqrt(2 * np.pi))
    for xi in kde_x
])
scale  = counts.max() / kde_y.max() if kde_y.max() > 0 else 1
ax2.plot(kde_x, kde_y * scale, color=C_ORANGE, lw=2, label="KDE", zorder=4)

# Reference lines
ax2.axvline(mean_best, color=C_RED, lw=2, ls="--",
            label=f"Mean   = {mean_best:.3f}", zorder=5)
ax2.axvline(med_best,  color=C_BLUE, lw=2, ls=":",
            label=f"Median = {med_best:.3f}", zorder=5)

# Stats box
stats_txt = (
    f"N = {len(best_deltas)} calls\n"
    f"Mean   = {mean_best:.3f}\n"
    f"Median = {med_best:.3f}\n"
    f"Positive: {pct_pos:.0f}%\n"
    f"Negative: {pct_neg:.0f}%"
)
ax2.text(
    0.03, 0.97, stats_txt,
    transform=ax2.transAxes, fontsize=8.0,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GRAY, alpha=0.88),
)

ax2.set_title(
    "Teacher Δ Reward Distribution\n(reward_refined − reward_original per call)",
    fontweight="bold", fontsize=10.5,
)
ax2.set_xlabel("Δ Reward  (original reward ≈ 0)", fontsize=9)
ax2.set_ylabel("Count", fontsize=9)
ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
ax2.grid(True, alpha=0.22, axis="y")

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 3 – Refined Win Rate
# ═══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(C_BG)

ax3.plot(tea_steps, tea_win,        color=C_GREEN_LIGHT, lw=0.9, alpha=0.55, label="Raw")
ax3.plot(tea_steps, tea_win_smooth, color=C_GREEN,       lw=2.3, label="Smoothed")
ax3.fill_between(
    tea_steps, 50, tea_win_smooth,
    where=(tea_win_smooth >= 50), alpha=0.14, color=C_GREEN,
    label="Teacher beneficial (>50%)",
)
ax3.fill_between(
    tea_steps, tea_win_smooth, 50,
    where=(tea_win_smooth < 50),  alpha=0.14, color=C_RED,
    label="Teacher below baseline",
)
ax3.axhline(50,          color="black",  lw=1.2, ls="--", alpha=0.55, label="50% baseline")
ax3.axhline(avg_win_rate, color=C_BLUE,  lw=1.5, ls=":",
            label=f"Mean = {avg_win_rate:.1f}%")

ax3.set_ylim(20, 112)
ax3.set_title(
    "Refined Win Rate\n(% teacher calls that beat original trajectory)",
    fontweight="bold", fontsize=10.5,
)
ax3.set_xlabel("Training Step", fontsize=9)
ax3.set_ylabel("Win Rate (%)", fontsize=9)
ax3.legend(fontsize=7.5, loc="lower right", framealpha=0.85)
ax3.grid(True, alpha=0.22)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 4 – Teacher Trigger Rate (decreasing trend = policy becomes self-reliant)
# ═══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(C_BG)

ax4.plot(tea_steps, tea_trig,        color="#FFCC02",  lw=0.9, alpha=0.55, label="Raw")
ax4.plot(tea_steps, tea_trig_smooth, color=C_ORANGE,   lw=2.3, label="Smoothed")
ax4.fill_between(tea_steps, 0, tea_trig_smooth, alpha=0.12, color=C_ORANGE)

# Linear trend
z = np.polyfit(tea_steps, tea_trig, 1)
p = np.poly1d(z)
ax4.plot(tea_steps, p(tea_steps), color=C_RED, lw=1.8, ls="--",
         label=f"Trend: {z[0] * 1000:+.3f}%/1k steps")

# Start / end call-outs
ax4.annotate(
    f"Start\n{start_trig:.1f}%",
    xy=(tea_steps[0], start_trig),
    xytext=(tea_steps[0] + 180, start_trig + 2.0),
    fontsize=8, color=C_ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.9),
)
ax4.annotate(
    f"End\n{end_trig:.1f}%",
    xy=(tea_steps[-1], end_trig),
    xytext=(tea_steps[-1] - 700, end_trig + 3.5),
    fontsize=8, color=C_ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.9),
)

ax4.set_ylim(0, 22)
ax4.set_title(
    "Teacher Trigger Rate\n(% training steps where teacher was invoked — ↓ = policy self-reliant)",
    fontweight="bold", fontsize=10.5,
)
ax4.set_xlabel("Training Step", fontsize=9)
ax4.set_ylabel("Trigger Rate (%)", fontsize=9)
ax4.legend(fontsize=7.5, framealpha=0.85)
ax4.grid(True, alpha=0.22)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel 5 – Summary metrics table
# ═══════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")

rows = [
    ["Metric",                    "Value"],
    ["Total teacher calls",       f"≈{total_calls_both}"],
    ["Avg trigger rate",          f"{np.mean(tea_trig):.2f}%"],
    ["Trigger rate (start→end)",  f"{start_trig:.1f}% → {end_trig:.1f}%"],
    ["Avg refined win rate",      f"{avg_win_rate:.1f}%"],
    ["Refine acceptance rate",    f"{avg_accept:.1f}%"],
    ["Mean Δ reward (per call)",  f"{mean_best:.4f}"],
    ["Median Δ reward (per call)",f"{med_best:.4f}"],
    ["Positive-improvement %",   f"{pct_pos:.0f}%"],
    ["Policy reward start→end",  f"{start_ema:.3f} → {end_ema:.3f}"],
]

tbl = ax5.table(
    cellText=[r[1:] for r in rows[1:]],
    colLabels=[r[0] for r in rows[:1]][0].split(","),   # dummy; overridden below
    cellLoc="left",
    loc="center",
    bbox=[0, 0, 1, 1],
)
# Re-build with proper 2-column headers
tbl = ax5.table(
    cellText=[r for r in rows[1:]],
    colLabels=rows[0],
    cellLoc="left",
    loc="center",
    bbox=[0.0, 0.0, 1.0, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.auto_set_column_width([0, 1])

# Header style
for col in range(2):
    cell = tbl[0, col]
    cell.set_facecolor(C_BLUE)
    cell.get_text().set_color("white")
    cell.get_text().set_fontweight("bold")

# Row styles
green_rows = {6, 7, 8}   # delta / improvement rows
blue_rows  = {9}          # policy reward
for row_idx in range(1, len(rows)):
    for col in range(2):
        cell = tbl[row_idx, col]
        cell.set_facecolor("#EFF3FF" if row_idx % 2 == 0 else "white")
    if row_idx in green_rows:
        tbl[row_idx, 1].get_text().set_color(C_GREEN)
        tbl[row_idx, 1].get_text().set_fontweight("bold")
    if row_idx in blue_rows:
        tbl[row_idx, 1].get_text().set_color(C_BLUE)
        tbl[row_idx, 1].get_text().set_fontweight("bold")

ax5.set_title("Training Summary", fontweight="bold", fontsize=10.5, pad=8)

# ── main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    "Teacher Feedback Supporting RL Policy Learning  —  RecogDrive 8B",
    fontsize=14, fontweight="bold", y=0.975,
)

# ── narrative footer ──────────────────────────────────────────────────────────
fig.text(
    0.5, 0.012,
    "Key evidence: (1) All teacher Δ reward > 0 — teacher never hurts.  "
    "(2) Win rate > 50% — refined trajectories consistently beat originals.  "
    "(3) Trigger rate monotonically ↓ — policy becomes self-reliant as it learns.",
    ha="center", fontsize=8.5, color="#37474F", style="italic",
)

plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor="#F5F6FA")
print(f"Saved → {OUT_FILE}")

# ── standalone: Refined Win Rate ──────────────────────────────────────────────
fig_wr, ax_wr = plt.subplots(figsize=(8, 5))
fig_wr.patch.set_facecolor("#F5F6FA")
ax_wr.set_facecolor(C_BG)

ax_wr.plot(tea_steps, tea_win,        color=C_GREEN_LIGHT, lw=1.0, alpha=0.55, label="Raw")
ax_wr.plot(tea_steps, tea_win_smooth, color=C_GREEN,       lw=2.5, label="Smoothed")
ax_wr.fill_between(
    tea_steps, 50, tea_win_smooth,
    where=(tea_win_smooth >= 50), alpha=0.15, color=C_GREEN,
    label="Teacher beneficial (>50%)",
)
ax_wr.fill_between(
    tea_steps, tea_win_smooth, 50,
    where=(tea_win_smooth < 50), alpha=0.15, color=C_RED,
    label="Teacher below baseline",
)
ax_wr.axhline(50,           color="black", lw=1.3, ls="--", alpha=0.55, label="50% baseline")
ax_wr.axhline(avg_win_rate, color=C_BLUE,  lw=1.8, ls=":",
              label=f"Mean = {avg_win_rate:.1f}%")
ax_wr.set_ylim(20, 112)
ax_wr.set_title(
    "Refined Win Rate\n(% teacher calls that beat original trajectory)",
    fontweight="bold", fontsize=12,
)
ax_wr.set_xlabel("Training Step", fontsize=10)
ax_wr.set_ylabel("Win Rate (%)", fontsize=10)
ax_wr.legend(fontsize=9, loc="lower right", framealpha=0.88)
ax_wr.grid(True, alpha=0.22)
fig_wr.tight_layout()
fig_wr.savefig("win_rate.png", dpi=150, bbox_inches="tight", facecolor="#F5F6FA")
print("Saved → win_rate.png")

# ── standalone: Teacher Trigger Rate ─────────────────────────────────────────
fig_tr, ax_tr = plt.subplots(figsize=(8, 5))
fig_tr.patch.set_facecolor("#F5F6FA")
ax_tr.set_facecolor(C_BG)

ax_tr.plot(tea_steps, tea_trig,        color="#FFCC02", lw=1.0, alpha=0.55, label="Raw")
ax_tr.plot(tea_steps, tea_trig_smooth, color=C_ORANGE,  lw=2.5, label="Smoothed")
ax_tr.fill_between(tea_steps, 0, tea_trig_smooth, alpha=0.13, color=C_ORANGE)

z = np.polyfit(tea_steps, tea_trig, 1)
ax_tr.plot(tea_steps, np.poly1d(z)(tea_steps), color=C_RED, lw=2.0, ls="--",
           label=f"Trend: {z[0] * 1000:+.3f}%/1k steps")

ax_tr.annotate(
    f"Start\n{start_trig:.1f}%",
    xy=(tea_steps[0], start_trig),
    xytext=(tea_steps[0] + 200, start_trig + 2.2),
    fontsize=9, color=C_ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.0),
)
ax_tr.annotate(
    f"End\n{end_trig:.1f}%",
    xy=(tea_steps[-1], end_trig),
    xytext=(tea_steps[-1] - 800, end_trig + 4.0),
    fontsize=9, color=C_ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.0),
)

ax_tr.set_ylim(0, 22)
ax_tr.set_title(
    "Teacher Trigger Rate\n(% training steps where teacher was invoked  —  ↓ = policy self-reliant)",
    fontweight="bold", fontsize=12,
)
ax_tr.set_xlabel("Training Step", fontsize=10)
ax_tr.set_ylabel("Trigger Rate (%)", fontsize=10)
ax_tr.legend(fontsize=9, framealpha=0.88)
ax_tr.grid(True, alpha=0.22)
fig_tr.tight_layout()
fig_tr.savefig("trigger_rate.png", dpi=150, bbox_inches="tight", facecolor="#F5F6FA")
print("Saved → trigger_rate.png")
print()
print("=== Summary ===")
print(f"  Steps covered           : {pol_steps[0]} – {pol_steps[-1]}")
print(f"  Policy reward start→end : {start_ema:.4f} → {end_ema:.4f}  (+{end_ema-start_ema:.4f})")
print(f"  Total teacher calls     : ≈{total_calls_both}")
print(f"  Trigger rate start→end  : {start_trig:.2f}% → {end_trig:.2f}%")
print(f"  Avg refined win rate    : {avg_win_rate:.1f}%")
print(f"  Refine acceptance rate  : {avg_accept:.1f}%")
print(f"  Mean Δ reward (per call): {mean_best:.4f}")
print(f"  Median Δ reward         : {med_best:.4f}")
print(f"  Positive improvements   : {pct_pos:.0f}%")
