# reconstruct.py
# Rebuild letters from mouse_velocities.csv.
# IMPORTANT: per challenge note, we must set velocity_y = 0 on import.

from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path("mouse_velocities.csv")

# --- Tunables you can nudge if letters still glue together ---
SPEED_WIN     = 9        # moving-average window for speed
STOP_FRAC     = 0.55     # fraction of max speed that counts as a "stop"
STOP_TRIM     = 6        # trim samples around each stop boundary
MIN_LEN       = 24       # minimum segment length to keep (in samples)
TIME_GAP      = 12       # gap in samples that separates two letters
OVERLAP_FRAC  = 0.20     # if x-overlap between groups >= 20%, merge into one letter
GAP_FRAC      = 0.50     # or if x-gap <= 0.5*median-width, merge into one letter
SMOOTH_POS    = 9        # smoothing window for positions

# --------------------------------------------------------------

def movavg(a, k):
    if k <= 1: return a.copy()
    pad = np.pad(a, (k//2, k-1-k//2), mode="edge")
    return np.convolve(pad, np.ones(k)/k, mode="valid")

def load_positions(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    dx = df["velocity_x"].astype(float).to_numpy()

    # PER INSTRUCTIONS: velocity_y must be treated as zero
    dy = np.zeros_like(dx)

    # integrate to positions
    X = np.cumsum(dx).astype(float)
    Y = np.cumsum(dy).astype(float)

    # de-mean and normalize scale
    X -= X.mean()
    Y -= Y.mean()
    s = max(np.ptp(X), np.ptp(Y))
    if s > 0:
        X /= s
        Y /= s
    return X, Y

def pca_rotate(X, Y):
    M = np.vstack([X, Y]).T
    M -= M.mean(axis=0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    Mr = M @ Vt.T
    return Mr[:, 0], Mr[:, 1]

def segment_strokes(X, Y):
    dX = np.diff(X, prepend=X[0])
    dY = np.diff(Y, prepend=Y[0])
    sp = np.hypot(dX, dY)
    sp_s = movavg(sp, SPEED_WIN)

    mx = float(np.nanmax(sp_s)) if np.isfinite(sp_s).any() else 1.0
    thr = STOP_FRAC * (mx if mx > 0 else 1.0)
    stops = sp_s < thr

    segs = []
    start = None
    for i, st in enumerate(stops):
        if start is None and not st:
            start = max(0, i - STOP_TRIM)
        elif start is not None and st:
            end = max(start, i - STOP_TRIM)
            if end - start >= MIN_LEN:
                segs.append((start, end))
            start = None
    if start is not None and len(X) - start >= MIN_LEN:
        segs.append((start, len(X) - 1))
    return segs

def group_by_time(segs):
    """Group consecutive stroke segments into letters based on TIME_GAP."""
    if not segs: return []
    segs = sorted(segs, key=lambda ab: ab[0])
    groups = [[segs[0]]]
    for (ps, pe), (cs, ce) in zip(segs, segs[1:]):
        if cs - pe >= TIME_GAP:
            groups.append([(cs, ce)])
        else:
            groups[-1].append((cs, ce))
    return groups

def group_bbox(X, Y, g):
    s = min(a for (a, b) in g)
    e = max(b for (a, b) in g)
    xs, ys = X[s:e+1], Y[s:e+1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

def merge_by_x(X, Y, groups):
    """Merge groups that overlap strongly in X or are very close."""
    if not groups: return []
    widths = []
    for g in groups:
        for (s, e) in g:
            xs = X[s:e+1]
            widths.append(float(xs.max() - xs.min()))
    med_w = float(np.median(widths)) if widths else 0.1

    merged = [groups[0]]
    for g in groups[1:]:
        prev = merged[-1]
        minx1, maxx1, *_ = group_bbox(X, Y, prev)
        minx2, maxx2, *_ = group_bbox(X, Y, g)
        w1 = maxx1 - minx1
        w2 = maxx2 - minx2
        overlap = min(maxx1, maxx2) - max(minx1, minx2)
        gap = max(0.0, minx2 - maxx1)

        if overlap >= OVERLAP_FRAC * min(w1, w2) or gap <= GAP_FRAC * med_w:
            merged[-1] = prev + g
        else:
            merged.append(g)
    return merged

def render_whole(X, Y, title, out_png):
    plt.figure(figsize=(11, 3))
    ax = plt.gca()
    ax.plot(X, Y, lw=1.8)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def draw_letter(ax, X, Y, g, color=None):
    s = min(a for (a, b) in g)
    e = max(b for (a, b) in g)
    if color:
        ax.plot(X[s:e+1], Y[s:e+1], lw=2.0, color=color)
    else:
        ax.plot(X[s:e+1], Y[s:e+1], lw=2.0)

def render_letters_numbered(X, Y, groups, title, out_png):
    # Left→right order
    ordered = sorted(groups, key=lambda g: group_bbox(X, Y, g)[0])
    W = len(ordered)
    plt.figure(figsize=(min(max(6, W), 18), 2.4))
    ax = plt.gca()
    palette = plt.cm.tab20.colors

    for i, g in enumerate(ordered, 1):
        draw_letter(ax, X, Y, g, palette[(i-1) % len(palette)])
        # Put small index above the letter bbox
        minx, maxx, miny, maxy = group_bbox(X, Y, g)
        ax.text((minx + maxx)/2, maxy + 0.08, f"{i}", ha="center", va="bottom", fontsize=9)

    ax.set_aspect("equal")
    ax.set_title(f"{title} — {W} letters")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    return W

def suggest_spaces(X, Y, groups, orientation, out_png):
    ordered = sorted(groups, key=lambda g: group_bbox(X, Y, g)[0])
    xs = [group_bbox(X, Y, g)[:2] for g in ordered]
    widths = [mx - mn for (mn, mx) in xs]
    med_w = float(np.median(widths)) if widths else 0.0

    gaps = []
    for (mn1, mx1), (mn2, mx2) in zip(xs, xs[1:]):
        gaps.append(mn2 - mx1)

    # render
    plt.figure(figsize=(10, 2.2))
    ax = plt.gca()
    for i, g in enumerate(ordered, 1):
        draw_letter(ax, X, Y, g)
        mn, mx, my, My = group_bbox(X, Y, g)
        ax.text((mn+mx)/2, My + 0.085, f"{i}", ha="center", va="bottom", fontsize=9)

    ax.set_aspect("equal")
    ax.set_title("Letters with potential spaces (based on X-gaps)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()

    print("\n=== Suggested spaces (based on X-gaps) ===")
    print(f"Orientation: {orientation}")
    print(f"Number of letters: {len(ordered)}")
    # very simple heuristic
    if not gaps:
        print("No gaps — single letter (unexpected).")
        return
    big = [g for g in gaps if g > 0.8 * med_w]
    if big:
        print("Large gaps found (could be spaces) at letter indices between:")
        cum = 0
        for idx, g in enumerate(gaps, 1):
            if g > 0.8 * med_w:
                print(f"  {idx} | {idx+1}  (gap ~ {g:.2f} vs median width {med_w:.2f})")
    else:
        print("No large gaps detected — likely a single word.")

def main():
    X, Y = load_positions(CSV)

    # Smooth positions slightly for stability
    Xs = movavg(X, SMOOTH_POS)
    Ys = movavg(Y, SMOOTH_POS)

    # PCA for canonical orientation
    Xr, Yr = pca_rotate(Xs, Ys)

    # We’ll export both orig and Y-inverted for human reading
    for orient, (XX, YY) in [("orig", (Xr, Yr)), ("yinverted", (Xr, -Yr))]:
        segs   = segment_strokes(XX, YY)
        groups = merge_by_x(XX, YY, group_by_time(segs))

        # whole path
        render_whole(XX, YY, f"Whole ({orient})", f"whole_{orient}.png")
        # numbered letters
        count = render_letters_numbered(XX, YY, groups, f"Letters by time ({orient})", f"letters_{orient}_NUMBERED.png")
        # suggestion figure
        suggest_spaces(XX, YY, groups, orient, f"letters_spaces_{orient}.png")

        print(f"Saved whole_{orient}.png")
        print(f"Saved letters_{orient}_NUMBERED.png")
        print(f"Saved letters_spaces_{orient}.png")

if __name__ == "__main__":
    main()
