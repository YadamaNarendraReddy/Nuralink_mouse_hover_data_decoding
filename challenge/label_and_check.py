# label_and_check.py
# Reconstructs letters from mouse_velocities.csv (respecting "velocity_y must be zero" note),
# saves one PNG per letter, then asks you to TYPE the character for each panel (left->right).
# It then tries sensible variants (O<->0, I<->1, S<->5, Z<->2, B<->8, G<->6, T<->7, A<->4, E<->3, L<->1)
# and reports the FIRST exact match to the salted hash used by check_answer.py.

from pathlib import Path
import sys, shutil
from hashlib import sha256
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path("mouse_velocities.csv")
OUTDIR = Path("letters_crops_yinverted")  # per-letter crops (yinverted orientation)

# === constants from check_answer.py (unchanged) ===
EXPECTED_HASH = "8acb1410c5af0ff76da758b9a0178c8efe34ba3f9d80417c849b3f3911799586"
SALT          = "d248fac4a4e8a0460e2b3f87ba6ee455"

# ---- reconstruction tunables (good defaults) ----
POS_WIN   = 9
SPEED_WIN = 9
STOP_FRAC = 0.48
MIN_LEN   = 30
STOP_TRIM = 6
TIME_GAP  = 8
MERGE_X_OVERLAP_FRAC = 0.20
MERGE_X_GAP_FRAC     = 0.50
# -------------------------------------------------

CONF = {
    "O":["O","0"], "0":["0","O"],
    "I":["I","1","L"], "1":["1","I","L"],
    "S":["S","5"], "5":["5","S"],
    "Z":["Z","2"], "2":["2","Z"],
    "B":["B","8"], "8":["8","B"],
    "G":["G","6"], "6":["6","G"],
    "T":["T","7"], "7":["7","T"],
    "A":["A","4"], "4":["4","A"],
    "E":["E","3"], "3":["3","E"],
    "L":["L","1"],
}

def salted_hash(s: str) -> str:
    return sha256((s + SALT).encode()).hexdigest()

def matches(s: str) -> bool:
    return salted_hash(s) == EXPECTED_HASH

def movavg(a, k=7):
    if k<=1: return a
    pad = np.pad(a, (k//2, k-1-k//2), mode="edge")
    return np.convolve(pad, np.ones(k)/k, mode="valid")

def load_deltas_zeroY():
    if not CSV.exists():
        print("[ERROR] CSV not found at", CSV.resolve()); sys.exit(1)
    df = pd.read_csv(CSV)
    dx = df["velocity_x"].astype(float).to_numpy()
    dy = np.zeros_like(dx)  # respect the instruction: velocity_y = 0 on import
    return dx, dy

def build_positions(dx, dy):
    X = np.cumsum(dx).astype(float); Y = np.cumsum(dy).astype(float)
    X -= X.mean(); Y -= Y.mean()
    s = max(np.ptp(X), np.ptp(Y))
    if s>0: X, Y = X/s, Y/s
    return X, Y

def pca_rotate(X, Y):
    M = np.vstack([X,Y]).T; M -= M.mean(0)
    _,_,Vt = np.linalg.svd(M, full_matrices=False)
    Mr = M @ Vt.T
    return Mr[:,0], Mr[:,1]

def segment_strokes(X, Y):
    dX = np.diff(X, prepend=X[0]); dY = np.diff(Y, prepend=Y[0])
    sp = np.hypot(dX,dY); sp_s = movavg(sp, SPEED_WIN)
    mx = float(np.nanmax(sp_s)) if np.isfinite(sp_s).any() else 1.0
    thr = STOP_FRAC * (mx if mx>0 else 1.0)
    stops = sp_s < thr
    segs=[]; start=None
    for i,st in enumerate(stops):
        if start is None and not st:
            start = max(0, i-STOP_TRIM)
        elif start is not None and st:
            end = max(start, i-STOP_TRIM)
            if end-start >= MIN_LEN: segs.append((start,end))
            start=None
    if start is not None and len(X)-start>=MIN_LEN:
        segs.append((start,len(X)-1))
    return segs

def letters_by_time(segs):
    if not segs: return []
    segs = sorted(segs, key=lambda ab: ab[0])
    groups=[[segs[0]]]
    for (ps,pe),(cs,ce) in zip(segs,segs[1:]):
        if cs-pe >= TIME_GAP: groups.append([(cs,ce)])
        else: groups[-1].append((cs,ce))
    return groups

def group_bbox(X,Y,g):
    s = min(a for (a,b) in g); e = max(b for (a,b) in g)
    xs,ys = X[s:e+1], Y[s:e+1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

def merge_adjacent_by_x(X,Y,groups):
    if not groups: return []
    widths=[]
    for g in groups:
        for (s,e) in g:
            xs = X[s:e+1]
            widths.append(float(xs.max()-xs.min()))
    med_w = float(np.median(widths)) if widths else 0.1
    merged=[groups[0]]
    for g in groups[1:]:
        prev=merged[-1]
        minx1,maxx1, *_ = group_bbox(X,Y,prev)
        minx2,maxx2, *_ = group_bbox(X,Y,g)
        w1=maxx1-minx1; w2=maxx2-minx2
        overlap = min(maxx1,maxx2) - max(minx1,minx2)
        gap = max(0.0, minx2 - maxx1)
        if overlap >= MERGE_X_OVERLAP_FRAC*min(w1,w2) or gap <= MERGE_X_GAP_FRAC*med_w:
            merged[-1] = prev + g
        else:
            merged.append(g)
    return merged

def save_letter_crops(X,Y,groups,outdir:Path):
    if outdir.exists(): shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i,g in enumerate(groups,1):
        s = min(a for (a,b) in g); e = max(b for (a,b) in g)
        xs,ys = X[s:e+1], Y[s:e+1]
        xmin,xmax,ymin,ymax = xs.min(), xs.max(), ys.min(), ys.max()
        dx,dy = xmax-xmin, ymax-ymin
        pad = 0.15*max(dx,dy) if max(dx,dy)>0 else 0.2
        plt.figure(figsize=(2.6,2.6))
        plt.plot(xs,ys,lw=2); plt.gca().set_aspect("equal"); plt.axis("off")
        plt.xlim(xmin-pad, xmax+pad); plt.ylim(ymin-pad, ymax+pad)
        plt.savefig(outdir/f"letter_{i:02d}.png", bbox_inches="tight", dpi=230)
        plt.close()
    print(f"Saved {len(groups)} letter images in: {outdir.resolve()}")

def reconstruct_and_export():
    dx,dy = load_deltas_zeroY()
    # y-inverted is typically easier to read (screen Y grows downward).
    X,Y = build_positions(dx, -dy)
    Xs,Ys = movavg(X,POS_WIN), movavg(Y,POS_WIN)
    Xr,Yr = pca_rotate(Xs,Ys)
    segs = segment_strokes(Xr,Yr)
    if not segs:
        print("[ERROR] No strokes detected. Raise STOP_FRAC or lower TIME_GAP.")
        sys.exit(2)
    letters = merge_adjacent_by_x(Xr,Yr, letters_by_time(segs))
    save_letter_crops(Xr,Yr,letters, OUTDIR)
    return len(letters)

def try_confusions(word):
    # breadth-first limited expansion
    agenda = [word]
    seen = set([word])
    out = []
    while agenda and len(out) < 100000:   # cap just in case
        s = agenda.pop(0)
        out.append(s)
        for i,ch in enumerate(s):
            if ch in CONF:
                for alt in CONF[ch]:
                    t = s[:i]+alt+s[i+1:]
                    if t not in seen:
                        seen.add(t)
                        agenda.append(t)
    return out

def main():
    n = reconstruct_and_export()
    # Prompt user for letters left->right:
    print("\nOpen this folder and read left->right by file name:")
    print(OUTDIR.resolve())
    chars = []
    for i in range(1, n+1):
        while True:
            v = input(f"Letter {i:02d} (A-Z/0-9; enter to leave blank): ").strip().upper()
            if v=="":
                print("Leave blank if unsure; we will try confusions later.")
                chars.append("")   # unknown allowed
                break
            if len(v)==1 and (v.isalnum() or v in "-_."):
                chars.append(v)
                break
            print("Please enter a single A-Z or 0-9.")
    base = "".join(ch if ch else "?" for ch in chars).upper()
    print("\nYour base entry:", base)

    # Replace '?' with 'O' first, then expand confusions:
    seed = base.replace("?", "O")
    candidates = try_confusions(seed)
    # Also try no-unknowns-by-zeros:
    candidates += try_confusions(base.replace("?", "0"))
    # And reversed, and no punctuation
    extras = []
    for c in list(candidates)[:2000]:
        extras.append(c[::-1])
        extras.append("".join(ch for ch in c if ch.isalnum()))
    candidates += extras

    tried = 0; seen=set()
    for c in candidates:
        if c in seen: continue
        seen.add(c); tried += 1
        if matches(c):
            print(f"\n✅ MATCH: {c}\n(Tried {tried} variants)")
            print("\nVerify with your checker:")
            print(f"    python check_answer.py {c if ' ' not in c else repr(c)}")
            return
        if tried % 5000 == 0:
            print(f"... tried {tried} variants without a match yet")

    print(f"\n❌ No match after {tried} variants.")
    print("Double-check a few ambiguous panels and re-run.")

if __name__ == "__main__":
    main()
