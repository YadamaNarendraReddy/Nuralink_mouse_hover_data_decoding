# smart_check.py
# Three ways to use:
#   1) Manual guess:
#        python smart_check.py "YOURBESTGUESS"
#      (Use single quotes in PowerShell if you include braces: 'Z{EO}N{E3}')
#   2) Heavy AUTO search:
#        python smart_check.py --auto --top 6 --beam 5000 --cmax 200000
#   3) Guided LABEL mode (recommended to force convergence):
#        python smart_check.py --label
#      Type each panel (ALL CAPS). For uncertain letters, type braces, e.g. {EO}, {I1L}.

import sys
from pathlib import Path
from hashlib import sha256
from itertools import product

EXPECTED_HASH = "8acb1410c5af0ff76da758b9a0178c8efe34ba3f9d80417c849b3f3911799586"
SALT          = "d248fac4a4e8a0460e2b3f87ba6ee455"

CONFUSIONS = {
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
ALPHANUM = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

def salted_hash(s): return sha256((s + SALT).encode()).hexdigest()
def matches(s):     return salted_hash(s) == EXPECTED_HASH

def expand_groups(p: str):
    out=[""]; i=0
    while i<len(p):
        ch=p[i]
        if ch in "{[":
            close = "}" if ch=="{" else "]"
            j=p.find(close,i+1)
            if j==-1:
                out=[s+ch for s in out]; i+=1; continue
            choices=list(p[i+1:j])
            out=[s+c for s in out for c in choices]
            i=j+1
        else:
            out=[s+ch for s in out]; i+=1
    return out

def base_variants(raw: str):
    raw = raw.strip().upper()
    if not raw: return set()
    base=set()
    for g in expand_groups(raw):
        base.add(g)
        base.add(g.replace(" ",""))
        base.add("".join(ch for ch in g if ch.isalnum()))
    more=set()
    for b in list(base):
        more.add(b[::-1])
        more.add(b.replace("_",""))
        more.add(b.replace("-",""))
    base |= more
    return base

def gen_confusions(s: str, cap: int=250000):
    pools=[CONFUSIONS.get(ch,[ch]) for ch in s]
    # guard explosion
    tot=1
    for p in pools:
        tot*=len(p)
        if tot>cap: break
    for prod_ in product(*pools):
        yield "".join(prod_)

# ---------------- AUTO mode ----------------
def auto_solve_from_csv(top_k=6, beam_w=5000, cmax=200000):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2, io

    CSV = Path("mouse_velocities.csv")
    if not CSV.exists():
        print("[ERROR] CSV not found:", CSV.resolve()); return

    # import and enforce velocity_y = 0
    df = pd.read_csv(CSV)
    dx = df["velocity_x"].astype(float).to_numpy()
    dy = np.zeros_like(dx)

    def movavg(a,k):
        if k<=1: return a
        pad=np.pad(a,(k//2,k-1-k//2),"edge")
        return np.convolve(pad,np.ones(k)/k,"valid")

    def build_positions(dx,dy):
        X=np.cumsum(dx).astype(float); Y=np.cumsum(dy).astype(float)
        X-=X.mean(); Y-=Y.mean()
        s=max(np.ptp(X),np.ptp(Y))
        if s>0: X,Y=X/s,Y/s
        return X,Y

    def pca_rotate(X,Y):
        M=np.vstack([X,Y]).T
        M-=M.mean(0)
        _,_,Vt=np.linalg.svd(M,full_matrices=False)
        Mr=M@Vt.T
        return Mr[:,0],Mr[:,1]

    def segment_strokes(X,Y,SPEED_WIN=9,STOP_FRAC=0.55,MIN_LEN=24,STOP_TRIM=6):
        dX=np.diff(X,prepend=X[0]); dY=np.diff(Y,prepend=Y[0])
        sp=np.hypot(dX,dY); sp_s=movavg(sp,SPEED_WIN)
        mx=float(np.nanmax(sp_s)) if np.isfinite(sp_s).any() else 1.0
        thr=STOP_FRAC*(mx if mx>0 else 1.0)
        stops=sp_s<thr
        segs=[]; start=None
        for i,st in enumerate(stops):
            if start is None and not st: start=max(0,i-STOP_TRIM)
            elif start is not None and st:
                end=max(start,i-STOP_TRIM)
                if end-start>=MIN_LEN: segs.append((start,end))
                start=None
        if start is not None and len(X)-start>=MIN_LEN:
            segs.append((start,len(X)-1))
        return segs

    def letters_by_time(segs,TIME_GAP=12):
        if not segs: return []
        segs=sorted(segs,key=lambda ab:ab[0])
        groups=[[segs[0]]]
        for (ps,pe),(cs,ce) in zip(segs,segs[1:]):
            if cs-pe>=TIME_GAP: groups.append([(cs,ce)])
            else: groups[-1].append((cs,ce))
        return groups

    def group_bbox(X,Y,g):
        s=min(a for (a,b) in g); e=max(b for (a,b) in g)
        xs,ys=X[s:e+1],Y[s:e+1]
        return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

    def merge_adjacent_by_x(X,Y,groups,OVER=0.20,GAP=0.50):
        if not groups: return []
        widths=[]
        for g in groups:
            for (s,e) in g:
                xs=X[s:e+1]; widths.append(float(xs.max()-xs.min()))
        med_w=float(np.median(widths)) if widths else 0.1
        merged=[groups[0]]
        for g in groups[1:]:
            prev=merged[-1]
            minx1,maxx1,*_=group_bbox(X,Y,prev)
            minx2,maxx2,*_=group_bbox(X,Y,g)
            w1=maxx1-minx1; w2=maxx2-minx2
            overlap=min(maxx1,maxx2)-max(minx1,minx2)
            gap=max(0.0,minx2-maxx1)
            if overlap>=OVER*min(w1,w2) or gap<=GAP*med_w:
                merged[-1]=prev+g
            else:
                merged.append(g)
        return merged

    # build positions and letters
    X,Y = build_positions(dx, -dy)   # invert Y for natural reading
    Xs,Ys = movavg(X,9), movavg(Y,9)
    Xr,Yr = pca_rotate(Xs,Ys)
    segs  = segment_strokes(Xr,Yr)
    letters = merge_adjacent_by_x(Xr,Yr, letters_by_time(segs))

    # templates for A–Z,0–9
    def render_template(ch, size=96):
        img=np.zeros((size,size), np.uint8)
        font=cv2.FONT_HERSHEY_SIMPLEX; scale,thick=2.0,4
        (w,h),base=cv2.getTextSize(ch,font,scale,thick)
        x=(size-w)//2; y=(size+h)//2
        cv2.putText(img,ch,(x,y),font,scale,255,thick,cv2.LINE_AA)
        return cv2.Canny(img,50,150)

    templs = {ch: render_template(ch) for ch in ALPHANUM}

    def crop_edges(xs,ys,size=96):
        xmin,xmax,ymin,ymax = xs.min(),xs.max(),ys.min(),ys.max()
        pad = 0.15*max(xmax-xmin,ymax-ymin) if max(xmax-xmin,ymax-ymin)>0 else 0.2
        fig=plt.figure(figsize=(2.0,2.0))
        ax=fig.add_axes([0,0,1,1])
        ax.plot(xs,ys,lw=2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(xmin-pad,xmax+pad); ax.set_ylim(ymin-pad,ymax+pad)
        buf=io.BytesIO(); fig.savefig(buf, format="png", dpi=220, bbox_inches="tight"); plt.close(fig)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
        return cv2.Canny(img,50,150)

    def sim(a,b):
        a=a.astype(np.float32); b=b.astype(np.float32)
        a=(a-a.mean())/(a.std()+1e-6); b=(b-b.mean())/(b.std()+1e-6)
        return float((a*b).mean())

    # per-letter top-K
    per_letter=[]
    for g in letters:
        s=min(a for (a,b) in g); e=max(b for (a,b) in g)
        xs,ys = Xr[s:e+1], Yr[s:e+1]
        ed    = crop_edges(xs,ys)
        sc=[(ch, sim(ed, templs[ch])) for ch in ALPHANUM]
        sc.sort(key=lambda t:t[1], reverse=True)
        per_letter.append([c for c,_ in sc[:max(1,top_k)]])

    # beam search
    beam=[("",0.0)]
    for choices in per_letter:
        nxt=[]
        for pref,score in beam:
            for ch in choices:
                nxt.append((pref+ch, score+1.0))
        nxt.sort(key=lambda t:t[1], reverse=True)
        beam=nxt[:max(1,beam_w)]

    seen=set(); tried=0
    for cand,_ in beam:
        # direct forms
        for c in {cand, cand.replace(" ",""), cand[::-1]}:
            if c in seen: continue
            seen.add(c); tried+=1
            if matches(c):
                print(f"✅ MATCH: {c}   (auto mode, {tried} tries)")
                print(f'Run:  python check_answer.py {c if " " not in c else f"{c!r}"}')
                return
        # confusion expansion
        for c2 in gen_confusions(cand, cap=int(cmax)):
            if c2 in seen: continue
            seen.add(c2); tried+=1
            if matches(c2):
                print(f"✅ MATCH: {c2}   (auto mode, {tried} tries)")
                print(f'Run:  python check_answer.py {c2 if " " not in c2 else f"{c2!r}"}')
                return

    print(f"❌ Auto mode: no match after {tried} candidates. Try:  python smart_check.py --label")

# ---------------- LABEL mode ----------------
def label_mode():
    import numpy as np, pandas as pd
    CSV = Path("mouse_velocities.csv")
    if not CSV.exists():
        print("[ERROR] CSV not found:", CSV.resolve()); return

    # The “letters_*_NUMBERED.png” were created by reconstruct.py.
    # Open the yinverted version for easy reading and type letters left→right.
    print("\nOpen  letters_yinverted_NUMBERED.png  (or  letters_orig_NUMBERED.png).")
    print("Type each panel (ALL CAPS). If you’re unsure, type braces:  {EO}  {I1L}  {S5}  {Z2}  ...")
    print("Press Enter after each letter/panel.\n")

    # Ask how many panels you see (to keep in sync)
    try:
        n = int(input("How many panels do you see? ").strip())
    except Exception:
        print("Please enter an integer."); return

    parts=[]
    for i in range(1, n+1):
        ans = input(f"Letter {i:02d}: ").strip().upper()
        parts.append(ans if ans else "?")

    raw = "".join(parts)
    print("\nYour pattern:", raw)

    tried=0; seen=set()
    for cand in base_variants(raw):
        tried+=1
        if matches(cand):
            print(f"\n✅ MATCH: {cand}   (tried {tried})")
            print(f'Run:  python check_answer.py {cand if " " not in cand else f"{cand!r}"}')
            return
    for b in base_variants(raw):
        for cand in gen_confusions(b):
            if cand in seen: continue
            seen.add(cand); tried+=1
            if matches(cand):
                print(f"\n✅ MATCH: {cand}   (tried {tried})")
                print(f'Run:  python check_answer.py {cand if " " not in cand else f"{cand!r}"}')
                return

    print(f"\n❌ No match after {tried} sensible variants.")
    print("Re-check ambiguous panels and try again (brace uncertain letters).")

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--auto":
        top  = 6
        beam = 5000
        cmax = 200000
        if "--top"  in sys.argv:  top  = int(sys.argv[sys.argv.index("--top")+1])
        if "--beam" in sys.argv:  beam = int(sys.argv[sys.argv.index("--beam")+1])
        if "--cmax" in sys.argv:  cmax = int(sys.argv[sys.argv.index("--cmax")+1])
        auto_solve_from_csv(top_k=top, beam_w=beam, cmax=cmax)
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "--label":
        label_mode(); return

    if len(sys.argv) < 2:
        print('Usage:')
        print('  python smart_check.py "YOURBESTGUESS"')
        print('  python smart_check.py --auto [--top 6 --beam 5000 --cmax 200000]')
        print('  python smart_check.py --label')
        return

    # Manual guess
    seed = sys.argv[1].strip().upper()
    tried=0; seen=set()

    for cand in base_variants(seed):
        tried+=1
        if matches(cand):
            print(f"✅ MATCH: {cand}   (tried {tried})")
            print(f'Run:  python check_answer.py {cand if " " not in cand else f"{cand!r}"}')
            return
    for b in base_variants(seed):
        for cand in gen_confusions(b):
            if cand in seen: continue
            seen.add(cand); tried+=1
            if matches(cand):
                print(f"✅ MATCH: {cand}   (tried {tried})")
                print(f'Run:  python check_answer.py {cand if " " not in cand else f"{cand!r}"}')
                return

    print(f"❌ No match after {tried} sensible variants.")
    print("Try:  python smart_check.py --auto   or   python smart_check.py --label")

if __name__ == "__main__":
    main()
