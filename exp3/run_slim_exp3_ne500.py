#!/usr/bin/env python3
"""
Experiment 3 stochastic extension SLiM runner — Ne=500 group
Main parameters: u=v=0.001, s=[0.005,0,-0.005], r12=0.01, r23=0.05, T=2000
Ne=500 diploid -> 2Ne=1000 haplotypes
Initial: p000=0.4*1000=400, p111=400, remaining 6 = 0.2/6*1000 ~ 33.3 each (rounded to 34,33,33,33,33,34, sum = 200)
100 reps, record every 10 generations (201 points), parallelism 2
Same seed convention as run_slim_exp3.py / run_slim_exp3_ne3000.py (1000 + 7i).
"""
import subprocess, numpy as np, os, time, multiprocessing as mp, sys

SLIM = os.environ.get('SLIM_BIN', '/usr/local/bin/slim')  # override: SLIM_BIN=/path/to/slim
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(_HERE, 'slim', 'slim_L3_v12.slim')
DATA = os.path.join(_HERE, 'data')
os.makedirs(DATA, exist_ok=True)

def run_one(args):
    seed, outfile = args
    cmd = [SLIM, '-s', str(seed)]
    cmd += ['-d', 'NE=500']
    cmd += ['-d', 'R12=0.01']
    cmd += ['-d', 'R23=0.05']
    cmd += ['-d', 'MU=0.001']
    cmd += ['-d', 'S1=0.01']
    cmd += ['-d', 'S2=0.0']
    cmd += ['-d', 'S3=-0.01']
    cmd += ['-d', 'H0=400', '-d', 'H1=34', '-d', 'H2=33', '-d', 'H3=33']
    cmd += ['-d', 'H4=33', '-d', 'H5=33', '-d', 'H6=34', '-d', 'H7=400']
    cmd += ['-d', 'TMAX=2000']
    cmd += ['-d', f"OUTFILE='{outfile}'"]
    cmd.append(SCRIPT)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if not os.path.exists(outfile):
        raise RuntimeError(f"SLiM failed seed={seed}: {res.stderr[-400:]}")
    return np.loadtxt(outfile)   # (201, 9): gen, h0..h7

if __name__ == '__main__':
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    fn = f'{DATA}/slim3_ne500.npy'
    if os.path.exists(fn):
        print("[skip] exists")
        sys.exit(0)
    t0 = time.time()
    tmpdir = f'{DATA}/tmp_ne500'
    os.makedirs(tmpdir, exist_ok=True)
    tasks = []
    for i in range(n_reps):
        f = f'{tmpdir}/rep_{i}.txt'
        tasks.append((1000 + i * 7, f))
    with mp.Pool(2) as pool:
        res = pool.map(run_one, tasks)
    arr = np.stack(res)   # (n_reps, 201, 9)
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
    np.save(fn, arr)
    h = arr[:, -1, 1:]
    print(f"[done] Ne=500 {n_reps} reps in {time.time()-t0:.0f}s, end mean h = {h.mean(axis=0).round(3)}", flush=True)
