#!/usr/bin/env python3
"""
Experiment 1 SLiM runner — L=1 single-locus mutation-selection balance
Parameter matrix:
  Main matrix p0=0.01: mu in {3e-4, 1e-3, 3e-3} x s in {0.005, 0.01, 0.05}  = 9 groups
  Bidirectional convergence p0=0.99: (mu=1e-3, s=0.01), (mu=3e-3, s=0.01)  = 2 groups
Each group has 100 reps, Ne=1000 (2Ne=2000 haplotypes), T=2000, record every 10 generations (201 points)
Parallelism 2 (2 cores)
"""
import subprocess, numpy as np, os, time, multiprocessing as mp, sys

SLIM = os.environ.get('SLIM_BIN', '/usr/local/bin/slim')  # override: SLIM_BIN=/path/to/slim
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(_HERE, 'slim', 'slim_L1_v1.slim')
DATA = os.path.join(_HERE, 'data')
os.makedirs(DATA, exist_ok=True)

def run_one(args):
    """Run a single rep, returning the (gen, p) array"""
    seed, mu, s, p0, outfile = args
    N_A1 = int(round(2000 * p0))
    cmd = [SLIM, '-s', str(seed)]
    cmd += ['-d', 'NE=1000']
    cmd += ['-d', f'U={mu}']
    cmd += ['-d', f'V={mu}']
    cmd += ['-d', f'S={2*s}']
    cmd += ['-d', f'N_A1={N_A1}']
    cmd += ['-d', 'TMAX=2000']
    cmd += ['-d', f"OUTFILE='{outfile}'"]
    cmd.append(SCRIPT)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not os.path.exists(outfile):
        raise RuntimeError(f"SLiM failed seed={seed}: {r.stderr[-400:]}")
    data = np.loadtxt(outfile)
    return data  # (201, 2): gen, p

def run_group(mu, s, p0, n_reps, tag):
    """Run a group of n_reps, saving the npy (n_reps, 201, 2) array"""
    key = f"mu{str(mu).replace('.','p')}_s{str(s).replace('.','p')}_p0{str(p0).replace('.','p')}"
    fn = f'{DATA}/slim1_{key}.npy'
    if os.path.exists(fn):
        print(f"[skip] {key} already exists", flush=True)
        return
    t0 = time.time()
    tmpdir = f'{DATA}/tmp_{key}'
    os.makedirs(tmpdir, exist_ok=True)
    tasks = []
    for i in range(n_reps):
        f = f'{tmpdir}/rep_{i}.txt'
        tasks.append((1000 + i * 7, mu, s, p0, f))
    with mp.Pool(2) as pool:
        res = pool.map(run_one, tasks)
    arr = np.stack(res)  # (n_reps, 201, 2)
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
    np.save(fn, arr)
    # Summary: p0 check + final state
    print(f"[done] {key}: p0={arr[0,0,1]:.4f} p_end={arr[:, -1, 1].mean():.4f} "
          f"±{arr[:, -1, 1].std():.4f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mus = [3e-4, 1e-3, 3e-3]
    ss  = [0.005, 0.01, 0.05]
    # Main matrix p0=0.01
    for mu in mus:
        for s in ss:
            run_group(mu, s, 0.01, n_reps, 'main')
    # Bidirectional convergence p0=0.99
    run_group(1e-3, 0.01, 0.99, n_reps, 'back')
    run_group(3e-3, 0.01, 0.99, n_reps, 'back')
    print("ALL SLiM GROUPS DONE", flush=True)
