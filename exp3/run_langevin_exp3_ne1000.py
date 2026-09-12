#!/usr/bin/env python3
"""
Experiment 3 Langevin SVPG batch run — VPG Ne=1000 group (corresponding to SLiM Ne=500)
Ne correspondence: SLiM diploid Ne=500 -> 1000 haploid copies -> VPG haploid Ne=1000
100 reps, T=2000, dt=0.01, record every 10 generations (201 points)
Same seed convention as the other groups (seed0 = 1000).
"""
import numpy as np, os, time, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vpg_core_exp3 as vc
from langevin_exp3 import run_mc

DATA = os.path.join(_HERE, 'data')
os.makedirs(DATA, exist_ok=True)

mu = 0.001
s = np.array([0.005, 0.0, -0.005])
radj = np.array([0.01, 0.05])
L = 3
p0 = np.zeros(8)
p0[0] = 0.4; p0[7] = 0.4
for k in range(1, 7):
    p0[k] = 0.2 / 6
M = vc.mut_matrix(mu, mu)
U = vc.U_vec(s)

def run_group(Ne_vpg, tag, n_reps=100):
    fn = f'{DATA}/langevin3_ne{Ne_vpg}.npy'
    if os.path.exists(fn):
        print(f"[skip] {tag} exists")
        return
    t0 = time.time()
    g, allp = run_mc(p0, M, U, radj, L, Ne_vpg, 2000, 0.01, n_reps, seed0=1000)
    np.save(fn, allp)   # (n_reps, 201, 8)
    af = np.array([[sum(pp[h] for h in range(8) if (h >> i) & 1) for i in range(3)] for pp in allp[:, -1]])
    print(f"[done] {tag}: end AF mean={af.mean(0).round(3)} ±SD={af.std(0).round(3)} "
          f"({time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_group(1000, 'VPG Ne=1000 (SLiM Ne=500)', n_reps)
    print("LANGEVIN Ne=1000 DONE", flush=True)
