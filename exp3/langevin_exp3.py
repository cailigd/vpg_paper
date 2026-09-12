#!/usr/bin/env python3
"""
Experiment 3 stochastic extension — Langevin SVPG (single-state VPG, with drift) vs SLiM comparison
v3.7.3 Eqs. 35/41: dp = (Jmut + Jrec + Jsel) dt + sqrt(M_FR(p)/Ne) dW
  M_FR(p) = diag(p) - p p^T  (inverse Shahshahani/Fisher-Rao metric)
Ne correspondence: SLiM diploid Ne_slim -> 2*Ne_slim haploid copies, VPG haploid Ne_vpg = 2*Ne_slim
  (SLiM Ne=1000 -> VPG Ne=2000; SLiM Ne=3000 -> VPG Ne=6000)
Univariate full speed (no 1/2 factor): SVPG is the classical equations + drift noise
"""
import numpy as np
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vpg_core_exp3 as vc

def sqrt_MFR(p):
    """sqrt(diag(p) - p p^T): analytic Householder structure, no eigendecomposition needed.

    M_FR = D(I - vv^T)D, where D = diag(sqrt(p)), v = sqrt(p),
    and ||v||^2 = Σp = 1 (guaranteed by probability normalization).
    Noise sampling N(0, M_FR): y = (I-vv^T)z = z - v(v·z), x = D y.
    Returns (v, D) for use by langevin_svpg, O(N) per step (original eigh was O(N³)).
    """
    v = np.sqrt(p)
    return v

def langevin_svpg(p0, M, U, radj, L, Ne, T, dt, seed=0, record_every=1000):
    """Single stochastic trajectory: Euler-Maruyama
    record_every: recording interval (in steps), default 1000 steps = 10 generations
    Returns: (gen, ps)  gen: (nrec,) recorded generations, ps: (nrec, 8)
    """
    rng = np.random.default_rng(seed)
    n = int(round(T / dt))
    nrec = n // record_every + 1
    ps = np.zeros((nrec, 8))
    gens = np.zeros(nrec)
    p = p0.copy()
    ps[0] = p
    idx = 1
    c = np.sqrt(dt / Ne)
    for k in range(1, n + 1):
        dp = (M @ p + vc.J_rec_proj(p, radj, L)
              + p * (U - np.dot(p, U)))
        p = p + dt * dp
        # drift noise: analytic sampling of N(0, M_FR) (Householder structure, O(N), no eigh needed)
        #   M_FR = D(I-vv^T)D, v=sqrt(p), ||v||²=1
        #   y = (I-vv^T)z = z - v(v·z),  x = D y = sqrt(p) ⊙ y
        v = sqrt_MFR(p)
        z = rng.standard_normal(8)
        y = z - v * (v @ z)
        p = p + c * (v * y)
        p = np.clip(p, 1e-15, 1.0)
        p /= p.sum()
        if k % record_every == 0:
            ps[idx] = p
            gens[idx] = k * dt
            idx += 1
    return gens[:idx], ps[:idx]

def run_mc(p0, M, U, radj, L, Ne, T, dt, n_reps, record_every=1000, seed0=0):
    """Run n_reps trajectories, return (gen, (n_reps, nrec, 8))"""
    gen0 = None
    allp = []
    for i in range(n_reps):
        g, ps = langevin_svpg(p0, M, U, radj, L, Ne, T, dt, seed=seed0 + i, record_every=record_every)
        if gen0 is None:
            gen0 = g
        allp.append(ps)
    return gen0, np.stack(allp)   # (n_reps, nrec, 8)

if __name__ == '__main__':
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
    # quick self-check: AF/SE of a single trajectory + 5 reps
    g, ps = run_mc(p0, M, U, radj, L, Ne=2000, T=2000, dt=0.01, n_reps=5, record_every=1000)
    print('gen points:', g[:3], '...', g[-1])
    af = np.array([[sum(pp[h] for h in range(8) if (h >> i) & 1) for i in range(3)] for pp in ps[-1]])
    print('Final AF (5 reps):', af.round(3))
    print('Final AF mean±SD:', af.mean(0).round(3), '±', af.std(0).round(3))
    print('OK')
