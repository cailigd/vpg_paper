#!/usr/bin/env python3
"""
Experiment 2 deterministic core — L=2 two-locus recombination-selection model
No mutation (u=v=0), no drift (deterministic), selection acts only on site 1 (s1=0, s2=s)
Haplotype encoding: h = b0 + 2*b1 (site0 LSB, site1 MSB)
  h=0 '00', h=1 '01', h=2 '10', h=3 '11'
SVPG (single-state VPG, single-variable full speed):  dp/dt = J_rec(p) + J_sel(p)
  J_rec,h = r*(Pi g - g)_h,  Pi g[h0,h1] = g0[h0]*g1[h1] (site-independent projection)
  J_sel,h = p_h*(U(h) - Ubar),  U(h) = s*((h>>1)&1)
DVPG (dual-state VPG, bivariate v3.7.3): 
  dg/dt = J_rec(g) + J_c,  drho/dt = J_sel(rho) - J_c,  J_c = gamma*(rho-g)
Equilibrium: site 1 under selection → p2* determined by selection; site 0 neutral → p1* drift-uncertain (conserved under determinism)
"""
import numpy as np

L = 2
N = 4
HAP_LABELS = ['00', '01', '10', '11']

def proj_indep(p):
    """Pi g[h] = g[site0 marginal] * g[site1 marginal]  (L=2 site-independent projection)"""
    g0 = np.array([p[0] + p[2], p[1] + p[3]])   # site0=0, site0=1
    g1 = np.array([p[0] + p[1], p[2] + p[3]])   # site1=0, site1=1
    out = np.zeros(4)
    for h in range(4):
        b0 = h & 1
        b1 = (h >> 1) & 1
        out[h] = g0[b0] * g1[b1]
    return out

def U_vec(s):
    """U(h) = s * (h>>1 & 1): selection acts only on site 1"""
    return np.array([s * ((h >> 1) & 1) for h in range(4)])

def build_R_tensor(L, radj):
    """Multilocus recombination tensor R[i,j,k]: probability that offspring haplotype i is generated from parents (j,k) under recombination.

    Classical multilocus recombination model: crossover events are arranged along the chromosome; the seg-th interval (between sites seg and seg+1) crosses over
    with probability radj[seg]; on crossover the parental source is swapped (parent flip),
    otherwise inheritance continues from the current parent. R satisfies:
        Σ_i R[i,j,k] = 1  (probability normalized for each parental pair to produce an offspring)
        J_rec,i(g) = r·(Σ_jk R[i,j,k]·g_j·g_k − g_i)
    Mathematically equivalent to the projection form J_rec = r(Πg−g) (verified max|Δ| ~ 1e-17),
    but closer to the classical multilocus model, easy to generalize to L loci, and convenient for adding complex recombination patterns.
    For L=1, radj=[] (no intervals), R[i,j,k]=δ_{ij}, J_rec≡0."""
    n_seg = L - 1
    N = 1 << L
    R = np.zeros((N, N, N))
    for h1 in range(N):
        for h2 in range(N):
            for mask in range(1 << n_seg):
                # Probability of this crossover pattern (mask): intervals are independent
                prob = 1.0
                for seg in range(n_seg):
                    if (mask >> seg) & 1:
                        prob *= radj[seg]
                    else:
                        prob *= (1.0 - radj[seg])
                # Assemble the offspring bit by bit along the chromosome: parent tracks the current parental source (0=h1, 1=h2)
                child = 0
                parent = 0
                for i in range(L):
                    if i > 0 and ((mask >> (i - 1)) & 1):
                        parent = 1 - parent        # crossover: swap parental source
                    bit = (h1 >> i) & 1 if parent == 0 else (h2 >> i) & 1
                    child |= bit << i
                R[child, h1, h2] += prob
    return R

def J_rec_tensor(g, R, r):
    """Recombination flux (tensor form, canonical definition): J_rec,i = r·(Σ_jk R[i,j,k] g_j g_k − g_i)

    Note: R_tensor is suitable as a mathematical definition, but the dense tensor
    costs O(2^{3L}) memory/computation, feasible only for L≤8; use J_rec_proj in actual computation (especially L≥10)."""
    offspring = np.einsum('ijk,j,k->i', R, g, g)
    return r * (offspring - g)

def proj_interval(g, L, mask):
    """Interval projection Π_S g: marginal over site set S (mask) ⊗ complement marginal.

    Vectorized implementation: bit operation (h>>l)&1 extracts bits and bit-compresses
    to indices inside S / its complement, O(N·L), no Python per-h loop."""
    N = len(g)
    h = np.arange(N)
    idxS = np.zeros(N, dtype=np.int64)
    idxC = np.zeros(N, dtype=np.int64)
    pi = 0; pc = 0
    for l in range(L):
        bit = (h >> l) & 1
        if (mask >> l) & 1:
            idxS |= bit << pi; pi += 1
        else:
            idxC |= bit << pc; pc += 1
    nS = pi
    m_S = np.bincount(idxS, weights=g, minlength=(1 << nS))
    m_C = np.bincount(idxC, weights=g, minlength=(1 << (L - nS)))
    return m_S[idxS] * m_C[idxC]

def J_rec_proj(g, radj, L):
    """Recombination flux (projection form, used in actual computation): J_rec = Σ_S r_S (Π_S g − g)

    Classical multilocus continuous-time recombination equation: adjacent interval S={0..i} crosses over at rate r_i,
    after which the sites inside/outside S decouple (Π_S projection). Complexity O(N·L²), memory O(N),
    scalable to L≥10. Mathematically equivalent to R_tensor (verified max|Δ| ~ 1e-17)."""
    Jr = np.zeros_like(g)
    for i in range(L - 1):
        mask = (1 << (i + 1)) - 1        # prefix block S = {0..i}
        Jr += radj[i] * (proj_interval(g, L, mask) - g)
    return Jr

def J_rec(p, r):
    """Recombination flux (projection form, equivalent to the tensor form): J_rec = r·(Πg − g)"""
    return r * (proj_indep(p) - p)

def J_sel(p, U):
    Ubar = np.dot(p, U)
    return p * (U - Ubar)

def svpg_rhs(p, radj, L, U):
    """Classical single-variable full-speed equation: dp/dt = J_rec + J_sel (projection recombination form)"""
    return J_rec_proj(p, radj, L) + J_sel(p, U)

def dvpg_rhs(g, rho, radj, L, U, gamma):
    """Full RHS for L=2 of GF-VPG v3.7.3 Eqs. (25)-(26) (with coupling flux, projection recombination form):
        ∂t g   = J_rec(g) + γ(ρ-g)
        ∂t ρ   = J_sel(ρ) - γ(ρ-g)
    Returns (∂t g, ∂t ρ). g, rho: (4,) vectors [00,01,10,11]."""
    Jc = gamma * (rho - g)
    return J_rec_proj(g, radj, L) + Jc, J_sel(rho, U) - Jc

def integrate_svpg(p0, radj, L, s, T, dt):
    U = U_vec(s)
    n = int(round(T / dt))
    ps = np.zeros((n + 1, N))
    p = p0.copy()
    ps[0] = p
    for k in range(n):
        p = p + dt * svpg_rhs(p, radj, L, U)
        p = np.clip(p, 1e-15, 1.0)
        p /= p.sum()
        ps[k + 1] = p
    return ps

def integrate_dvpg(p0, radj, L, s, gamma, T, dt, rho0=None):
    """Numerically integrate Eqs. (25)-(26) for L=2, integrating the full RHS directly (scipy BDF stiff solver).

    Paper dynamics (Eqs. 25-26):
        ∂t g   = J_rec(g) + γ(ρ-g)
        ∂t ρ   = J_sel(ρ) - γ(ρ-g)
    Integrated directly with BDF (implicit stiff solver), no hand-written splitting/matrix exponential:
      - automatically handles stiffness from large γ (fast-slow system: γ and r,s time scales differ greatly)
      - controllable accuracy (rtol=1e-9, atol=1e-12)
      - code mirrors the formulas term by term
    Recombination terms use the projection form J_rec_proj (scalable to L≥10).
    Verification: half-speed factor max|p_SVPG(t/2)-p_DVPG(t)| ≈ 8×10^-6."""
    from scipy.integrate import solve_ivp
    U = U_vec(s)

    def rhs(t, x):
        g = x[:N]
        rho = x[N:]
        Jc = gamma * (rho - g)
        dg = J_rec_proj(g, radj, L) + Jc
        dr = J_sel(rho, U) - Jc
        return np.concatenate([dg, dr])

    n = int(round(T / dt))
    t_eval = np.arange(n + 1) * dt
    x0 = np.concatenate([p0, p0 if rho0 is None else rho0])
    sol = solve_ivp(rhs, [0.0, T], x0, method='BDF', t_eval=t_eval,
                    rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f'BDF integration failed: {sol.message}')
    gs = np.clip(sol.y[:N].T, 1e-15, 1.0); gs /= gs.sum(axis=1, keepdims=True)
    rs = np.clip(sol.y[N:].T, 1e-15, 1.0); rs /= rs.sum(axis=1, keepdims=True)
    return gs, rs

def AF(p, site):
    """Derived allele frequency at site (site=0: site 0, site=1: site 1)"""
    return sum(p[h] for h in range(4) if ((h >> site) & 1))

def AF_series(ps, site):
    return np.array([AF(p, site) for p in ps])

def D12(p):
    """LD: D = p11 - p1*p2 (standard D = p_AB - p_A p_B)"""
    p1 = AF(p, 0)
    p2 = AF(p, 1)
    return p[3] - p1 * p2

def D_series(ps):
    return np.array([D12(p) for p in ps])

if __name__ == '__main__':
    # ---- Quick self-check ----
    radj = np.array([1.0])                # L=2 single interval (sites 0-1) recombination rate
    L = 2
    # Equivalence of the projection form vs R_tensor
    R = build_R_tensor(L, [1.0])
    p0 = np.array([0.45, 0.05, 0.05, 0.45])   # initial: p00=0.45 p11=0.45 p01=0.05 p10=0.05
    print('Projection vs R_tensor equivalence: max|Δ| =',
          np.max(np.abs(J_rec_tensor(p0, R, 0.5) - J_rec_proj(p0, 0.5 * radj, L))))
    print('Initial D12 =', D12(p0), '(expected 0.20)')
    print('Initial AF1 =', AF(p0, 0), ', AF2 =', AF(p0, 1), '(expected 0.5/0.5)')
    # Pure recombination (s=0): LD decays, marginals conserved
    ps = integrate_svpg(p0, radj * 0.01, L, 0.0, 2000, 0.01)
    print('Pure recombination r=0.01, s=0: D_end =', D_series(ps)[-1], '(→0)')
    print('  Marginal conservation: AF1_end =', AF_series(ps, 0)[-1], ', AF2_end =', AF_series(ps, 1)[-1], '(expected ≈0.5)')
    # Pure selection (r=0): site 1 fixes, site 0 conserved (neutral)
    ps = integrate_svpg(p0, radj * 0.0, L, 0.01, 2000, 0.01)
    print('Pure selection r=0, s=0.01: AF2_end =', AF_series(ps, 1)[-1], '(→1), AF1_end =', AF_series(ps, 0)[-1], '(≈0.5)')
    # Recombination + selection: half-speed factor
    ps = integrate_svpg(p0, radj * 0.005, L, 0.005, 2000, 0.01)
    gs, rs = integrate_dvpg(p0, radj * 0.005, L, 0.005, 100.0, 4000, 0.01)
    idx = np.arange(0, len(gs), 500)
    err = np.max(np.abs(ps[idx//2, :] - rs[idx, :]))
    print(f'Half-speed factor γ=100: max|p_SVPG(t/2) - p_DVPG(t)| = {err:.2e}')
    print('OK')
