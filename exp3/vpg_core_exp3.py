#!/usr/bin/env python3
"""
Experiment 3 deterministic core — L=3 three-locus integrated model (mutation + recombination + selection)
Reuses the validated implementation from vpg36_core.py in the old joint_exp_v2.zip package (line-by-line identical):
  - Recombination generator J_rec = Σ r_S(Π_S g - g), Π_{0|12}/Π_{01|2} projections (proj_P0/proj_P2)
  - Selection flow J_FR,h = ρ_h(U(h)-Ubar), additive potential U(h)=Σ s_l b_{h,l}
  - SVPG (single-state VPG)/DVPG (dual-state VPG) integrators (dvpg splitting: nonlinear terms Euler + coupled linear subsystem solved analytically)
v3.7.3 core formulas (Eqs. 25-26):  F[g,rho] = D_KL(g||pi) - <U,rho>,  Jc = gamma*(rho-g)
  dg/dt = J_mut + J_rec + Jc
  drho/dt = J_FR - Jc
Haplotype encoding: h -> binary (h>>i)&1 = site i; label site2 site1 site0 (MSB->LSB)
Selection: U(h) = sum_l s_l * b_{h,l}  (s = [s0, s1, s2], Experiment 3 s=[0.005, 0, -0.005])
Mutation: bidirectional u=v=mu per site
Recombination: two intervals r12 (site0-site1), r23 (site1-site2)
"""
import numpy as np

L = 3
N = 8

def label_hap(h):
    return ''.join(str((h >> i) & 1) for i in range(L - 1, -1, -1))

HAP_LABELS = [label_hap(h) for h in range(N)]

def mut_matrix(u, v):
    """L=3 bidirectional mutation generator matrix M[i,j] = rate from j to i"""
    M = np.zeros((N, N))
    for j in range(N):
        for l in range(L):
            bit = (j >> l) & 1
            i = j ^ (1 << l)
            rate = u if bit == 0 else v
            M[i, j] += rate
            M[j, j] -= rate
    return M

def build_R_tensor(L, radj):
    """Multi-locus recombination tensor R[i,j,k]: probability that offspring haplotype i is generated from parents (j,k) under recombination.

    Classical multi-locus recombination model: crossover events are arranged along the chromosome; the seg-th interval (between sites seg and seg+1)
    crosses over with probability radj[seg]; on a crossover the parental origin flips (parent flip),
    otherwise inheritance continues along the current parent. R satisfies:
        Σ_i R[i,j,k] = 1  (normalized: each pair of parents produces some offspring)
        J_rec,i(g) = r·(Σ_jk R[i,j,k]·g_j·g_k − g_i)
    Mathematically equivalent to the projection form J_rec = r(Πg−g) (verified max|Δ| ~ 1e-17),
    but closer to the classical multi-locus model, easy to generalize to L loci, and convenient for adding complex recombination patterns.
    For L=1, radj=[] (no intervals), R[i,j,k]=δ_{ij}, and J_rec≡0."""
    n_seg = L - 1
    N = 1 << L
    R = np.zeros((N, N, N))
    for h1 in range(N):
        for h2 in range(N):
            for mask in range(1 << n_seg):
                # probability of this crossover pattern (mask): intervals are independent
                prob = 1.0
                for seg in range(n_seg):
                    if (mask >> seg) & 1:
                        prob *= radj[seg]
                    else:
                        prob *= (1.0 - radj[seg])
                # assemble the offspring bit by bit along the chromosome: parent tracks the current parent (0=h1, 1=h2)
                child = 0
                parent = 0
                for i in range(L):
                    if i > 0 and ((mask >> (i - 1)) & 1):
                        parent = 1 - parent        # crossover: flip the parent
                    bit = (h1 >> i) & 1 if parent == 0 else (h2 >> i) & 1
                    child |= bit << i
                R[child, h1, h2] += prob
    return R

def J_rec_tensor(g, R, r):
    """Recombination flow (tensor form, canonical definition): J_rec,i = r·(Σ_jk R[i,j,k] g_j g_k − g_i)

    Note: R_tensor is suitable as a mathematical definition (transparent interval probabilities, intuitive L-locus generalization),
    but the dense tensor costs O(2^{3L}) memory and O(2^{3L}) computation, suitable only for teaching/validation with L≤8;
    actual computation (especially L≥10) should use the projection form J_rec_proj (O(N·L) per interval)."""
    offspring = np.einsum('ijk,j,k->i', R, g, g)
    return r * (offspring - g)

_PROJ_CACHE = {}   # (L, mask) -> (idxS, idxC, nS): index cache (fixed for given mask/L, avoids rebuilding every step)

def proj_interval(g, L, mask):
    """Interval projection Π_S g: marginal over the site set S (mask) ⊗ complement marginal.

    Π_S g[h] = g[S marginal](h_S) · g[complement marginal](h_C),
    i.e., decouples the sites of h inside S from those outside S (S-internal and S-external each jointly independent).
    In classical multi-locus recombination, the crossover operator for interval S (prefix block {0..i}) is exactly this projection.
    Vectorized implementation: extract the bit of each site with bit operations (h>>l)&1, bit-compress to
    S-internal/complement indices, O(N·L), no per-h Python loop.
    Indices (idxS, idxC) are independent of g and depend only on (L, mask); the module-level cache avoids
    rebuilding them at every step of Langevin/integrators (~3x faster)."""
    key = (L, mask)
    cached = _PROJ_CACHE.get(key)
    if cached is None:
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
        cached = (idxS, idxC, nS)
        _PROJ_CACHE[key] = cached
    idxS, idxC, nS = cached
    m_S = np.bincount(idxS, weights=g, minlength=(1 << nS))
    m_C = np.bincount(idxC, weights=g, minlength=(1 << (L - nS)))
    return m_S[idxS] * m_C[idxC]

def J_rec_proj(g, radj, L):
    """Recombination flow (projection form, used in practice): J_rec = Σ_S r_S (Π_S g − g)

    Classical continuous-time multi-locus recombination equation (standard form, e.g., Hudson): for each adjacent interval
    S = {0..i} (the crossover region between sites i and i+1), crossing over occurs at rate r_i,
    and after a crossover the sites inside S are decoupled from those outside S (Π_S projection).
    Complexity: O(N·L) per interval, O(N·L²) total, O(N) memory — for L=10, ~10^5
    operations per step (~10^4x faster than the dense R_tensor, O(N) instead of O(N³) memory).
    Mathematically equivalent to R_tensor (verified max|Δ| ~ 1e-17)."""
    Jr = np.zeros_like(g)
    for i in range(L - 1):
        mask = (1 << (i + 1)) - 1        # prefix block S = {0..i}
        Jr += radj[i] * (proj_interval(g, L, mask) - g)
    return Jr

def U_vec(s):
    """s: selection coefficient vector of length L"""
    if np.isscalar(s):
        s = [s] * L
    return np.array([sum(s[l] * ((h >> l) & 1) for l in range(L)) for h in range(N)])

def J_FR(rho, U):
    Ubar = np.dot(rho, U)
    return rho * (U - Ubar)

def pi_uniform():
    return np.full(N, 1.0 / N)

def svpg_rhs(p, M, U, radj, L):
    """Classic univariate full-speed equation: dp/dt = J_mut + J_rec + J_sel (projection recombination form)"""
    return (M @ p + J_rec_proj(p, radj, L)
            + p * (U - np.dot(p, U)))

def dvpg_rhs(g, rho, M, U, radj, L, gamma):
    """L=3 full RHS of GF-VPG v3.7.3 Eqs. (25)-(26) (with coupling flow, projection recombination form):
        ∂t g   = J_mut(g) + J_rec(g) + γ(ρ-g)
        ∂t ρ   = J_sel(ρ) - γ(ρ-g)
    Returns (∂t g, ∂t ρ). g, rho: (8,) vectors [000..111]."""
    Jc = gamma * (rho - g)
    Jm = M @ g
    Jr = J_rec_proj(g, radj, L)
    Jsel = rho * (U - np.dot(rho, U))
    return Jm + Jr + Jc, Jsel - Jc

def integrate_svpg(p0, M, U, radj, L, T, dt):
    n = int(round(T / dt))
    ps = np.zeros((n + 1, N))
    p = p0.copy()
    ps[0] = p
    for k in range(n):
        p = p + dt * svpg_rhs(p, M, U, radj, L)
        p = np.clip(p, 1e-15, 1.0)
        p /= p.sum()
        ps[k + 1] = p
    return ps

def integrate_dvpg(p0, M, U, radj, L, gamma, T, dt):
    """Numerical integration of Eqs. (25)-(26) for L=3, matrix-exponential (exponential integrator) scheme.

    The paper dynamics (Eqs. 25-26) are written as a linear + nonlinear split:
        d/dt [g] = C [g] + [J_mut(g)+J_rec(g)] ,   C = [ -γ   γ ]
            [ρ]      [ρ]   [J_sel(ρ)      ]            [  γ  -γ ]
    (the same 2×2 coupling matrix C for every haplotype component h)
    Eigendecomposition of the linear coupling matrix C: eigenvalue 0 (direction (1,1), mean preserved) and -2γ
    (direction (1,-1), difference decays). Its matrix exponential is:
        e^{C·dt} = [ a  b ],  a = (1+e^{-2γdt})/2,  b = (1-e^{-2γdt})/2
                   [ b  a ]
    Each step dt:
      1) one explicit Euler step of the nonlinear flows J_mut, J_rec, J_sel:
           A = g + dt·(J_mut+J_rec)(g),   B = ρ + dt·J_sel(ρ)
      2) exact solution of the coupled linear system (matrix exponential applied componentwise, one step):
           g' = a·A + b·B,   ρ' = b·A + a·B
    This scheme is unconditionally stable (converges for arbitrarily large γ) and exact for the coupled subsystem (not a first-order approximation).

    Note: why not use scipy BDF directly? BDF/Radau trial steps can push the probability components temporarily off
    the simplex (ρ_i < 0), so the selection flow loses its probabilistic meaning and bilinear recombination products
    explode (NaN/divergence in practice). The exponential splitting here treats the coupled stiff subsystem exactly and
    applies a simplex projection (clip + normalization) each step to correct numerical deviations of the explicit substep.

    The recombination term uses the projection form J_rec_proj (O(N·L²), scalable to L≥10);
    R_tensor is kept as the canonical definition (see build_R_tensor comment).

    Validation: half-speed factor max|p_SVPG(t/2)-p_DVPG(t)| ≈ 2×10^-5."""
    n = int(round(T / dt))
    gs = np.zeros((n + 1, N)); rs = np.zeros((n + 1, N))
    g = p0.copy(); rho = p0.copy()
    gs[0] = g; rs[0] = rho
    e = np.exp(-2.0 * gamma * dt)     # difference-decay factor of e^{C·dt}
    a = (1.0 + e) / 2.0               # diagonal element of the matrix exponential
    b = (1.0 - e) / 2.0               # off-diagonal element of the matrix exponential
    for k in range(n):
        # 1) explicit Euler of the nonlinear flows (mutation/recombination act on g, selection on ρ)
        dg, dr = dvpg_rhs(g, rho, M, U, radj, L, 0.0)
        A = g + dt * dg
        B = rho + dt * dr
        # 2) exact solution of the coupled linear system: u' = e^{C·dt} u (componentwise), done in one step
        g = a * A + b * B
        rho = b * A + a * B
        g = np.clip(g, 1e-15, 1.0); g /= g.sum()
        rho = np.clip(rho, 1e-15, 1.0); rho /= rho.sum()
        gs[k + 1] = g; rs[k + 1] = rho
    return gs, rs

def AF_series(ps, site):
    return np.array([sum(p[h] for h in range(N) if ((h >> site) & 1)) for p in ps])

def D_pair_series(ps, i, j):
    """D_ij = p(site i=1, site j=1) - p_i p_j"""
    out = np.zeros(len(ps))
    for k, p in enumerate(ps):
        p11 = sum(p[h] for h in range(N) if ((h >> i) & 1) and ((h >> j) & 1))
        pi = sum(p[h] for h in range(N) if (h >> i) & 1)
        pj = sum(p[h] for h in range(N) if (h >> j) & 1)
        out[k] = p11 - pi * pj
    return out

if __name__ == '__main__':
    # ---- quick self-check ----
    mu = 5e-4
    s = np.array([0.005, 0.0, -0.005])
    radj = np.array([0.005, 0.005])          # two-interval recombination rates r1, r2
    p0 = np.zeros(N)
    p0[0] = 0.4; p0[7] = 0.4            # 000 and 111
    for h in range(1, 7):
        p0[h] = 0.2 / 6
    M = mut_matrix(mu, mu)
    U = U_vec(s)
    # equivalence of projection form vs R_tensor
    R1 = build_R_tensor(L, [1.0, 0.0]); R2 = build_R_tensor(L, [0.0, 1.0])
    Jr_t = J_rec_tensor(p0, R1, radj[0]) + J_rec_tensor(p0, R2, radj[1])
    Jr_p = J_rec_proj(p0, radj, L)
    print('Projection vs R_tensor equivalence: max|Δ| =', np.max(np.abs(Jr_t - Jr_p)))
    ps = integrate_svpg(p0, M, U, radj, L, 2000, 0.01)
    print('Initial AF:', [AF_series(np.array([p0]), i)[0] for i in range(3)])
    print('Final AF:', [AF_series(ps, i)[-1] for i in range(3)])
    print('Initial D12/D23/D13:', [D_pair_series(np.array([p0]), i, j)[0] for i, j in [(0,1),(1,2),(0,2)]])
    print('Final D12/D23/D13:', [D_pair_series(ps, i, j)[-1] for i, j in [(0,1),(1,2),(0,2)]])
    # half-speed factor validation
    gs, rs = integrate_dvpg(p0, M, U, radj, L, 100.0, 4000, 0.01)
    idx = np.arange(0, len(gs), 500)
    err = np.max(np.abs(ps[idx//2, :] - rs[idx, :]))
    print(f'Half-speed factor γ=100: max|p_SVPG(t/2) - p_DVPG(t)| = {err:.2e}')
    print('OK')
