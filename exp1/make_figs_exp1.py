#!/usr/bin/env python3
"""
Experiment 1 analysis and plotting — L=1 single-locus mutation-selection balance
Fig 1A: AF(t) trajectories for different s: SVPG (single-state VPG)/DVPG (dual-state VPG)/SLiM (mean±SE)   (µ=1e-3, p0=0.01)
Fig 1B: Equilibrium frequency p* comparison: VPG vs SLiM (3x3 parameter matrix + p0=0.99 bidirectional)
Fig 1C: DVPG AF(g) vs AF(ρ) coupling synchronization (main parameters)
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vpg_core_exp1 as vc

os.makedirs(os.path.join(_HERE, 'figs'), exist_ok=True)
FIG = os.path.join(_HERE, 'figs')
DATA = os.path.join(_HERE, 'data')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

C_SV = '#2E86AB'; C_DV = '#E67E22'; C_SL = '#111111'
T = 2000.0; dt = 0.01
N_REC = 201          # every 10 generations

def slim_summary(fn):
    arr = np.load(fn)               # (50, 201, 2): gen, p
    gen = arr[0, :, 0]
    p = arr[:, :, 1]                # (50, 201)
    return dict(gen=gen, mean=p.mean(axis=0), se=p.std(axis=0)/np.sqrt(p.shape[0]),
                reps=p)

def key_of(mu, s, p0):
    return f'slim1_mu{str(mu).replace(".","p")}_s{str(s).replace(".","p")}_p0{str(p0).replace(".","p")}.npy'

# ============ Deterministic computation ============
mus = [3e-4, 1e-3, 3e-3]
ss  = [0.005, 0.01, 0.05]
GAMMA = 100.0
T_DVPG = 4000.0               # DVPG half-speed: integrate to 4000 generations, divide by 2 to align with SVPG full-speed 0-2000

det = {}
for mu in mus:
    for s in ss:
        ps = vc.integrate_svpg(0.01, mu, mu, s, T, dt)
        g1, r1 = vc.integrate_dvpg(0.01, mu, mu, s, GAMMA, T_DVPG, dt)
        det[(mu, s)] = dict(ps=ps, g1=g1, r1=r1, peq=vc.eq_balance(mu, mu, s))
# p0=0.99 bidirectional
for mu in [1e-3, 3e-3]:
    ps = vc.integrate_svpg(0.99, mu, mu, 0.01, T, dt)
    g1, r1 = vc.integrate_dvpg(0.99, mu, mu, 0.01, GAMMA, T_DVPG, dt)
    det[('back', mu)] = dict(ps=ps, g1=g1, r1=r1, peq=vc.eq_balance(mu, mu, 0.01))

t = np.arange(N_REC) * 10.0          # SLiM recording times (0,10,...,2000)
t_sv = np.arange(len(det[(1e-3, 0.01)]['ps'])) * dt
t_dv = np.arange(len(det[(1e-3, 0.01)]['r1'])) * dt
t_dv_half = t_dv / 2.0               # DVPG time axis divided by 2 (half-speed factor), covering 0-2000

# Sample SLiM grid (50, 201) -> match t_sv (200001 points) by taking every 10 generations
def thin(x, stride=1000):
    """x: uniform dt=0.01 sequence; take every 10 generations (stride=1000)"""
    return x[::stride]

# ============ Fig 1A: AF(t) for different s — single panel (µ=1e-3, p0=0.01) ============
# Convention: same color for same s, line style distinguishes model (SVPG thin solid / DVPG dashed / SLiM dotted); all with transparency
# SLiM shading: mean ± SE
S_COLORS = {'0.005': '#2E86AB', '0.01': '#27AE60', '0.05': '#E74C3C'}
fig, ax = plt.subplots(figsize=(11, 6))
for s in ss:
    mu = 1e-3
    c = S_COLORS[str(s)]
    d = det[(mu, s)]
    fn = os.path.join(DATA, key_of(mu, s, 0.01))
    slim = slim_summary(fn)
    ax.plot(t_sv, d['ps'], '-', color=c, lw=1.4, alpha=0.65, label=f'SVPG s={s}')
    ax.plot(t_dv_half, d['r1'], '--', color=c, lw=2.0, alpha=0.85, label=f'DVPG ρ (t/2) s={s}')
    ax.plot(slim['gen'], slim['mean'], ':', color=c, lw=1.6, alpha=0.9, label=f'SLiM mean±SE s={s}')
    ax.fill_between(slim['gen'], slim['mean']-slim['se'], slim['mean']+slim['se'],
                    color=c, alpha=0.15)
    ax.axhline(d['peq'], color=c, lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel('t (generation)', fontsize=13)
ax.set_ylabel('p = AF(A1)', fontsize=13)
ax.set_xlim(0, 2000); ax.set_ylim(0, 1.05)
ax.set_title('Experiment 1: single-locus mutation-selection balance — AF trajectories (μ=10^-3, p(0)=0.01, γ=100, Ne=1000, 100 reps)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, ncol=3, loc='center right')
plt.tight_layout()
plt.savefig(f'{FIG}/exp1_figA_traj.png', dpi=150, bbox_inches='tight'); plt.close()
print("Fig 1A saved (single panel, 95%CI)")

# ============ Fig 1B: Equilibrium frequency p* comparison VPG vs SLiM ============
# 9 (µ,s) groups + 2 p0=0.99 groups; all start from p0=0.01 (converge within 2000 generations)
labels = []
vpg_eq = []
slim_eq = []
slim_se = []
for mu in mus:
    for s in ss:
        d = det[(mu, s)]
        slim = slim_summary(os.path.join(DATA, key_of(mu, s, 0.01)))
        labels.append(f'μ={mu:g}\ns={s:g}')
        vpg_eq.append(d['peq'])
        slim_eq.append(slim['mean'][-1])
        slim_se.append(slim['se'][-1])
# Bidirectional convergence groups (p0=0.99)
for mu in [1e-3, 3e-3]:
    d = det[('back', mu)]
    slim = slim_summary(os.path.join(DATA, key_of(mu, 0.01, 0.99)))
    labels.append(f'p0=0.99\nμ={mu:g}')
    vpg_eq.append(d['peq'])
    slim_eq.append(slim['mean'][-1])
    slim_se.append(slim['se'][-1])

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(labels))
ax.errorbar(x, slim_eq, yerr=np.array(slim_se), fmt='o', color=C_SL, capsize=4,
            markersize=6, label='SLiM mean±SE (t=2000)')
ax.plot(x, vpg_eq, 's', color=C_SV, markersize=7, label='VPG equilibrium p*')
for xi, a, b in zip(x, vpg_eq, slim_eq):
    ax.plot([xi, xi], [a, b], color='gray', lw=0.8, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Equilibrium frequency p*', fontsize=13)
ax.set_ylim(0, 1.1)
ax.set_title('Experiment 1: equilibrium frequency p* — VPG (classical equilibrium) vs SLiM (t=2000 mean±SE)', fontsize=14)
ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/exp1_figB_balance.png', dpi=150, bbox_inches='tight'); plt.close()
print("Fig 1B saved")

# ============ Fig 1C: DVPG AF(g) vs AF(ρ) coupling synchronization (γ scan) — single panel ============
# Convention: same color for same γ, g solid / ρ dashed; SVPG black thick line as reference
mu = 0.01; s = 0.05          # user-specified: μ=0.01, s=0.05 (large Jsel, clearer separation)
P0 = 0.01                    # initial g0=ρ0=0.01
GAMMAS = [0.01, 0.1, 1.0, 10.0, 100.0]
G_COLORS = {0.01: '#8E44AD', 0.1: '#2E86AB', 1.0: '#27AE60', 10.0: '#E67E22', 100.0: '#E74C3C'}
T_DV = 1000.0                 # DVPG half-speed: integrate to 1000 generations, divide by 2 when plotting → shows generations 0-500
coupl = {}
for gv in GAMMAS:
    g1c, r1c = vc.integrate_dvpg(P0, mu, mu, s, gv, T_DV, dt)
    coupl[gv] = dict(g1=g1c, r1=r1c)
sv_ref = vc.integrate_svpg(P0, mu, mu, s, 500.0, dt)   # SVPG full-speed trajectory (0-500)
t_dv = np.arange(int(round(T_DV/dt)) + 1) * dt
t_dv_half = t_dv / 2.0        # DVPG half-speed: time axis divided by 2 to align with full-speed SVPG

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(t_sv[:50001], sv_ref, '-', color='#111111', lw=1.3, alpha=0.55, label='SVPG')
for gv in GAMMAS:
    cc = coupl[gv]
    c = G_COLORS[gv]
    ax.plot(t_dv_half, cc['g1'], '-', color=c, lw=1.8, alpha=0.75, label=f'AF(g) γ={gv:g}')
    ax.plot(t_dv_half, cc['r1'], '--', color=c, lw=1.8, alpha=0.75, label=f'AF(ρ) γ={gv:g}')
ax.set_xlabel('t (generation, SVPG full-speed time)', fontsize=13)
ax.set_ylabel('AF', fontsize=13)
ax.set_xlim(0, 500); ax.set_ylim(0, 1.05)
ax.set_title('Experiment 1: DVPG coupling synchronization — AF(g)/AF(ρ) γ scan (μ=0.01, s=0.05, g0=ρ0=0.01, first 500 gen, DVPG t/2 aligned)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=2, loc='center right')
plt.tight_layout()
plt.savefig(f'{FIG}/exp1_figC_coupling.png', dpi=150, bbox_inches='tight'); plt.close()
print("Fig 1C saved (single panel, gamma scan, p0=0.01, 0-500 gen)")

# ============ Summary table ============
print("\n===== Summary of results =====")
print(f"{'params':<24}{'p*(VPG)':>10}{'p_SLiM':>10}{'SE':>8}{'p_end_repSD':>12}")
for mu in mus:
    for s in ss:
        d = det[(mu, s)]
        slim = slim_summary(os.path.join(DATA, key_of(mu, s, 0.01)))
        print(f"μ={mu:g} s={s:g}{'':<12} {d['peq']:10.4f}{slim['mean'][-1]:10.4f}"
              f"{slim['se'][-1]:8.4f}{slim['reps'][:,-1].std():12.4f}")
for mu in [1e-3, 3e-3]:
    d = det[('back', mu)]
    slim = slim_summary(os.path.join(DATA, key_of(mu, 0.01, 0.99)))
    print(f"μ={mu:g} s=0.01 p0=0.99 {d['peq']:10.4f}{slim['mean'][-1]:10.4f}"
          f"{slim['se'][-1]:8.4f}{slim['reps'][:,-1].std():12.4f}")

# ============ Appendix: half-speed factor validation ============
ps = det[(1e-3, 0.01)]['ps']
r1 = det[(1e-3, 0.01)]['r1']
idx = np.arange(0, len(ps), 500)
err_half = np.max(np.abs(ps[idx//2] - r1[idx]))   # p_SVPG(t/2) vs p_DVPG(t)
print(f"\nHalf-speed factor validation (µ=1e-3, s=0.01): max|p_SVPG(t/2) - p_DVPG(t)| = {err_half:.2e}")
print("ALL FIGURES DONE")
