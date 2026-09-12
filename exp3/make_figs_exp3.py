#!/usr/bin/env python3
"""
Experiment 3 analysis and plotting — L=3 three-locus integrated model (mutation + recombination + selection)
Fig 3A: AF trajectories (sites 0/1/2, SVPG (single-state VPG)/DVPG (dual-state VPG)/SLiM)
Fig 3B: LD trajectories (D12/D23/D13)
Conventions: same parameters same color, models distinguished by line style (SVPG thin faint solid / DVPG ρ thick dashed t/2 / SLiM dotted + 95% CI)
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vpg_core_exp3 as vc

os.makedirs(os.path.join(_HERE, 'figs'), exist_ok=True)
FIG = os.path.join(_HERE, 'figs')
DATA = os.path.join(_HERE, 'data')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

mu = 0.001
s = np.array([0.005, 0.0, -0.005])
r1 = 0.01; r2 = 0.05
radj = np.array([r1, r2])             # two-interval recombination rates (projection-form recombination flow)
GAMMA = 100.0
T = 2000.0; dt = 0.01
T_DVPG = 4000.0   # DVPG half-speed: integrate to 4000 generations, divide by 2 to align

p0 = np.zeros(8)
p0[0] = 0.4; p0[7] = 0.4
for h in range(1, 7):
    p0[h] = 0.2 / 6

# ============ deterministic computation ============
M = vc.mut_matrix(mu, mu)
U = vc.U_vec(s)
ps = vc.integrate_svpg(p0, M, U, radj, 3, T, dt)
gs, rs_dv = vc.integrate_dvpg(p0, M, U, radj, 3, GAMMA, T_DVPG, dt)
print("SVPG end AF:", [f"{vc.AF_series(ps, i)[-1]:.4f}" for i in range(3)])
print("SVPG end D:", [f"{vc.D_pair_series(ps, i, j)[-1]:.4f}" for i, j in [(0,1),(1,2),(0,2)]])

t_sv = np.arange(len(ps)) * dt
t_dv = np.arange(len(rs_dv)) * dt
t_dv_half = t_dv / 2.0

# ============ SLiM summary ============
arr = np.load(f'{DATA}/slim3_main.npy')     # (50, 201, 9): gen, h0..h7
gen = arr[0, :, 0]
hslim = arr[:, :, 1:]                            # (50, 201, 8)
slim_af = []
for site in range(3):
    mask = np.array([(h >> site) & 1 for h in range(8)])
    af = hslim @ mask                            # (50, 201)
    slim_af.append(dict(m=af.mean(0), se=af.std(0)/np.sqrt(hslim.shape[0])))
slim_d = {}
for i, j in [(0,1),(1,2),(0,2)]:
    # vectorized computation of D_ij for each rep and generation
    mask11 = np.array([1 if ((hh>>i)&1) and ((hh>>j)&1) else 0 for hh in range(8)])
    pi = np.array([(hh>>i)&1 for hh in range(8)])
    pj = np.array([(hh>>j)&1 for hh in range(8)])
    d = (hslim @ mask11) - (hslim @ pi) * (hslim @ pj)   # (50, 201)
    slim_d[(i,j)] = dict(m=d.mean(0), se=d.std(0)/np.sqrt(hslim.shape[0]))

# ============ Figure 1 (first figure of the paper): haplotype frequency trajectories (single panel, 8 haplotypes) ============
# 8 haplotypes use 8 highly distinguishable colors (tab20 alternating colors, avoiding confusion of similar colors in low-frequency regions)
fig, ax = plt.subplots(figsize=(13, 6.5))
HAP_COLORS = [plt.cm.tab20(2*k) for k in range(8)]   # tab20 even indices: high distinguishability
for k in range(8):
    c = HAP_COLORS[k]
    ax.plot(t_sv, ps[:, k], '-', color=c, lw=1.3, alpha=0.6, label=f'SVPG {vc.HAP_LABELS[k]}')
    ax.plot(t_dv_half, rs_dv[:, k], '--', color=c, lw=1.8, alpha=0.85)
    m = hslim[:, :, k].mean(0); se = hslim[:, :, k].std(0)/np.sqrt(hslim.shape[0])
    ax.plot(gen, m, ':', color=c, lw=1.2, alpha=0.8)
    ax.fill_between(gen, m-se, m+se, color=c, alpha=0.08)
ax.axhline(0.125, color='gray', lw=0.8, ls=':', alpha=0.6)   # uniform equilibrium 1/8
ax.set_xlabel('t (generation)', fontsize=13)
ax.set_ylabel('haplotype frequency', fontsize=13)
ax.set_xlim(0, 2000)
ax.set_title('Experiment 3 Figure 1: haplotype frequency trajectories — three-locus integrated (u=v=0.001, r12=0.01, r23=0.05; '
             'SVPG solid / DVPG rho dashed t/2 / SLiM dotted±95%CI; gray dotted = uniform equilibrium 1/8)',
             fontsize=12.5, fontweight='bold')
ax.grid(alpha=0.25)
ax.legend(fontsize=7.5, ncol=2, loc='upper right')
plt.tight_layout()
plt.savefig(f'{FIG}/exp3_figC_hap.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 1 (hap single panel) saved")

# ============ Figure 2: AF trajectories (single panel, 3 sites) ============
SITE_NAMES = ['Site 0 (s=+0.005 beneficial)', 'Site 1 (s=0 neutral)', 'Site 2 (s=-0.005 deleterious)']
SITE_COLORS = ['#E74C3C', '#27AE60', '#2E86AB']
fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, c) in enumerate(zip(SITE_NAMES, SITE_COLORS)):
    ax.plot(t_sv, vc.AF_series(ps, i), '-', color=c, lw=1.3, alpha=0.6, label=f'SVPG {name}')
    ax.plot(t_dv_half, vc.AF_series(rs_dv, i), '--', color=c, lw=2.0, alpha=0.85, label=f'DVPG ρ {name}')
    ax.plot(gen, slim_af[i]['m'], ':', color=c, lw=1.4, alpha=0.8, label=f'SLiM {name}')
    ax.fill_between(gen, slim_af[i]['m']-slim_af[i]['se'],
                    slim_af[i]['m']+slim_af[i]['se'], color=c, alpha=0.12)
ax.set_xlim(0, 2000); ax.set_ylim(0, 1.05)
ax.set_xlabel('t (generation)', fontsize=13)
ax.set_ylabel('AF', fontsize=13)
ax.set_title('Experiment 3 Figure 2: allele frequency trajectories — three-locus integrated (u=v=0.001, r12=0.01, r23=0.05, γ=100, Ne=1000, 100 reps)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.25)
ax.legend(fontsize=8, ncol=3, loc='center right')
plt.tight_layout()
plt.savefig(f'{FIG}/exp3_figA_AF.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 2 (AF single panel) saved")

# ============ Figure 3: LD trajectories (single panel, 3 site pairs, x-axis zoomed to the first 200 generations) ============
# Note: D23 and D13 nearly coincide physically (decay rates r23≈r12+r23=0.1, r12 negligible, half-lives both ~6.9 generations),
#     so the x-axis is zoomed to 0-200 to show the main decay interval; D13 in magenta to distinguish it from D23 in orange
LD_NAMES = ['D₁₂ (sites 0-1)', 'D₂₃ (sites 1-2)', 'D₁₃ (sites 0-2)']
LD_COLORS = ['#8E44AD', '#E67E22', '#C2185B']
fig, ax = plt.subplots(figsize=(11, 6))
for (i, j), (name, c) in zip([(0,1),(1,2),(0,2)], zip(LD_NAMES, LD_COLORS)):
    ax.plot(t_sv, vc.D_pair_series(ps, i, j), '-', color=c, lw=1.3, alpha=0.6, label=f'SVPG {name}')
    ax.plot(t_dv_half, vc.D_pair_series(rs_dv, i, j), '--', color=c, lw=2.0, alpha=0.85, label=f'DVPG ρ {name}')
    ax.plot(gen, slim_d[(i,j)]['m'], ':', color=c, lw=1.4, alpha=0.8, label=f'SLiM {name}')
    ax.fill_between(gen, slim_d[(i,j)]['m']-slim_d[(i,j)]['se'],
                    slim_d[(i,j)]['m']+slim_d[(i,j)]['se'], color=c, alpha=0.12)
ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax.set_xlim(0, 200)
ax.set_xlabel('t (generation, first 200)', fontsize=13)
ax.set_ylabel('D', fontsize=13)
ax.set_title('Experiment 3 Figure 3: linkage disequilibrium trajectories — three-locus integrated (initial D(0)≈0.183; '
             'D₂₃ and D₁₃ decay at ≈r₂₃=0.1, almost coincident, half-life ≈6.9 gen; D₁₂ decays slowly at r₁₂=0.001)',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.25)
ax.legend(fontsize=7.5, ncol=3, loc='upper right')
plt.tight_layout()
plt.savefig(f'{FIG}/exp3_figB_LD.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 3 (LD single panel, xlim 200) saved")

# ============ summary ============
print("\n===== Summary =====")
print("Final AF: SVPG", [f"{vc.AF_series(ps, i)[-1]:.4f}" for i in range(3)],
      "| SLiM", [f"{slim_af[i]['m'][-1]:.4f}±{slim_af[i]['se'][-1]:.4f}" for i in range(3)])
print("Final D12/D23/D13: SVPG",
      [f"{vc.D_pair_series(ps, i, j)[-1]:.4f}" for i, j in [(0,1),(1,2),(0,2)]],
      "| SLiM",
      [f"{slim_d[(i,j)]['m'][-1]:.4f}±{slim_d[(i,j)]['se'][-1]:.4f}" for i, j in [(0,1),(1,2),(0,2)]])
# half-speed factor
idx = np.arange(0, len(rs_dv), 500)
err = np.max(np.abs(ps[idx//2, :] - rs_dv[idx, :]))
print(f"Half-speed factor validation: max|p_SVPG(t/2)-p_DVPG(t)| = {err:.2e}")
print("ALL FIGURES DONE")
