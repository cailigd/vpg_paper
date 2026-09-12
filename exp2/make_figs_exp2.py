#!/usr/bin/env python3
"""
Experiment 2 analysis and plotting — L=2 two-locus recombination-selection model
Fig 2A: AF1, AF2 trajectories (9 groups: 3×3 grid r×s, SVPG (single-state VPG)/DVPG (dual-state VPG)/SLiM)
Fig 2B: D12(t) LD trajectories (9 groups, same layout)
Convention: same parameters share the same color, line styles distinguish models (SVPG thin solid / DVPG dashed t/2 / SLiM dotted mean±SE)
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vpg_core_exp2 as vc
from vpg_core_exp2 import HAP_LABELS

os.makedirs(os.path.join(_HERE, 'figs'), exist_ok=True)
FIG = os.path.join(_HERE, 'figs')
DATA = os.path.join(_HERE, 'data')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

p0 = np.array([0.45, 0.05, 0.05, 0.45])
ss = [0.005, 0.05]          # 2x2 grid (original s={0.002,0.01} changed to {0.005,0.05})
rs = [0.001, 0.01]          # 2x2 grid
GAMMA = 100.0
T = 2000.0; dt = 0.01
T_DVPG = 4000.0   # DVPG half-speed: integrate to 4000 generations, align by dividing time by 2

S_COLORS = {0.005: '#2E86AB', 0.05: '#E74C3C'}

def slim_summary(fn):
    arr = np.load(fn)               # (n_reps, 201, 5): gen, p00..p11 (100/400 reps)
    gen = arr[0, :, 0]
    h = arr[:, :, 1:]               # (n_reps, 201, 4)
    af1 = h[:, :, 1] + h[:, :, 3]   # p01 + p11
    af2 = h[:, :, 2] + h[:, :, 3]   # p10 + p11
    d = h[:, :, 0] * h[:, :, 3] - h[:, :, 1] * h[:, :, 2]   # p00*p11 - p01*p10
    return dict(gen=gen, af1=af1, af2=af2, d=d,
                af1m=af1.mean(0), af1se=af1.std(0)/np.sqrt(af1.shape[0]),
                af2m=af2.mean(0), af2se=af2.std(0)/np.sqrt(af2.shape[0]),
                dm=d.mean(0), dse=d.std(0)/np.sqrt(d.shape[0]))

def key_of(r, s):
    return f'slim2_r{str(r).replace(".","p")}_s{str(s).replace(".","p")}.npy'

# ============ Deterministic computation ============
det = {}
for r in rs:
    radj = np.array([r])              # L=2 single-interval recombination rate (projection-form recombination flux)
    for s in ss:
        ps = vc.integrate_svpg(p0, radj, 2, s, T, dt)
        gs, rs_dv = vc.integrate_dvpg(p0, radj, 2, s, GAMMA, T_DVPG, dt)
        det[(r, s)] = dict(ps=ps, gs=gs, rs_dv=rs_dv)
        # Summary
        a1 = vc.AF_series(ps, 0)[-1]; a2 = vc.AF_series(ps, 1)[-1]
        print(f"det r={r} s={s}: AF1_end={a1:.4f} AF2_end={a2:.4f} D_end={vc.D_series(ps)[-1]:.4f}")

t_sv = np.arange(len(det[(0.001, 0.005)]['ps'])) * dt
t_dv = np.arange(len(det[(0.001, 0.005)]['rs_dv'])) * dt
t_dv_half = t_dv / 2.0

# ============ Figure 1 (first figure in the report): haplotype frequency trajectories (2×2 grid r×s) ============
# Show the dynamics of the 4 haplotypes: favorable combination 11 is maintained/enriched, neutral 00 is diluted, recombinant 01/10 rise then fall
# Line styles: SVPG thin light solid / DVPG thick dashed (distinguishable when overlapping) / SLiM dotted mean±SE
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
HAP_COLORS = {0: '#2E86AB', 1: '#8E44AD', 2: '#E67E22', 3: '#E74C3C'}
for i, r in enumerate(rs):
    for j, s in enumerate(ss):
        ax = axes[i][j]
        d = det[(r, s)]
        arr = np.load(os.path.join(DATA, key_of(r, s)))
        harr = arr[:, :, 1:]          # (n_reps, 201, 4)
        for h in range(4):
            c = HAP_COLORS[h]
            # SVPG thin light solid line
            ax.plot(t_sv, d['ps'][:, h], '-', color=c, lw=1.1, alpha=0.5,
                    label=f'SVPG {HAP_LABELS[h]}')
            # DVPG rho thick dashed line (t/2 aligned) — when overlapping, distinguished by line width/alpha
            ax.plot(t_dv_half, d['rs_dv'][:, h], '--', color=c, lw=2.0, alpha=0.85)
            # SLiM dotted line mean±SE
            m = harr[:, :, h].mean(0); se = harr[:, :, h].std(0)/np.sqrt(harr.shape[0])
            ax.plot(arr[0, :, 0], m, ':', color=c, lw=1.1, alpha=0.65)
            ax.fill_between(arr[0, :, 0], m-se, m+se, color=c, alpha=0.07)
        ax.axhline(0.25, color='gray', lw=0.8, ls=':', alpha=0.6)
        ax.set_title(f'r={r}, s={s}', fontsize=12)
        ax.set_xlim(0, 2000); ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
        if j == 0: ax.set_ylabel(f'r={r}\nhaplotype frequency', fontsize=10)
        if i == 1: ax.set_xlabel('t (generation)', fontsize=10)
        if i == 0 and j == 1:
            ax.legend(fontsize=7, ncol=2, loc='upper right')
fig.suptitle('Figure 1: haplotype frequency trajectories — beneficial combination 11 maintained (SVPG thin solid / DVPG thick dashed t/2 / SLiM dotted±SE; '
             'gray dotted = linkage equilibrium 0.25)', fontsize=12.5, fontweight='bold')
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(f'{FIG}/exp2_figC_hap.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 1 (hap) saved")

# ============ Figure 2 (second figure in the report): AF trajectories (2×2 grid) ============
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for i, r in enumerate(rs):
    for j, s in enumerate(ss):
        ax = axes[i][j]
        d = det[(r, s)]
        slim = slim_summary(os.path.join(DATA, key_of(r, s)))
        c = S_COLORS[s]
        # AF1 (site 0 neutral)
        ax.plot(t_sv, vc.AF_series(d['ps'], 0), '-', color=c, lw=1.2, alpha=0.55)
        ax.plot(t_dv_half, vc.AF_series(d['rs_dv'], 0), '--', color=c, lw=1.8, alpha=0.8)
        ax.plot(slim['gen'], slim['af1m'], ':', color=c, lw=1.1, alpha=0.6)
        ax.fill_between(slim['gen'], slim['af1m']-slim['af1se'], slim['af1m']+slim['af1se'],
                        color=c, alpha=0.08)
        # AF2 (site 1 selected)
        ax.plot(t_sv, vc.AF_series(d['ps'], 1), '-', color='#111111', lw=1.3, alpha=0.65)
        ax.plot(t_dv_half, vc.AF_series(d['rs_dv'], 1), '--', color='#111111', lw=2.0, alpha=0.8)
        ax.plot(slim['gen'], slim['af2m'], ':', color='#111111', lw=1.2, alpha=0.7)
        ax.fill_between(slim['gen'], slim['af2m']-slim['af2se'], slim['af2m']+slim['af2se'],
                        color='#111111', alpha=0.10)
        ax.set_title(f'r={r}, s={s}', fontsize=12)
        ax.set_xlim(0, 2000); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        if j == 0: ax.set_ylabel(f'r={r}\nAF', fontsize=10)
        if i == 1: ax.set_xlabel('t (generation)', fontsize=10)
fig.suptitle('Figure 2: AF trajectories — two-locus recombination-selection (colored = site 0 neutral AF1, black = site 1 selected AF2; '
             'SVPG solid / DVPG dashed t/2 / SLiM dotted±SE)', fontsize=12.5, fontweight='bold')
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(f'{FIG}/exp2_figA_AF.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 2 (AF) saved")

# ============ Figure 3 (third figure in the report): D12(t) LD trajectories (2×2 grid) ============
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for i, r in enumerate(rs):
    for j, s in enumerate(ss):
        ax = axes[i][j]
        d = det[(r, s)]
        slim = slim_summary(os.path.join(DATA, key_of(r, s)))
        c = S_COLORS[s]
        ax.plot(t_sv, vc.D_series(d['ps']), '-', color=c, lw=2.2, alpha=0.8, label='SVPG')
        ax.plot(t_dv_half, vc.D_series(d['rs_dv']), '--', color=c, lw=1.7, alpha=0.75, label='DVPG (t/2)')
        ax.plot(slim['gen'], slim['dm'], ':', color=c, lw=1.3, alpha=0.8, label='SLiM mean')
        ax.fill_between(slim['gen'], slim['dm']-slim['dse'], slim['dm']+slim['dse'],
                        color=c, alpha=0.12)
        ax.axhline(0, color='gray', lw=0.8, ls='-', alpha=0.5)
        ax.axhline(0.2, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax.set_title(f'r={r}, s={s}', fontsize=12)
        ax.set_xlim(0, 2000)
        ax.grid(alpha=0.25)
        if j == 0: ax.set_ylabel(f'r={r}\nD₁₂', fontsize=10)
        if i == 1: ax.set_xlabel('t (generation)', fontsize=10)
        if i == 0 and j == 1: ax.legend(fontsize=7)
fig.suptitle('Figure 3: LD trajectories D₁₂(t) — two-locus recombination-selection (initial D(0)=0.2; gray dotted lines at 0 and D(0) as references)',
             fontsize=12.5, fontweight='bold')
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(f'{FIG}/exp2_figB_LD.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 3 (LD) saved")

# ============ Summary: LD half-life + final state ============
print("\n===== Summary =====")
print(f"{'r':>6}{'s':>7} | {'AF1*':>7}{'AF2*':>7}{'D_end':>8}{'D_half_t':>9}")
for r in rs:
    for s in ss:
        d = det[(r, s)]
        slim = slim_summary(os.path.join(DATA, key_of(r, s)))
        dser = vc.D_series(d['ps'])
        t_arr = t_sv
        half_t = t_arr[np.argmax(np.abs(dser) < 0.1)] if np.any(np.abs(dser) < 0.1) else np.nan
        print(f'{r:6.3f}{s:7.3f} | {vc.AF_series(d["ps"],0)[-1]:7.4f}{vc.AF_series(d["ps"],1)[-1]:7.4f}'
              f'{dser[-1]:8.4f}{half_t:9.0f}')
print("ALL FIGURES DONE")
