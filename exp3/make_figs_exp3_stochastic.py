#!/usr/bin/env python3
"""
Experiment 3 Figure 4: SVPG (single-state VPG) with drift (Langevin) vs SLiM — mean±SE comparison of AF and LD
Two groups: SLiM Ne=1000 vs VPG Ne=2000; SLiM Ne=3000 vs VPG Ne=6000
Focus: whether SE_VPG approaches SE_SLiM (consistent stochastic scales)
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

def af_d_from_haps(harr):
    """harr: (reps, T, 8) -> mean and SE of the af (3,) dict and d (3 pairs) dict
    Returns (af_m, af_se, d_m, d_se), each a (3, T) array"""
    reps = harr.shape[0]
    af = np.zeros((reps, harr.shape[1], 3))
    for site in range(3):
        mask = np.array([(h >> site) & 1 for h in range(8)])
        af[:, :, site] = harr @ mask
    d = np.zeros((reps, harr.shape[1], 3))
    for kk, (i, j) in enumerate([(0,1),(1,2),(0,2)]):
        mask11 = np.array([1 if ((h>>i)&1) and ((h>>j)&1) else 0 for h in range(8)])
        pi = np.array([(h>>i)&1 for h in range(8)])
        pj = np.array([(h>>j)&1 for h in range(8)])
        d[:, :, kk] = (harr @ mask11) - (harr @ pi) * (harr @ pj)
    return (af.mean(0), af.std(0)/np.sqrt(reps),
            d.mean(0), d.std(0)/np.sqrt(reps))

def load_slim(fn):
    arr = np.load(fn)          # (50, 201, 9)
    return arr[0, :, 0], arr[:, :, 1:]

def load_langevin(fn):
    arr = np.load(fn)          # (50, 201, 8)
    gen = np.arange(arr.shape[1]) * 10.0
    return gen, arr

SITE_NAMES = ['Site 0 (beneficial)', 'Site 1 (neutral)', 'Site 2 (deleterious)']
SITE_COLORS = ['#E74C3C', '#27AE60', '#2E86AB']
PAIR_NAMES = ['D₁₂', 'D₂₃', 'D₁₃']
PAIR_COLORS = ['#8E44AD', '#E67E22', '#C2185B']

groups = [
    ('SLiM Ne=1000',  'slim3_main.npy',    'VPG Ne=2000', 'langevin3_ne2000.npy'),
    ('SLiM Ne=3000',  'slim3_ne3000.npy',  'VPG Ne=6000', 'langevin3_ne6000.npy'),
]

# ============ Figure 4: 2 rows (one per Ne group) x 2 columns (AF, LD) ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for row, (slim_name, slim_fn, vpg_name, vpg_fn) in enumerate(groups):
    gen_s, hslim = load_slim(f'{DATA}/{slim_fn}')
    gen_v, hlang = load_langevin(f'{DATA}/{vpg_fn}')
    # AF / LD: mean ± SE (100 reps)
    af_s_m, af_s_se, d_s_m, d_s_se = af_d_from_haps(hslim)
    af_v_m, af_v_se, d_v_m, d_v_se = af_d_from_haps(hlang)
    # --- left column: AF ---
    ax = axes[row][0]
    for i in range(3):
        c = SITE_COLORS[i]
        ax.plot(gen_s, af_s_m[:, i], '-', color=c, lw=1.6, alpha=0.85,
                label=f'SLiM {SITE_NAMES[i]}')
        ax.fill_between(gen_s, af_s_m[:, i]-af_s_se[:, i],
                        af_s_m[:, i]+af_s_se[:, i], color=c, alpha=0.15)
        ax.plot(gen_v, af_v_m[:, i], '--', color=c, lw=1.4, alpha=0.7,
                label=f'VPG Langevin {SITE_NAMES[i]}')
        ax.fill_between(gen_v, af_v_m[:, i]-af_v_se[:, i],
                        af_v_m[:, i]+af_v_se[:, i], color=c, alpha=0.08)
    ax.set_title(f'AF — {slim_name} vs {vpg_name}', fontsize=12)
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1.05)   # AF shown for the first 1000 generations only (main rise interval)
    ax.set_xlabel('t (generation, first 1000)', fontsize=11)
    ax.grid(alpha=0.25)
    if row == 0:
        ax.set_ylabel('AF', fontsize=12)
    # --- right column: LD ---
    ax = axes[row][1]
    for kk in range(3):
        c = PAIR_COLORS[kk]
        ax.plot(gen_s, d_s_m[:, kk], '-', color=c, lw=1.6, alpha=0.85,
                label=f'SLiM {PAIR_NAMES[kk]}')
        ax.fill_between(gen_s, d_s_m[:, kk]-d_s_se[:, kk],
                        d_s_m[:, kk]+d_s_se[:, kk], color=c, alpha=0.15)
        ax.plot(gen_v, d_v_m[:, kk], '--', color=c, lw=1.4, alpha=0.7,
                label=f'VPG Langevin {PAIR_NAMES[kk]}')
        ax.fill_between(gen_v, d_v_m[:, kk]-d_v_se[:, kk],
                        d_v_m[:, kk]+d_v_se[:, kk], color=c, alpha=0.08)
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.set_title(f'LD — {slim_name} vs {vpg_name}', fontsize=12)
    ax.set_xlim(0, 200)
    ax.set_xlabel('t (generation, first 200)', fontsize=11)
    ax.grid(alpha=0.25)
    if row == 0:
        ax.set_ylabel('D', fontsize=12)
    # final-state SE comparison
    print(f"[{slim_name} vs {vpg_name}]")
    for i in range(3):
        print(f"  AF_{i}: SLiM SE={af_s_se[-1, i]:.4f} vs VPG SE={af_v_se[-1, i]:.4f} "
              f"(ratio {af_v_se[-1, i]/af_s_se[-1, i]:.2f})")
    for kk in range(3):
        print(f"  {PAIR_NAMES[kk]}: SLiM SE={d_s_se[-1, kk]:.4f} vs VPG SE={d_v_se[-1, kk]:.4f} "
              f"(ratio {d_v_se[-1, kk]/d_s_se[-1, kk]:.2f})")
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], ls='-',  color='0.3', lw=1.6, label='SLiM'),
    Line2D([0], [0], ls='--', color='0.3', lw=1.4, label='VPG Langevin'),
] + [Line2D([0], [0], ls='-', color=SITE_COLORS[i], lw=1.6, label=f'AF {SITE_NAMES[i]}') for i in range(3)] \
  + [Line2D([0], [0], ls='-', color=PAIR_COLORS[kk], lw=1.6, label=f'LD {PAIR_NAMES[kk]}') for kk in range(3)]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           ncol=4, fontsize=9, frameon=True, columnspacing=1.2)
fig.suptitle('Experiment 3 Figure 4: SVPG (single-state VPG) with drift (Langevin) vs SLiM — AF and LD mean±SE (solid SLiM / dashed VPG; '
             'VPG haploid Ne = 2×SLiM diploid Ne)', fontsize=12.5, fontweight='bold', y=0.975)
plt.tight_layout(rect=(0, 0, 1, 0.93))   # leave space at the top for legend + suptitle
plt.savefig(f'{FIG}/exp3_figD_stochastic.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 4 saved")
