#!/usr/bin/env python3
"""
Experiment 1 equilibrium frequency comparison figure (paper version) — modeled after make_paper_panels71.fig71B style
VPG analytic equilibrium p* (blue squares) vs SLiM t=2000 mean (red circles mean±2SE)
Updated parameters: mu in {3e-4, 1e-3, 3e-3} x s in {0.005, 0.01, 0.05}  (9 groups, p0=0.01)
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vpg_core_exp1 as vc1

DATA = os.path.join(_HERE, 'data')
OUT = os.path.join(_HERE, 'figs')
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 17
plt.rcParams['axes.labelsize'] = 19
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

mus = [3e-4, 1e-3, 3e-3]
ss  = [0.005, 0.01, 0.05]

fig, ax = plt.subplots(figsize=(6.5, 4.6))
labels = []; vpg_eq = []; slim_eq = []; slim_se = []
for mu in mus:
    for s in ss:
        key = f"mu{str(mu).replace('.','p')}_s{str(s).replace('.','p')}_p00p01"
        arr = np.load(f'{DATA}/slim1_{key}.npy')
        p = arr[:, -1, 1]
        slim_eq.append(p.mean()); slim_se.append(p.std()/np.sqrt(p.shape[0]))
        vpg_eq.append(vc1.eq_balance(mu, mu, s))
        labels.append(f'$\\mu$={mu:.0e}\n$s$={s}')
x = np.arange(len(labels))
ax.errorbar(x, slim_eq, yerr=[2*e for e in slim_se], fmt='o', ms=7, color='#E74C3C',
            capsize=3, label='SLiM $t$=2000 (mean±2SE)')
ax.plot(x, vpg_eq, 's', ms=8, color='#2E86AB', label='VPG analytic equilibrium $p^*$')
for xi, a, b in zip(x, vpg_eq, slim_eq):
    ax.plot([xi, xi], [a, b], color='0.6', lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.55, 1.05)          # new parameter set has minimum p*=0.681 (μ=3e-3, s=0.005); the original 0.75 lower bound would clip it
ax.set_ylabel('Equilibrium frequency $p^*$')
ax.set_title('B  Equilibrium frequency comparison (9 parameter sets)')
ax.legend(loc='lower left', fontsize=12)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f'{OUT}/exp1_figB_balance_paper.png', bbox_inches='tight')
print("saved:", f'{OUT}/exp1_figB_balance_paper.png')

# Summary of numerical values
print(f"{'params':<16}{'p*(VPG)':>10}{'p_SLiM':>10}{'2SE':>8}")
for mu in mus:
    for s in ss:
        key = f"mu{str(mu).replace('.','p')}_s{str(s).replace('.','p')}_p00p01"
        arr = np.load(f'{DATA}/slim1_{key}.npy')
        p = arr[:, -1, 1]
        m, se = p.mean(), p.std()/np.sqrt(p.shape[0])
        peq = vc1.eq_balance(mu, mu, s)
        print(f"μ={mu:g} s={s:g}  {peq:10.4f}{m:10.4f}{2*se:8.4f}")
