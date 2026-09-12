#!/usr/bin/env python3
"""
Figure for experiment 1 — single-locus mutation-selection balance (L = 1).
Integration panel version (formerly make_fig71_arial.py in the separate figure
package): writes fig_exp1.png / fig_exp1.pdf.
The PDF embeds Arial TrueType (pdf.fonttype=42), so the text stays editable in
Illustrator.
Layout: 3 rows x 1 column (panels A/B/C), one composite figure drawn directly
with matplotlib; all panel labels are English.
Panels:
  A: allele-frequency trajectories, mu = 1e-3, gamma = 100 (three values of s)
  B: equilibrium frequency for 11 parameter sets (3x3 mu x s matrix plus the
     two p0 = 0.99 sets); SLiM mean +/- 1 SE vs the VPG analytic p*
  C: dual-state coupling, gamma scan (solid: g, dashed: rho)
Includes built-in text-overlap / legend-occlusion checks.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vpg_core_exp1 as vc

OUT = HERE
os.makedirs(OUT, exist_ok=True)
from matplotlib import font_manager as _fm
for _f in ['/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',
           '/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf',
           '/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf',
           '/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf']:
    if os.path.exists(_f):          # tolerant: fall back to DejaVu if Arial is absent
        _fm.fontManager.addfont(_f)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

DATA = os.path.join(HERE, 'data')
u = v = 1e-3
T = 2000.0; dt = 0.01

# ============ overlap-check helpers ============
def check_overlaps(fig, ax_list, tag):
    """Check 1) text-text overlaps and 2) the fraction of curve points falling inside each legend box."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    # --- text-text (within one axes; tick labels legitimately sit next to axis labels) ---
    for ax in ax_list:
        texts = []
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            bb = t.get_window_extent(renderer)
            texts.append((t.get_text()[:28], bb))
        # panel labels A/B/C are checked here as well (they may overlap the title)
        # tick labels
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            bb = lab.get_window_extent(renderer)
            texts.append((lab.get_text()[:28], bb))
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                b1, b2 = texts[i][1], texts[j][1]
                if b1.overlaps(b2):
                    inter = b1.intersection(b2)
                    if inter is not None and inter.width > 3 and inter.height > 3:
                        problems.append(f'[{tag}] text-text: "{texts[i][0]}" vs "{texts[j][0]}" '
                                        f'inter {inter.width:.0f}x{inter.height:.0f}px')
    # --- legend occluding curves: fraction of curve points inside the legend box ---
    for ax in ax_list:
        for leg in ax.get_legend_handles_labels()[0]:
            pass
        # legend objects
        leg_objs = [c for c in ax.get_children() if isinstance(c, Legend)]
        for leg in leg_objs:
            bb = leg.get_window_extent(renderer)
            inv = ax.transData.inverted()
            # only the lines of this axes
            for ln in ax.lines:
                xd = np.asarray(ln.get_xdata(), dtype=float)
                yd = np.asarray(ln.get_ydata(), dtype=float)
                if len(xd) < 2:
                    continue
                # subsample
                idx = np.linspace(0, len(xd)-1, 300).astype(int)
                pts_disp = ax.transData.transform(np.column_stack([xd[idx], yd[idx]]))
                inside = np.sum((pts_disp[:,0] >= bb.x0) & (pts_disp[:,0] <= bb.x1) &
                                (pts_disp[:,1] >= bb.y0) & (pts_disp[:,1] <= bb.y1))
                if inside > 0.25 * 300:
                    problems.append(f'[{tag}] legend occludes curve: {leg.get_title().get_text() or "untitled"} '
                                    f'({inside/3:.0f}% of points inside)')
    if problems:
        print('  [!] problems found:')
        for p in problems:
            print('   ', p)
    else:
        print('  OK: no text overlap / legend occlusion')
    return problems

# ============ A. allele-frequency trajectories ============
fig = plt.figure(figsize=(6.5, 8.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.46)
axA = fig.add_subplot(gs[0])
for s, c in [(0.005, '#2E86AB'), (0.01, '#27AE60'), (0.05, '#E74C3C')]:
    ps = vc.integrate_svpg(0.01, u, v, s, T, dt)
    g1, r1 = vc.integrate_dvpg(0.01, u, v, s, 100.0, 4000.0, dt)
    t = np.arange(len(ps)) * dt
    td = np.arange(len(r1)) * dt / 2
    axA.plot(t, ps, '-', color=c, lw=1.3, alpha=0.75)
    axA.plot(td, r1, '--', color=c, lw=1.2, alpha=0.6)
    fn = f'{DATA}/slim1_mu0p001_s{str(s).replace(".","p")}_p00p01.npy'
    arr = np.load(fn); gen = arr[0,:,0]; p = arr[:,:,1]
    m = p.mean(0); se = p.std(0)/np.sqrt(p.shape[0])
    axA.plot(gen, m, ':', color=c, lw=1.1, alpha=0.85)
    axA.fill_between(gen, m-se, m+se, color=c, alpha=0.12)
    axA.axhline(vc.eq_balance(u, v, s), color=c, lw=0.8, ls=':', alpha=0.4)
axA.set_xlim(0, 2000); axA.set_ylim(0, 1.05)
axA.set_xlabel('t (generation)'); axA.set_ylabel('Allele frequency p(t)')
axA.set_title('Allele frequency, μ=1e-3, γ=100')
axA.text(0.02, 0.95, 'A', transform=axA.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
handlesA = [plt.Line2D([0],[0], color=c, lw=2.2, label=f's={s}') for s, c in
            [(0.005, '#2E86AB'), (0.01, '#27AE60'), (0.05, '#E74C3C')]]
axA.legend(handles=handlesA, ncol=3, fontsize=9.5, loc='lower right', framealpha=0.9,
           title='solid: SVPG, dashed: DVPG ρ, dotted: SLiM (mean±SE)',
           title_fontsize=8.5)
axA.grid(alpha=0.25)

# ============ B. equilibrium frequency (11 parameter sets) ============
axB = fig.add_subplot(gs[1])
mus = [3e-4, 1e-3, 3e-3]; ss = [0.005, 0.01, 0.05]
labels = []; vpg_eq = []; slim_eq = []; slim_se = []
for mu in mus:
    for s in ss:
        key = f"mu{str(mu).replace('.','p')}_s{str(s).replace('.','p')}_p00p01"
        arr = np.load(f'{DATA}/slim1_{key}.npy')
        p = arr[:, -1, 1]
        slim_eq.append(p.mean()); slim_se.append(p.std()/np.sqrt(p.shape[0]))
        vpg_eq.append(vc.eq_balance(mu, mu, s))
        labels.append(f'μ={mu:.0e}\ns={s}')
# p0 = 0.99 bidirectional-convergence sets
for mu in [1e-3, 3e-3]:
    key = f"mu{str(mu).replace('.','p')}_s0p01_p00p99"
    arr = np.load(f'{DATA}/slim1_{key}.npy')
    p = arr[:, -1, 1]
    slim_eq.append(p.mean()); slim_se.append(p.std()/np.sqrt(p.shape[0]))
    vpg_eq.append(vc.eq_balance(mu, mu, 0.01))
    labels.append(f'p0=0.99\nμ={mu:.0e}')
x = np.arange(len(labels))
axB.errorbar(x, slim_eq, yerr=slim_se, fmt='o', ms=4, color='#E74C3C',
             capsize=4, elinewidth=1.2, label='SLiM (mean±SE)')
axB.plot(x, vpg_eq, 's', ms=5, color='#2E86AB', label='VPG p*')
for xi, a, b in zip(x, vpg_eq, slim_eq):
    axB.plot([xi, xi], [a, b], color='0.6', lw=1.0)
axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=7.8)
axB.set_ylim(0.55, 1.02)
axB.set_ylabel('Equilibrium frequency p*')
axB.set_title('Equilibrium frequency (11 parameter sets)')
axB.text(0.02, 0.95, 'B', transform=axB.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
axB.legend(loc='lower left', fontsize=9)
axB.grid(alpha=0.25)

# ============ C. gamma scan (g / rho trajectories) ============
axC = fig.add_subplot(gs[2])
muC = 0.01; sC = 0.05; p0C = 0.01; TC = 500.0
gammas = [0.01, 0.1, 1, 10, 100]
colors = ['#8E44AD', '#2E86AB', '#27AE60', '#E67E22', '#E74C3C']
for gv, c in zip(gammas, colors):
    g1s, r1s = vc.integrate_dvpg(p0C, muC, muC, sC, gv, TC, dt)
    t = np.arange(len(g1s)) * dt
    axC.plot(t, g1s, '-', color=c, lw=1.3, alpha=0.8)
    axC.plot(t, r1s, '--', color=c, lw=1.2, alpha=0.65)
axC.set_xlim(0, 500); axC.set_ylim(0, 1.05)
axC.set_xlabel('t (generation)'); axC.set_ylabel('Frequency')
axC.set_title('Dual-state coupling, μ=0.01, s=0.05')
axC.text(0.02, 0.95, 'C', transform=axC.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
handlesC = [plt.Line2D([0],[0], color=c, lw=2.0, label=f'γ={gv:g}') for gv, c in zip(gammas, colors)]
axC.legend(handles=handlesC, ncol=3, fontsize=8.5, loc='lower right', framealpha=0.9,
           title='solid: g, dashed: ρ',
           title_fontsize=8.5)
axC.grid(alpha=0.25)

plt.savefig(f'{OUT}/fig_exp1.png', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_exp1.pdf', bbox_inches='tight')
print(f'fig_exp1 saved: png {os.path.getsize(OUT+"/fig_exp1.png")} bytes, pdf {os.path.getsize(OUT+"/fig_exp1.pdf")} bytes')

# ============ overlap check ============
print('fig_exp1 overlap check:')
check_overlaps(fig, [axA, axB, axC], 'fig_exp1')
