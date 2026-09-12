#!/usr/bin/env python3
"""
Figure for experiment 2 — two-locus recombination-selection dynamics (L = 2).
Integration panel version (formerly make_fig72_arial.py in the separate figure
package): writes fig_exp2.png / fig_exp2.pdf.
Layout: 3 rows x 4 columns (A haplotype frequencies / B LD / C allele
frequencies; one column per parameter set), all labels in English.
Parameter sets: (r, s) = (0.001, 0.005), (0.001, 0.05), (0.01, 0.005),
(0.01, 0.05).
Shared line-style convention: SVPG thin solid / DVPG rho dashed (t/2) /
SLiM dotted (mean +/- SE).
Both PNG (raster, for LaTeX) and PDF (vector) are written, plus built-in
text-overlap / legend-occlusion checks.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D as MplLine2D
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vpg_core_exp2 as vc

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
plt.rcParams['axes.labelsize'] = 14.5
plt.rcParams['axes.titlesize'] = 14.5
plt.rcParams['legend.fontsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 11.5
plt.rcParams['ytick.labelsize'] = 11.5
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

DATA = os.path.join(HERE, 'data')
p0 = np.array([0.45, 0.05, 0.05, 0.45])
T = 2000.0; dt = 0.01
GAMMA = 100.0
GRID = [(0.001, 0.005), (0.001, 0.05), (0.01, 0.005), (0.01, 0.05)]
HAP_LABELS = ['00', '01', '10', '11']
HAP_COLORS = ['#2E86AB', '#8E44AD', '#E67E22', '#E74C3C']

def key_of(r, s):
    return f'{DATA}/slim2_r{str(r).replace(".","p")}_s{str(s).replace(".","p")}.npy'

def load_slim(r, s):
    arr = np.load(key_of(r, s)); gen = arr[0,:,0]; h = arr[:,:,1:]
    return gen, h

def check_overlaps(fig, ax_list, tag):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for ax in ax_list:
        texts = []
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            bb = t.get_window_extent(renderer)
            texts.append((t.get_text()[:28], bb))
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
        for leg in [c for c in ax.get_children() if isinstance(c, Legend)]:
            bb = leg.get_window_extent(renderer)
            for ln in ax.lines:
                xd = np.asarray(ln.get_xdata(), dtype=float)
                yd = np.asarray(ln.get_ydata(), dtype=float)
                if len(xd) < 2:
                    continue
                idx = np.linspace(0, len(xd)-1, 300).astype(int)
                pts_disp = ax.transData.transform(np.column_stack([xd[idx], yd[idx]]))
                inside = np.sum((pts_disp[:,0] >= bb.x0) & (pts_disp[:,0] <= bb.x1) &
                                (pts_disp[:,1] >= bb.y0) & (pts_disp[:,1] <= bb.y1))
                if inside > 0.25 * 300:
                    problems.append(f'[{tag}] legend occludes curve ({inside/3:.0f}% of points inside)')
    for leg in [c for c in fig.get_children() if isinstance(c, Legend)]:
        bb = leg.get_window_extent(renderer)
        ttl = leg.get_title().get_text()
        for ax in ax_list:
            for ln in ax.lines:
                xd = np.asarray(ln.get_xdata(), dtype=float)
                yd = np.asarray(ln.get_ydata(), dtype=float)
                if len(xd) < 2:
                    continue
                idx = np.linspace(0, len(xd)-1, 300).astype(int)
                pts_disp = ax.transData.transform(np.column_stack([xd[idx], yd[idx]]))
                inside = np.sum((pts_disp[:,0] >= bb.x0) & (pts_disp[:,0] <= bb.x1) &
                                (pts_disp[:,1] >= bb.y0) & (pts_disp[:,1] <= bb.y1))
                if inside > 0.25 * 300:
                    problems.append(f'[{tag}] fig-legend "{ttl}" occludes curve ({inside/3:.0f}% of points inside, '
                                    f'bbox=({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f}))')
    if problems:
        print('  [!] problems found:')
        for p in problems:
            print('   ', p)
    else:
        print('  OK: no text overlap / legend occlusion')
    return problems

# ============ deterministic integration (all four parameter sets) ============
det = {}
for r, s in GRID:
    radj = np.array([r])
    ps = vc.integrate_svpg(p0, radj, 2, s, T, dt)
    gs, rs_dv = vc.integrate_dvpg(p0, radj, 2, s, GAMMA, 2*T, dt)
    det[(r, s)] = dict(ps=ps, rs_dv=rs_dv)

t_sv = np.arange(len(det[GRID[0]]['ps'])) * dt
t_dv = np.arange(len(det[GRID[0]]['rs_dv'])) * dt / 2.0

# ============ layout: 3 rows x 4 columns ============
fig = plt.figure(figsize=(14.5, 11.0))
gs = fig.add_gridspec(3, 4, hspace=0.90, wspace=0.45,
                      top=0.80, bottom=0.075, left=0.075, right=0.975)

def setup_ax(ax, r, s, ylab, show_y=True):
    ax.set_title(f'r={r}, s={s}', fontsize=13.5)
    ax.set_xlim(0, 2000)
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel('t (generation)', fontsize=13.5)
    if show_y:
        ax.set_ylabel(ylab, fontsize=13.5)
    else:
        ax.tick_params(labelleft=False)   # non-leftmost columns: hide y tick labels
    ax.grid(alpha=0.25)

# ---------- A. haplotype frequencies ----------
for k, (r, s) in enumerate(GRID):
    ax = fig.add_subplot(gs[0, k])
    d = det[(r, s)]
    gen, h = load_slim(r, s)
    for kk in range(4):
        c = HAP_COLORS[kk]
        ax.plot(t_sv, d['ps'][:, kk], '-', color=c, lw=1.2, alpha=0.7)
        ax.plot(t_dv, d['rs_dv'][:, kk], '--', color=c, lw=1.1, alpha=0.55)
        m = h[:,:,kk].mean(0); se = h[:,:,kk].std(0)/np.sqrt(h.shape[0])
        ax.plot(gen, m, ':', color=c, lw=0.95, alpha=0.85)
        ax.fill_between(gen, m-se, m+se, color=c, alpha=0.08)
    setup_ax(ax, r, s, 'Haplotype\nfrequency', show_y=(k == 0))

# ---------- B. LD ----------
for k, (r, s) in enumerate(GRID):
    ax = fig.add_subplot(gs[1, k])
    d = det[(r, s)]
    gen, h = load_slim(r, s)
    p1 = h[:,:,1] + h[:,:,3]; p2 = h[:,:,2] + h[:,:,3]
    D = h[:,:,3] - p1*p2
    m = D.mean(0); se = D.std(0)/np.sqrt(D.shape[0])
    ax.plot(t_sv, vc.D_series(d['ps']), '-', color='#8E44AD', lw=1.5, alpha=0.8)
    ax.plot(t_dv, vc.D_series(d['rs_dv']), '--', color='#8E44AD', lw=1.3, alpha=0.6)
    ax.plot(gen, m, ':', color='#8E44AD', lw=1.1, alpha=0.9)
    ax.fill_between(gen, m-se, m+se, color='#8E44AD', alpha=0.12)
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.axhline(0.2, color='gray', lw=0.8, ls=':', alpha=0.5)
    setup_ax(ax, r, s, 'D12', show_y=(k == 0))

# ---------- C. AF ----------
for k, (r, s) in enumerate(GRID):
    ax = fig.add_subplot(gs[2, k])
    d = det[(r, s)]
    gen, h = load_slim(r, s)
    af1 = h[:,:,1] + h[:,:,3]; af2 = h[:,:,2] + h[:,:,3]
    for af, c in [(af1, '#E74C3C'), (af2, '#111111')]:
        m = af.mean(0); se = af.std(0)/np.sqrt(af.shape[0])
        ax.plot(gen, m, ':', color=c, lw=1.2, alpha=0.9)
        ax.fill_between(gen, m-se, m+se, color=c, alpha=0.10)
    ax.plot(t_sv, vc.AF_series(d['ps'], 0), '-', color='#E74C3C', lw=1.35, alpha=0.75)
    ax.plot(t_sv, vc.AF_series(d['ps'], 1), '-', color='#111111', lw=1.35, alpha=0.8)
    ax.plot(t_dv, vc.AF_series(d['rs_dv'], 0), '--', color='#E74C3C', lw=1.15, alpha=0.55)
    ax.plot(t_dv, vc.AF_series(d['rs_dv'], 1), '--', color='#111111', lw=1.15, alpha=0.55)
    ax.set_ylim(0, 1.05)
    setup_ax(ax, r, s, 'Allele\nfrequency', show_y=(k == 0))

# ---- row labels A/B/C (figure level) ----
fig.text(0.018, 0.70, 'A', fontsize=22, fontweight='bold', va='center')
fig.text(0.018, 0.42, 'B', fontsize=22, fontweight='bold', va='center')
fig.text(0.018, 0.14, 'C', fontsize=22, fontweight='bold', va='center')

# ---- legends (figure level, stacked at the top) ----
LS_HANDLES = [MplLine2D([0],[0], ls='-',  color='0.3', lw=2.0, label='SVPG'),
              MplLine2D([0],[0], ls='--', color='0.3', lw=2.0, label='DVPG ρ (t/2)'),
              MplLine2D([0],[0], ls=':',  color='0.3', lw=2.0, label='SLiM (mean±SE)')]
legA = fig.legend(handles=[MplLine2D([0],[0], color=c, lw=2.0, label=l) for c, l in zip(HAP_COLORS, HAP_LABELS)],
                  loc='upper center', bbox_to_anchor=(0.50, 0.985), ncol=4, fontsize=11.5,
                  title='haplotype colors (A)', title_fontsize=10, framealpha=0.9, frameon=False)
legB = fig.legend(handles=LS_HANDLES, loc='upper center', bbox_to_anchor=(0.50, 0.933), ncol=3,
                  fontsize=11.5, title='line styles (A/B/C)', title_fontsize=10, framealpha=0.9, frameon=False)
legC = fig.legend(handles=[MplLine2D([0],[0], color='#E74C3C', lw=2.0, label='AF1 site 0 (neutral)'),
                           MplLine2D([0],[0], color='#111111', lw=2.0, label='AF2 site 1 (selected)')],
                  loc='upper center', bbox_to_anchor=(0.50, 0.878), ncol=2, fontsize=11.5,
                  title='colors (C)', title_fontsize=10, framealpha=0.9, frameon=False)

plt.savefig(f'{OUT}/fig_exp2.png', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_exp2.pdf', bbox_inches='tight')
print(f'fig_exp2 saved: png {os.path.getsize(OUT+"/fig_exp2.png")} bytes, pdf {os.path.getsize(OUT+"/fig_exp2.pdf")} bytes')

# ============ overlap check ============
print('fig_exp2 overlap check:')
ax_list = [c for c in fig.get_axes() if c.get_visible()]
check_overlaps(fig, ax_list, 'fig_exp2')
