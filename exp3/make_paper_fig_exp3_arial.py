#!/usr/bin/env python3
"""
Figure for experiment 3 — three-locus integrated dynamics and its stochastic
extension (one composite figure drawn directly with matplotlib).
Integration panel version (formerly make_paper_fig73_arial.py in the separate
figure package): writes fig_exp3.png / fig_exp3.pdf.
Layout: top row A|B|C, bottom row D|E|F (2x6 gridspec, each panel spans two
columns, ~2.64 x 3.09 in each).
Shared line-style convention (same as fig_exp1 / fig_exp2):
    VPG (SVPG) solid '-' ; DVPG rho dashed '--' ; SLiM dotted ':' (mean +/- SE)
Bottom panels D/E/F compare the Langevin (drift) extension, one Ne pair each:
    D = SLiM Ne=500  vs VPG Ne=1000
    E = SLiM Ne=1000 vs VPG Ne=2000
    F = SLiM Ne=3000 vs VPG Ne=6000
Data: slim3_ne500.npy / slim3_main.npy / slim3_ne3000.npy and
      langevin3_ne1000.npy / langevin3_ne2000.npy / langevin3_ne6000.npy
      (100 reps each)
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vpg_core_exp3 as vc

OUT = HERE
DATA = os.path.join(HERE, 'data')
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
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

mu = 0.001; s = np.array([0.005, 0.0, -0.005])
radj = np.array([0.01, 0.05]); L = 3
p0 = np.zeros(8); p0[0] = 0.4; p0[7] = 0.4
for k in range(1, 7): p0[k] = 0.2 / 6
M = vc.mut_matrix(mu, mu); U = vc.U_vec(s)
T = 2000.0; dt = 0.01; GAMMA = 100.0

HAP_LABELS = [vc.HAP_LABELS[h] for h in range(8)]
HAP_COLORS = [plt.cm.tab20(2 * k) for k in range(8)]
SITE_NAMES = ['Site 0 (+)', 'Site 1 (0)', 'Site 2 (−)']
SITE_COLORS = ['#E74C3C', '#27AE60', '#2E86AB']
PAIR_NAMES = ['D12', 'D23', 'D13']
PAIR_COLORS = ['#8E44AD', '#E67E22', '#C2185B']

STY_SLIM = dict(ls=':', color=None, lw=1.0, alpha=0.85)   # SLiM dotted (same as A/B/C)
STY_VPG = dict(ls='-', color=None, lw=1.3, alpha=0.75)    # VPG solid (same as A/B/C)
BAND_ALPHA = 0.12                                         # equal weight for both SE bands

def load_slim(fn):
    arr = np.load(fn)
    return arr[0, :, 0], arr[:, :, 1:]

def af_d_from_haps(harr):
    """harr: (reps, T, 8) -> mean/SE of AF (3 sites) and D (3 pairs); returns (3, T) each.
    Defined locally (not imported from make_figs_exp3_stochastic, whose
    module-level plotting code would override rcParams and write stray PNGs)."""
    reps = harr.shape[0]
    af = np.zeros((reps, harr.shape[1], 3))
    for site in range(3):
        mask = np.array([(h >> site) & 1 for h in range(8)])
        af[:, :, site] = harr @ mask
    d = np.zeros((reps, harr.shape[1], 3))
    for kk, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)]):
        mask11 = np.array([1 if ((h >> i) & 1) and ((h >> j) & 1) else 0 for h in range(8)])
        pi = np.array([(h >> i) & 1 for h in range(8)])
        pj = np.array([(h >> j) & 1 for h in range(8)])
        d[:, :, kk] = (harr @ mask11) - (harr @ pi) * (harr @ pj)
    return (af.mean(0), af.std(0) / np.sqrt(reps),
            d.mean(0), d.std(0) / np.sqrt(reps))

ps = vc.integrate_svpg(p0, M, U, radj, L, T, dt)
gs_dv, rs_dv = vc.integrate_dvpg(p0, M, U, radj, L, GAMMA, 4000.0, dt)
t = np.arange(len(ps)) * dt; td = np.arange(len(rs_dv)) * dt / 2
gen, hslim = load_slim(f'{DATA}/slim3_main.npy')

fig = plt.figure(figsize=(10.5, 8.6))
gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.90,
                      left=0.075, right=0.985, top=0.955, bottom=0.075)

def panel_label(ax, txt, fs=16):
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=fs,
            fontweight='bold', va='top', ha='left')

# ---------- A. 8 haplotype frequencies ----------
ax = fig.add_subplot(gs[0, 0:2])
for k in range(8):
    c = HAP_COLORS[k]
    ax.plot(t, ps[:, k], '-', color=c, lw=1.0, alpha=0.7)
    ax.plot(td, rs_dv[:, k], '--', color=c, lw=0.9, alpha=0.45)
    m = hslim[:, :, k].mean(0); se = hslim[:, :, k].std(0)/np.sqrt(hslim.shape[0])
    ax.plot(gen, m, ':', color=c, lw=0.8, alpha=0.85)
    ax.fill_between(gen, m-se, m+se, color=c, alpha=0.06)
ax.set_xlim(0, 2000); ax.set_ylim(0, 0.45)
ax.set_xlabel('t (generation)'); ax.set_ylabel('Haplotype frequency')
ax.set_title('Haplotype frequencies', fontsize=13)
panel_label(ax, 'A')
handles = [plt.Line2D([0],[0], color=HAP_COLORS[k], lw=2, label=HAP_LABELS[k]) for k in range(8)]
ax.legend(handles=handles, ncol=4, fontsize=8, loc='upper right', framealpha=0.9,
          handlelength=1.1, columnspacing=0.6, labelspacing=0.3)
ax.grid(alpha=0.2)

# ---------- B. AF ----------
ax = fig.add_subplot(gs[0, 2:4])
for i in range(3):
    c = SITE_COLORS[i]
    mask = np.array([(h >> i) & 1 for h in range(8)])
    af = hslim @ mask
    m = af.mean(0); se = af.std(0)/np.sqrt(af.shape[0])
    ax.plot(gen, m, ':', color=c, lw=1.0, alpha=0.85)
    ax.fill_between(gen, m-se, m+se, color=c, alpha=0.10)
    ax.plot(t, vc.AF_series(ps, i), '-', color=c, lw=1.3, alpha=0.75)
    ax.plot(td, vc.AF_series(rs_dv, i), '--', color=c, lw=1.1, alpha=0.55)
ax.set_xlim(0, 2000); ax.set_ylim(0, 1.05)
ax.set_xlabel('t (generation)'); ax.set_ylabel('Allele frequency')
ax.set_title('Allele frequencies', fontsize=13)
panel_label(ax, 'B')
handles = [plt.Line2D([0],[0], color=c, lw=2, label=n) for c, n in zip(SITE_COLORS, SITE_NAMES)]
ax.legend(handles=handles, ncol=1, fontsize=8, loc='center right', framealpha=0.9)
ax.grid(alpha=0.2)

# ---------- C. LD ----------
ax = fig.add_subplot(gs[0, 4:6])
for kk, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)]):
    c = PAIR_COLORS[kk]
    mask11 = np.array([1 if ((h>>i)&1) and ((h>>j)&1) else 0 for h in range(8)])
    pi = np.array([(h>>i)&1 for h in range(8)]); pj = np.array([(h>>j)&1 for h in range(8)])
    D = (hslim @ mask11) - (hslim @ pi) * (hslim @ pj)
    m = D.mean(0); se = D.std(0)/np.sqrt(D.shape[0])
    ax.plot(gen, m, ':', color=c, lw=1.0, alpha=0.85)
    ax.fill_between(gen, m-se, m+se, color=c, alpha=0.10)
    ax.plot(t, vc.D_pair_series(ps, i, j), '-', color=c, lw=1.3, alpha=0.75)
    ax.plot(td, vc.D_pair_series(rs_dv, i, j), '--', color=c, lw=1.1, alpha=0.55)
ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax.set_xlim(0, 200)
ax.set_xlabel('t (generation)'); ax.set_ylabel('Linkage disequilibrium Dij')
ax.set_title('Linkage disequilibrium', fontsize=13)
panel_label(ax, 'C')
handles = [plt.Line2D([0],[0], color=c, lw=2, label=n) for c, n in zip(PAIR_COLORS, PAIR_NAMES)]
ax.legend(handles=handles, ncol=1, fontsize=8, loc='upper right', framealpha=0.9)
ax.grid(alpha=0.2)

# ---------- D / E / F. stochastic extension (allele frequency), one Ne pair per panel ----------
groups = [
    ('D', 'SLiM Ne=500 vs VPG Ne=1000',  'slim3_ne500.npy',   'langevin3_ne1000.npy'),
    ('E', 'SLiM Ne=1000 vs VPG Ne=2000', 'slim3_main.npy',    'langevin3_ne2000.npy'),
    ('F', 'SLiM Ne=3000 vs VPG Ne=6000', 'slim3_ne3000.npy',  'langevin3_ne6000.npy'),
]
for span, (lab, ttl, sfn, vfn) in zip([(0, 2), (2, 4), (4, 6)], groups):
    axD = fig.add_subplot(gs[1, span[0]:span[1]])
    gen_s, hs = load_slim(f'{DATA}/{sfn}')
    hl = np.load(f'{DATA}/{vfn}')
    gen_v = np.arange(hl.shape[1]) * 10.0
    af_s_m, af_s_se, _, _ = af_d_from_haps(hs)
    af_v_m, af_v_se, _, _ = af_d_from_haps(hl)
    for i in range(3):
        c = SITE_COLORS[i]
        axD.plot(gen_s, af_s_m[:, i], **{**STY_SLIM, 'color': c})            # SLiM dotted
        axD.fill_between(gen_s, af_s_m[:, i]-af_s_se[:, i],
                         af_s_m[:, i]+af_s_se[:, i], color=c, alpha=BAND_ALPHA)
        axD.plot(gen_v, af_v_m[:, i], **{**STY_VPG, 'color': c})             # VPG solid
        axD.fill_between(gen_v, af_v_m[:, i]-af_v_se[:, i],
                         af_v_m[:, i]+af_v_se[:, i], color=c, alpha=BAND_ALPHA)
    axD.set_xlim(0, 1000); axD.set_ylim(0, 1.05)
    axD.set_xlabel('t (generation)'); axD.set_ylabel('Allele frequency')
    axD.set_title(ttl, fontsize=12.5)
    panel_label(axD, lab, fs=15)
    handles = [plt.Line2D([0],[0], ls=':', color='0.25', lw=2, label='SLiM'),
               plt.Line2D([0],[0], ls='-', color='0.25', lw=2, label='VPG')]
    axD.legend(handles=handles, ncol=1, fontsize=8.5, loc='upper right',
               framealpha=0.9, handlelength=1.4, labelspacing=0.3)
    axD.grid(alpha=0.2)

plt.savefig(f'{OUT}/fig_exp3.png', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_exp3.pdf', bbox_inches='tight')
print(f'fig_exp3 saved: png {os.path.getsize(OUT+"/fig_exp3.png")} bytes, pdf {os.path.getsize(OUT+"/fig_exp3.pdf")} bytes')
