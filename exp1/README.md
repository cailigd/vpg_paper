# exp1 — Experiment 1: single-locus mutation–selection balance (L = 1)

Deterministic VPG (single-state SVPG and dual-state DVPG) compared with SLiM
Wright–Fisher simulations at one biallelic locus, with reversible mutation
(u = v = mu) and additive selection (s).

## Files

| File | Purpose |
|---|---|
| `vpg_core_exp1.py` | Deterministic core (library, imported by the other scripts; no CLI). `svpg_rhs` / `integrate_svpg`: dp/dt = u(1-p) - v p + s p (1-p). `dvpg_rhs` / `integrate_dvpg`: coupled (g, rho) system with coupling flux J_c = gamma (rho - g). `eq_balance`: analytic equilibrium p*. |
| `run_slim_exp1.py` | SLiM runner for the 11 parameter sets: the 3x3 grid mu in {3e-4, 1e-3, 3e-3} x s in {0.005, 0.01, 0.05} at p0 = 0.01, plus two bidirectional-convergence sets (p0 = 0.99; mu = 1e-3 / 3e-3, s = 0.01). SLiM diploid Ne = 1000 (2Ne = 2000 haplotypes), T = 2000, one sample every 10 generations (201 points), 100 reps per set, seed = 1000 + 7i. Writes `data/slim1_<mu>_s<s>_p0<p0>.npy`, shape (100, 201, 2) = (reps, time, [generation, p]). |
| `make_figs_exp1.py` | Summary figures (CJK fonts, Chinese labels) -> `figs/exp1_figA_traj.png` (AF trajectories for three s), `figs/exp1_figB_balance.png` (equilibrium frequency, VPG vs SLiM), `figs/exp1_figC_coupling.png` (dual-state g vs rho). |
| `make_fig_exp1_balance.py` | Alternative paper-style rendering of the equilibrium panel -> `figs/exp1_figB_balance_paper.png` (SLiM mean +/- 2SE). |
| `make_fig_exp1_arial.py` | **Paper figure for experiment 1 (integration panel)** (English labels, Arial) -> `fig_exp1.png` / `fig_exp1.pdf`. Panels: A = AF trajectories (mu = 1e-3, gamma = 100), B = equilibrium frequency for the 11 parameter sets, C = dual-state coupling gamma scan. Reads `data/` only. Runtime ~4 s. |
| `data/*.npy` | SLiM results for the 11 parameter sets (100 reps each). |

## How to run

```bash
python3 make_fig_exp1_arial.py        # paper figure (experiment 1) -> fig_exp1.png / fig_exp1.pdf
python3 make_figs_exp1.py             # figures -> figs/*.png
python3 make_fig_exp1_balance.py      # alternative equilibrium panel
python3 run_slim_exp1.py 100          # regenerate SLiM data with 100 reps
```

* Every script resolves paths relative to its own directory (`_HERE`), so the
  package can be cloned anywhere and each script can be started from any
  working directory: `python3 exp1/make_fig_exp1_arial.py`.
* SLiM runners need SLiM 4.2. The binary is taken from the `SLIM_BIN`
  environment variable, falling back to `/usr/local/bin/slim`:
  `SLIM_BIN=/path/to/slim python3 run_slim_exp1.py 100`.
* Runners skip a parameter set whose `.npy` output already exists; delete the
  file first to recompute it.
* `run_slim_exp1.py` defaults to **100** reps, matching the shipped data
  (`python3 run_slim_exp1.py [n_reps]`). Reference cost: ~1.6 s per SLiM rep at
  Ne = 1000; the runner uses `multiprocessing.Pool(2)`.
* Python dependencies: `numpy`, `matplotlib` (see `../requirements.txt`).
* `make_fig_exp1_arial.py` needs Arial installed under
  `/usr/share/fonts/truetype/msttcorefonts/` (`ttf-mscorefonts-installer`); it
  registers the fonts at runtime, sets `pdf.fonttype = 42` and uses no mathtext,
  so the PDF text stays editable in Illustrator.
* The `figs/*.png` figures are labelled in Chinese, so a CJK font must be
  installed (`WenQuanYi Zen Hei` or `Noto Sans CJK SC`); the `fig_exp*` figures
  are English/Arial and do not need a CJK font.
* Shared figure convention of `fig_exp1` / `fig_exp2` / `fig_exp3`: SVPG solid,
  DVPG dashed (its time axis is drawn at half speed, t/2), SLiM dotted with a
  mean +/- SE band.
