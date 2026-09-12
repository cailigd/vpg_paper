# exp3 — Experiment 3: three-locus mutation–recombination–selection + drift (L = 3)

Deterministic VPG (single-state SVPG and dual-state DVPG) and stochastic
Langevin SVPG compared with SLiM Wright–Fisher simulations for three loci
(8 haplotypes). Mutation u = v = 0.001, additive selection
s = [0.005, 0, -0.005] on sites 0/1/2, recombination r12 = 0.01, r23 = 0.05,
T = 2000, one sample every 10 generations (201 points), 100 reps per group,
seed = 1000 + 7i.

Ploidy/Ne correspondence (SLiM diploid Ne -> VPG haploid Ne = 2 x Ne_slim):

| group | SLiM | VPG | figure panel |
|---|---|---|---|
| D | Ne = 500 | Ne = 1000 | `fig_exp3` panel D |
| E | Ne = 1000 | Ne = 2000 | `fig_exp3` panel E |
| F | Ne = 3000 | Ne = 6000 | `fig_exp3` panel F |

## Files

| File | Purpose |
|---|---|
| `vpg_core_exp3.py` | Deterministic core (library, imported by the other scripts; no CLI). Mutation matrix `mut_matrix`, recombination tensor `build_R_tensor` / `J_rec_tensor` / `J_rec_proj`, additive potential `U_vec`, Fisher–Rao selection flow `J_FR`, integrators `integrate_svpg` / `integrate_dvpg` (coupling flux J_c = gamma (rho - g)), series helpers `AF_series` / `D_pair_series`, and `HAP_LABELS` (h -> "site2 site1 site0"). |
| `langevin_exp3.py` | Stochastic core. dp = (J_mut + J_rec + J_sel) dt + sqrt(M_FR(p)/Ne) dW with M_FR = diag(p) - p p^T; `sqrt_MFR` builds the analytic (Householder) factor used to sample N(0, M_FR); `langevin_svpg` integrates one rep, `run_mc` runs a batch. The analytic sampler replaced an earlier `eigh`-based version, so the same seed now gives a different noise path (data from the old version are not reproducible with this file). |
| `run_slim_exp3.py` | SLiM runner, Ne = 1000 -> `data/slim3_main.npy` (panel E). |
| `run_slim_exp3_ne500.py` | SLiM runner, Ne = 500 -> `data/slim3_ne500.npy` (panel D). |
| `run_slim_exp3_ne3000.py` | SLiM runner, Ne = 3000 -> `data/slim3_ne3000.npy` (panel F). |
| `run_langevin_exp3.py` | Langevin batches for VPG Ne = 2000 and Ne = 6000 -> `data/langevin3_ne2000.npy`, `data/langevin3_ne6000.npy`. |
| `run_langevin_exp3_ne1000.py` | Langevin batch for VPG Ne = 1000 -> `data/langevin3_ne1000.npy`. |
| `slim/slim_L3_v12.slim` | Three-locus SLiM model shared by all SLiM runners; driven by `-d NE / R12 / R23 / MU / S1 / S2 / S3`. |
| `make_figs_exp3.py` | Summary figures (CJK fonts, Chinese labels) -> `figs/exp3_figC_hap.png`, `figs/exp3_figA_AF.png`, `figs/exp3_figB_LD.png`. |
| `make_figs_exp3_stochastic.py` | Stochastic summary figure -> `figs/exp3_figD_stochastic.png` (AF and LD mean +/- SE for the two original Ne pairs, 1000/2000 and 3000/6000; the Ne = 500/1000 pair is not included). |
| `make_paper_fig_exp3_arial.py` | **Paper figure for experiment 3 (integration panel)** (English labels, Arial) -> `fig_exp3.png` / `fig_exp3.pdf`. Layout 2x3: A = haplotype frequencies, B = allele frequencies, C = LD; D/E/F = AF for Ne 500/1000, 1000/2000, 3000/6000 (SLiM dotted, VPG solid, SE bands). Reads `data/` and `vpg_core_exp3.py` only. Runtime ~27 s. |
| `data/*.npy` | SLiM shape (100, 201, 9) = (reps, time, [generation, 8 haplotype frequencies]); Langevin shape (100, 201, 8). |

## How to run

```bash
# figures (need only numpy + matplotlib + the shipped data)
python3 make_paper_fig_exp3_arial.py   # paper figure (experiment 3) -> fig_exp3.png / fig_exp3.pdf
python3 make_figs_exp3.py              # figures -> figs/*.png
python3 make_figs_exp3_stochastic.py   # stochastic figure -> figs/exp3_figD_stochastic.png

# data regeneration (SLiM 4.2 required for the SLiM runners)
python3 run_slim_exp3.py 100           # Ne = 1000
python3 run_slim_exp3_ne500.py 100     # Ne = 500
python3 run_slim_exp3_ne3000.py 100    # Ne = 3000
python3 run_langevin_exp3.py 100       # VPG Ne = 2000 and 6000
python3 run_langevin_exp3_ne1000.py    # VPG Ne = 1000
```

* Every script resolves paths relative to its own directory (`_HERE`), so the
  package can be cloned anywhere and each script can be started from any
  working directory: `python3 exp3/make_figs_exp3.py`.
* SLiM runners need SLiM 4.2. The binary is taken from the `SLIM_BIN`
  environment variable, falling back to `/usr/local/bin/slim`:
  `SLIM_BIN=/path/to/slim python3 run_slim_exp3.py 100`.
* Runners skip a group whose `.npy` output already exists; delete the file
  first to recompute it.
* All runners default to **100** reps, matching the shipped data
  (`python3 run_slim_exp3...py [n_reps]`). Reference cost per rep on a 2-core
  machine: SLiM Ne = 1000 ~2.3 s (`multiprocessing.Pool(2)`), Langevin ~4.4 s
  per rep regardless of Ne.
* Python dependencies: `numpy`, `matplotlib` (see `../requirements.txt`).
* `make_paper_fig_exp3_arial.py` needs Arial installed under
  `/usr/share/fonts/truetype/msttcorefonts/` (`ttf-mscorefonts-installer`); it
  registers the fonts at runtime, sets `pdf.fonttype = 42` and uses no mathtext,
  so the PDF text stays editable in Illustrator.
* The `figs/*.png` figures are labelled in Chinese, so a CJK font must be
  installed (`WenQuanYi Zen Hei` or `Noto Sans CJK SC`); the `fig_exp*` figures
  are English/Arial and do not need a CJK font.
* Shared figure convention of `fig_exp1` / `fig_exp2` / `fig_exp3`: SVPG solid,
  DVPG dashed (its time axis is drawn at half speed, t/2), SLiM dotted with a
  mean +/- SE band; in `fig_exp3` panels D/E/F only SLiM (dotted) and VPG
  (solid) are shown.
