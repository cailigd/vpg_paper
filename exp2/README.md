# exp2 — Experiment 2: two-locus recombination–selection dynamics (L = 2)

Deterministic VPG (single-state SVPG and dual-state DVPG) compared with SLiM
Wright–Fisher simulations for two loci with no mutation. Selection acts on site
1 only (s2 = s), site 0 is neutral. Haplotype encoding h = b0 + 2*b1
(h = 0 '00', 1 '01', 2 '10', 3 '11'). Four parameter groups:
(r, s) in {0.001, 0.01} x {0.005, 0.05}, initial haplotype frequencies
(0.45, 0.05, 0.05, 0.45), SLiM diploid Ne = 1000 (counts 900/100/100/900),
T = 2000, one sample every 10 generations (201 points), 100 reps per group,
seed = 1000 + 7i.

## Files

| File | Purpose |
|---|---|
| `vpg_core_exp2.py` | Deterministic core (library, imported by the other scripts; no CLI). Site-independent projection `proj_indep`, recombination tensor `build_R_tensor` / `J_rec_tensor` / `J_rec_proj` / `J_rec`, selection flow `J_sel`, integrators `integrate_svpg` / `integrate_dvpg` (coupling flux J_c = gamma (rho - g)), and series helpers `AF` / `AF_series` / `D12` / `D_series`. |
| `run_slim_exp2.py` | SLiM runner for the four (r, s) groups using the nucleotide-based model. Writes `data/slim2_nuc_r<R>_s<S>.npy`, shape (100, 201, 5) = (reps, time, [generation, 4 haplotype frequencies]). |
| `slim/slim_L2_nuc.slim` | Current SLiM model. Fitness is assigned from the **nucleotide state** via `fitnessEffect()`, which removes the spurious selection (~2e-4) that appeared when fitness was keyed on mutation-object identity. Driven by `-d NE / R / S / H0..H3 / TMAX / OUTFILE`. |
| `slim/slim_L2_v1.slim` | Earlier (pre-fix) model, kept for reference only; no runner invokes it. |
| `make_figs_exp2.py` | Summary figures (CJK fonts, Chinese labels) -> `figs/exp2_figC_hap.png`, `figs/exp2_figA_AF.png`, `figs/exp2_figB_LD.png`. |
| `make_fig_exp2_arial.py` | **Paper figure for experiment 2 (integration panel)** (English labels, Arial) -> `fig_exp2.png` / `fig_exp2.pdf`. Layout: 3 rows (A = haplotype frequencies, B = D12, C = AF1/AF2) x 4 parameter columns. Reads `data/` and `vpg_core_exp2.py` only. Runtime ~35 s. |
| `data/*.npy` | SLiM results for the four groups (100 reps each). |

## How to run

```bash
python3 make_fig_exp2_arial.py        # paper figure (experiment 2) -> fig_exp2.png / fig_exp2.pdf
python3 make_figs_exp2.py             # figures -> figs/*.png
python3 run_slim_exp2.py             # regenerate SLiM data (100 reps per group)
```

* Every script resolves paths relative to its own directory (`_HERE`), so the
  package can be cloned anywhere and each script can be started from any
  working directory: `python3 exp2/make_fig_exp2_arial.py`.
* SLiM runners need SLiM 4.2. The binary is taken from the `SLIM_BIN`
  environment variable, falling back to `/usr/local/bin/slim`:
  `SLIM_BIN=/path/to/slim python3 run_slim_exp2.py`.
* Runners skip a group whose `.npy` output already exists; delete the file
  first to recompute it. `run_slim_exp2.py` defaults to 100 reps.
* `run_slim_exp2.py` writes `data/slim2_r<R>_s<S>.npy` (shape (100, 201, 5)),
  the same file names and seed set (1000 + 7i) as the shipped data, so a rerun
  overwrites/reproduces them.
* Python dependencies: `numpy`, `matplotlib` (see `../requirements.txt`).
* `make_fig_exp2_arial.py` needs Arial installed under
  `/usr/share/fonts/truetype/msttcorefonts/` (`ttf-mscorefonts-installer`); it
  registers the fonts at runtime, sets `pdf.fonttype = 42` and uses no mathtext,
  so the PDF text stays editable in Illustrator.
* The `figs/*.png` figures are labelled in Chinese, so a CJK font must be
  installed (`WenQuanYi Zen Hei` or `Noto Sans CJK SC`); the `fig_exp*` figures
  are English/Arial and do not need a CJK font.
* Shared figure convention of `fig_exp1` / `fig_exp2` / `fig_exp3`: SVPG solid,
  DVPG dashed (its time axis is drawn at half speed, t/2), SLiM dotted with a
  mean +/- SE band.
