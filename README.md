This repository contains the code for the numerical experiments conducted in the
paper *"A Variational Modeling Framework for Population Genetic Dynamics"*.

Package version: **v3.3** (2026-09-12).

## Experiments

| Directory | Model | Parameters | Simulation setup |
|---|---|---|---|
| `exp1` | L = 1, single locus: mutation–selection balance | u = v = mu, mu in {3e-4, 1e-3, 3e-3} × s in {0.005, 0.01, 0.05} at p0 = 0.01 (9 sets), plus two bidirectional-convergence sets at p0 = 0.99 | 100 reps, SLiM diploid Ne = 1000 (2Ne = 2000 haplotypes), T = 2000, sampled every 10 generations |
| `exp2` | L = 2, two loci: recombination–selection, no mutation | (r, s) in {0.001, 0.01} × {0.005, 0.05}; selection acts on site 1 only; initial haplotype frequencies (0.45, 0.05, 0.05, 0.45) | 100 reps, SLiM Ne = 1000, T = 2000, sampled every 10 generations |
| `exp3` | L = 3, three loci: mutation + recombination + selection, plus drift (Langevin extension) | u = v = 0.001, s = [0.005, 0, −0.005], r12 = 0.01, r23 = 0.05 | 100 reps per Ne pair, T = 2000, sampled every 10 generations |

Finite-population scale in `exp3` is paired as SLiM diploid Ne ↔ VPG haploid
Ne = 2 × Ne_slim: 500 ↔ 1000, 1000 ↔ 2000, 3000 ↔ 6000.

## Repository layout

Each experiment directory contains:

| File | Purpose |
|---|---|
| `vpg_core_exp*.py` | Deterministic VPG core (library: right-hand sides, integrators, observable helpers) |
| `langevin_exp3.py` | Stochastic Langevin core (`exp3` only): dp = (J_mut + J_rec + J_sel) dt + sqrt(M_FR(p)/Ne) dW |
| `run_*.py` | SLiM / Langevin batch runners; they produce the `data/*.npy` files |
| `make_figs_exp*.py` | Summary figures written to `figs/` |
| `make_fig*_arial.py` | Paper integration panels → `fig_exp{1,2,3}.png` / `.pdf` (Arial, text editable in Illustrator) |
| `slim/` | SLiM 4.2 model scripts |
| `data/` | Simulation output shipped with the repository (`.npy`) |
| `README.md` | Per-experiment details: file purposes and how to run |

Root files: `requirements.txt` (numpy, matplotlib), `.gitignore`.

## Running

```bash
pip install -r requirements.txt        # numpy, matplotlib
cd exp1
python3 make_fig_exp1_arial.py         # paper integration panel -> fig_exp1.png / .pdf
python3 make_figs_exp1.py              # summary figures -> figs/
python3 run_slim_exp1.py 100           # regenerate the SLiM data (needs SLiM 4.2)
```

* All scripts resolve their paths relative to their own directory, so they run
  from any working directory and straight after `git clone`.
* SLiM runners take the binary from the `SLIM_BIN` environment variable,
  falling back to `/usr/local/bin/slim`.
* Runners skip a group whose `.npy` output already exists — delete the file to
  recompute it.
* Reproducibility: every shipped data file is reproduced exactly by the runners
  in this repository (100 reps; SLiM seeds 1000 + 7i, Langevin seeds 1000 + i).
* Paper figures need Arial (`ttf-mscorefonts-installer`); the summary figures in
  `figs/` are labelled in Chinese and need a CJK font (e.g. Noto Sans CJK SC).
* `figs/*.png` are the summary figures of each experiment; `fig_exp*.png|pdf`
  are the integration panels used in the paper.
