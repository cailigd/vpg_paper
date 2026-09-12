#!/usr/bin/env python3
"""Experiment 2 official rerun with the nucleotide version: Ne=1000, 4 groups × 100 reps
r∈{0.001,0.01} × s∈{0.005,0.05}; initial (0.45,0.05,0.05,0.45) → H=900/100/100/900
Seeds 1000+i*7 (consistent with the old data, for easy comparison)
Output: data/slim2_r{R}_s{S}.npy"""
import subprocess, numpy as np, os, multiprocessing as mp, time

SLIM = os.environ.get('SLIM_BIN', '/usr/local/bin/slim')  # override: SLIM_BIN=/path/to/slim
_HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(_HERE, 'slim', 'slim_L2_nuc.slim')
DATA = os.path.join(_HERE, 'data')

def run_one(args):
    seed, r, s, outfile = args
    cmd = [SLIM, '-s', str(seed), '-d', 'NE=1000', '-d', f'R={r}', '-d', f'S={2*s}',
           '-d', 'H0=900', '-d', 'H1=100', '-d', 'H2=100', '-d', 'H3=900',
           '-d', 'TMAX=2000', '-d', f"OUTFILE='{outfile}'", SCRIPT]
    rr = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if not os.path.exists(outfile):
        print(f"[rep error] seed={seed} r={r} s={s}: {rr.stderr[-300:]}")
        return np.zeros((201, 5))
    return np.loadtxt(outfile)

def run_group(r, s, n_reps=100):
    key = f'r{str(r).replace(".","p")}_s{str(s).replace(".","p")}'
    fn = f'{DATA}/slim2_{key}.npy'
    if os.path.exists(fn):
        print(f"[skip] {key}"); return
    t0 = time.time()
    tmpdir = f'{DATA}/tmp_nuc_{key}'
    os.makedirs(tmpdir, exist_ok=True)
    tasks = [(1000 + i * 7, r, s, f'{tmpdir}/rep_{i}.txt') for i in range(n_reps)]
    with mp.Pool(2) as pool:
        res = pool.map(run_one, tasks)
    arr = np.stack(res)
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
    np.save(fn, arr)
    gen = arr[0, :, 0]
    af1 = arr[:, :, 2] + arr[:, :, 4]
    af2 = arr[:, :, 3] + arr[:, :, 4]
    q0 = np.zeros(n_reps)
    for i in range(n_reps):
        idx = np.argmax(af2[i] >= 0.99999)
        q0[i] = af1[i, idx] if af2[i, idx] >= 0.99999 else af1[i, -1]
    qe = af1[:, -1]
    d = qe - q0
    print(f"[nuc {key}] n={n_reps}: AF1_end={qe.mean():.4f}±{qe.std()/np.sqrt(n_reps):.4f} "
          f"fix={(qe>0.999).sum()} loss={(qe<0.001).sum()} | E[q0]={q0.mean():.4f} "
          f"E[q_end-q0]={d.mean():+.4f} z={d.mean()/(d.std()/np.sqrt(n_reps)):+.1f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    for r in [0.001, 0.01]:
        for s in [0.005, 0.05]:
            run_group(r, s)
    print("EXP2 NUC DONE")
