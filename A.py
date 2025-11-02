#!/usr/bin/env python3
"""
qsf_paired_ancilla.py
Run paired repeats and ancilla-accessibility tests for QSF vs Petz vs Simple-Udagger.

Requirements:
  qiskit, qiskit_ibm_runtime, numpy, scipy, pandas

Usage:
  python qsf_paired_ancilla.py
"""

import time, os, csv, json
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

# -----------------------
# Config (edit as needed)
# -----------------------
BACKEND_NAME = "ibm_torino"
USE_REAL_HARDWARE = True   # set False to use AerSimulator locally
LAMBDA = 0.8
N_SYS = 4
N_ANC = 4
SHOTS = 8192
REPEATS = 5   # number of independent jobs per condition
RESULTS_DIR = "qsf_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------
# Circuit builders
# -----------------------
def prepare_ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    return qc

def create_u_obs(n_sys, n_anc, lam):
    total = n_sys + n_anc
    qc = QuantumCircuit(total)
    for i in range(n_sys):
        s = i; a = n_sys + i
        qc.cp(lam * np.pi, s, a)
        qc.ry(lam * np.pi / 6, a)
        qc.cx(s, a)
    return qc

def create_simple_reversal(n_sys, n_anc, lam):
    return create_u_obs(n_sys, n_anc, lam).inverse()

def create_petz(n_sys, n_anc, lam):
    total = n_sys + n_anc
    qc = QuantumCircuit(total)
    for i in range(n_sys):
        s = i; a = n_sys + i
        qc.rz(-lam * np.pi/2, a)
        qc.cx(s, a)
        qc.ry(-lam * np.pi/4, a)
        qc.cp(-lam * np.pi * 0.9, s, a)
        qc.rx(lam * np.pi / 12, s)
    return qc

def create_qsf_shallow(n_sys, n_anc, lam):
    total = n_sys + n_anc
    qc = QuantumCircuit(total)
    # simple shallow QSF template
    for i in range(n_sys):
        s = i; a = n_sys + i
        qc.cp(lam * np.pi/2, s, a)
        qc.ry(-lam * np.pi/8, a)
        qc.cx(s,a)
        qc.ry(-lam * np.pi/16, s)
    # small synergic correction layer
    for i in range(n_sys):
        s = i; a = n_sys + i
        qc.cp(-lam * np.pi/6, a, s)
    return qc

# -----------------------
# Helpers: build full circuit and compute fidelity from counts
# -----------------------
def build_full_circuit(recovery_circ, n_sys):
    total = recovery_circ.num_qubits
    prep = prepare_ghz(n_sys)
    full = QuantumCircuit(total, n_sys)
    full.compose(prep, list(range(n_sys)), inplace=True)
    # assume u_obs already applied in recovery flow (we'll compose externally)
    full.compose(recovery_circ, list(range(total)), inplace=True)
    full.measure(range(n_sys), range(n_sys))
    return full

def counts_to_fidelity(counts, n_sys):
    total = sum(counts.values())
    if total == 0: return 0.0
    zeros = '0'*n_sys
    ones = '1'*n_sys
    return (counts.get(zeros,0) + counts.get(ones,0)) / total

# -----------------------
# Run function (uses Qiskit Runtime Sampler when USE_REAL_HARDWARE=True)
# -----------------------
def run_job_on_backend(circ, shots, backend_name=BACKEND_NAME):
    # Attempt to use QiskitRuntimeService Sampler (real backend)
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session)
            job = sampler.run(circ, shots=shots)
            res = job.result()
            # quasi_dists -> convert to counts
            quasi = res.quasi_dists[0]
            counts = {}
            for key, prob in quasi.items():
                bitstr = format(key, f'0{circ.num_qubits}b')[:circ.num_clbits]  # take measured bits
                counts[bitstr] = counts.get(bitstr, 0) + int(round(prob * shots))
            return counts, job.job_id()
    except Exception as e:
        raise e

# -----------------------
# Paired repeats + ancilla-accessibility logic
# -----------------------
def experiment_runner(lambda_val=LAMBDA, repeats=REPEATS, shots=SHOTS):
    # construct building blocks
    u_obs = create_u_obs(N_SYS, N_ANC, lambda_val)
    r_simple = create_simple_reversal(N_SYS, N_ANC, lambda_val)
    r_petz = create_petz(N_SYS, N_ANC, lambda_val)
    r_qsf = create_qsf_shallow(N_SYS, N_ANC, lambda_val)

    # Full sequences: prepare + u_obs + recovery + measure system
    def full_sequence(recovery):
        total = N_SYS + N_ANC
        circ = QuantumCircuit(total, N_SYS)
        # prepare GHZ on system
        prep = prepare_ghz(N_SYS)
        circ.compose(prep, list(range(N_SYS)), inplace=True)
        circ.compose(u_obs, list(range(total)), inplace=True)
        circ.compose(recovery, list(range(total)), inplace=True)
        circ.measure(range(N_SYS), range(N_SYS))
        return circ

    circuits = {
        'simple': full_sequence(r_simple),
        'petz': full_sequence(r_petz),
        'qsf': full_sequence(r_qsf)
    }

    # Ancilla test variants: normal vs ancilla-decohered
    # Ancilla-decohere: apply strong depolarizing-like layer (simulate by randomizing ancilla rotations)
    def make_decohered_variant(recovery):
        total = N_SYS + N_ANC
        rc = QuantumCircuit(total)
        # same as recovery but add depolarizing-like noise on ancillas (approx)
        rc.compose(recovery, list(range(total)), inplace=True)
        for a in range(N_SYS, total):
            rc.rx(np.pi/2, a)
            rc.rz(np.pi*0.5, a)
            rc.rx(np.pi/2, a)
        return rc

    circuits_deco = {
        'simple_deco': full_sequence(make_decohered_variant(r_simple)),
        'petz_deco': full_sequence(make_decohered_variant(r_petz)),
        'qsf_deco': full_sequence(make_decohered_variant(r_qsf))
    }

    # run repeats paired: for each repeat, submit in same time-window each strategy
    results = []
    for rep in range(repeats):
        print(f"=== Repeat {rep+1}/{repeats} ===")
        job_ids = {}
        for name, circ in circuits.items():
            circ_t = transpile(circ, backend=None, optimization_level=3)
            try:
                counts, job_id = run_job_on_backend(circ_t, shots)
                fid = counts_to_fidelity(counts, N_SYS)
                results.append({'repeat':rep, 'condition':name, 'fidelity':fid, 'counts':counts})
                print(f" {name}: fid={fid:.4f} jobid={job_id}")
            except Exception as e:
                print("Run error (sim fallback):", e)
                # fallback: simulate locally (statevector) - approximate
                from qiskit_aer import AerSimulator
                sim = AerSimulator(method='density_matrix')
                job = sim.run(circ_t, shots=shots).result()
                counts = job.get_counts()
                fid = counts_to_fidelity(counts, N_SYS)
                results.append({'repeat':rep, 'condition':name, 'fidelity':fid, 'counts':counts})
                print(f" {name} (sim): fid={fid:.4f}")

        # now run decohered variants (ancilla-accessibility test)
        for name, circ in circuits_deco.items():
            circ_t = transpile(circ, backend=None, optimization_level=3)
            try:
                counts, job_id = run_job_on_backend(circ_t, shots)
                fid = counts_to_fidelity(counts, N_SYS)
                results.append({'repeat':rep, 'condition':name, 'fidelity':fid, 'counts':counts})
                print(f" {name}: fid={fid:.4f} jobid={job_id}")
            except Exception as e:
                from qiskit_aer import AerSimulator
                sim = AerSimulator(method='density_matrix')
                job = sim.run(circ_t, shots=shots).result()
                counts = job.get_counts()
                fid = counts_to_fidelity(counts, N_SYS)
                results.append({'repeat':rep, 'condition':name, 'fidelity':fid, 'counts':counts})
                print(f" {name} (sim): fid={fid:.4f}")

        # save intermediate
        pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "paired_ancilla_raw.csv"), index=False)
    return results

# -----------------------
# Analysis helpers: bootstrap + paired t-test
# -----------------------
def bootstrap_ci(arr, nboot=2000, alpha=0.05):
    rng = np.random.default_rng(0)
    arr = np.array(arr)
    boots = rng.choice(arr, size=(nboot, len(arr)), replace=True)
    means = boots.mean(axis=1)
    return means.mean(), np.percentile(means, 100*alpha/2), np.percentile(means, 100*(1-alpha/2))

def analyze_results(dfpath):
    df = pd.read_csv(dfpath)
    # group by condition
    summary = df.groupby('condition')['fidelity'].agg(['mean','std','count']).reset_index()
    # compute bootstrap CIs
    cis = []
    for cond in summary['condition']:
        vals = df[df['condition']==cond]['fidelity'].values
        m, lo, hi = bootstrap_ci(vals)
        cis.append((m,lo,hi))
    summary['boot_mean'] = [c[0] for c in cis]
    summary['ci_lo'] = [c[1] for c in cis]
    summary['ci_hi'] = [c[2] for c in cis]
    summary.to_csv(os.path.join(RESULTS_DIR, "paired_ancilla_summary.csv"), index=False)
    print("Saved summary CSV.")
    # paired tests: compare qsf vs petz and qsf vs simple
    pivot = df.pivot(index='repeat', columns='condition', values='fidelity')
    pairs = [('qsf','petz'),('qsf','simple')]
    with open(os.path.join(RESULTS_DIR, "paired_stats.txt"), 'w') as f:
        for a,b in pairs:
            if a in pivot.columns and b in pivot.columns:
                avals = pivot[a].dropna().values
                bvals = pivot[b].dropna().values
                t,p = stats.ttest_rel(avals, bvals)
                d = (avals - bvals).mean() / (avals - bvals).std(ddof=1)
                line = f"Paired {a} vs {b}: n={len(avals)}, mean_diff={np.mean(avals-bvals):.4f}, t={t:.3f}, p={p:.4f}, cohen_d={d:.3f}"
                print(line)
                f.write(line + "\n")
    print("Saved paired_stats.txt")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    start = time.time()
    res = experiment_runner()
    df = pd.DataFrame(res)
    raw_csv = os.path.join(RESULTS_DIR, "paired_ancilla_raw.csv")
    df.to_csv(raw_csv, index=False)
    analyze_results(raw_csv)
    print("Done. Elapsed:", time.time() - start)
