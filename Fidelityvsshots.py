#!/usr/bin/env python3
"""
analyze_fidelity_vs_shots.py

Usage:
- Provide input JSON files describing runs. Each run entry is a dict with:
  {
    "run_id": "run1",
    "condition": "with" or "without",
    "quasi_dist": {0:0.49, 3:0.51}  # OR "counts": {"00":4000,...},  (bitstrings or int keys)
    "n_qubits": 2,
    "timestamp": "2025-10-28T12:00:00"
  }
- Then call main() or adapt for your pipeline.

Output:
- CSV "shots_summary.csv" with mean fidelity, std, bootstrap CI, paired-test p-values, recovery benefit.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats
import math
import csv
import os
import sys
from typing import Dict, List, Tuple, Any

# -------------------------
# Helpers
# -------------------------
def sample_counts_from_quasi(quasi: Dict[Any, float], shots: int, rng=None) -> Dict[str,int]:
    """
    Sample integer counts from a quasi-distribution (keys either bitstrings or ints).
    Returns dict of bitstring -> counts.
    """
    if rng is None:
        rng = np.random.default_rng()
    keys = list(quasi.keys())
    probs = np.array([quasi[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    samples = rng.choice(keys, size=shots, p=probs)
    counts = {}
    # convert int keys to bitstrings if necessary
    for k in samples:
        if isinstance(k, int):
            # need bit length: infer from max key
            max_key = max(int(x) for x in quasi.keys())
            bits = max(1, math.ceil(math.log2(max_key+1)))
            bs = format(int(k), f'0{bits}b')
        else:
            bs = str(k)
        counts[bs] = counts.get(bs, 0) + 1
    return counts

def counts_to_probs(counts: Dict[str,int]) -> Dict[str,float]:
    total = sum(counts.values())
    return {k: v/total for k,v in counts.items()}

def fidelity_from_counts(counts: Dict[str,int], ideal_probs: Dict[str,float]) -> float:
    """
    Classical fidelity (Bhattacharyya coefficient) between measured distribution and ideal distribution.
    ideal_probs should include all relevant bitstrings (missing keys assumed zero).
    """
    p = counts_to_probs(counts)
    keys = set(ideal_probs.keys()) | set(p.keys())
    bc = 0.0
    for k in keys:
        bc += math.sqrt(p.get(k,0.0) * ideal_probs.get(k,0.0))
    return float(bc)

def build_ideal_bell_probs(n_qubits:int) -> Dict[str,float]:
    """
    Return ideal distribution for GHZ/Bell: 50% all-zeros, 50% all-ones.
    """
    z = '0'*n_qubits
    o = '1'*n_qubits
    return {z:0.5, o:0.5}

# -------------------------
# Bootstrap & stats
# -------------------------
def bootstrap_mean(data: np.ndarray, nboot: int = 2000, seed: int = None) -> Tuple[float,float,float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    boots = rng.choice(data, size=(nboot, n), replace=True)
    means = boots.mean(axis=1)
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return float(np.mean(means)), float(lo), float(hi)

# paired t-test, return t and p
def paired_test(a: np.ndarray, b: np.ndarray) -> Tuple[float,float]:
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)

# -------------------------
# Main analysis function
# -------------------------
def analyze_runs_by_shots(runs: List[Dict],
                          shots_levels: List[int],
                          ideal_probs: Dict[str,float]=None,
                          n_repeats_sampling: int = 20,
                          nboot: int = 2000,
                          out_csv: str = "shots_summary.csv"):
    """
    runs: list of run dicts. Each run must have:
      - run_id, condition ('with' or 'without'), and either 'counts' or 'quasi_dist'
      - n_qubits optional (used to build default ideal)
    shots_levels: list of ints e.g. [1024, 4096, 8192]
    n_repeats_sampling: for runs providing quasi_dist, number of independent samplings per shots level
    """
    # Group runs by condition
    grouped = defaultdict(list)
    for r in runs:
        cond = r.get('condition','unknown')
        grouped[cond].append(r)

    # Determine ideal_probs if not provided (assume Bell if n_qubits present)
    if ideal_probs is None:
        # try infer n_qubits from first run
        nqubits = None
        for r in runs:
            if 'n_qubits' in r:
                nqubits = r['n_qubits']; break
            if 'quasi_dist' in r:
                # if keys are bitstrings, infer length
                keys = list(r['quasi_dist'].keys())
                if keys and isinstance(keys[0], str) and set(keys[0]) <= set('01'):
                    nqubits = len(keys[0]); break
        if nqubits is None:
            nqubits = 2
        ideal_probs = build_ideal_bell_probs(nqubits)

    summary_rows = []
    # For each shots level, compute fidelities per run (or sampling)
    for shots in shots_levels:
        # collect fidelities arrays for each condition
        fidelities = defaultdict(list)  # cond -> list of fidelity samples
        for cond, run_list in grouped.items():
            for r in run_list:
                # if run has direct counts, we can subsample if counts total >= shots
                if 'counts' in r:
                    counts = r['counts']
                    total = sum(counts.values())
                    if total >= shots:
                        # sample without replacement from counts
                        # expand counts to list (may be large) -> use probabilistic sampling
                        keys = list(counts.keys()); probs = np.array([counts[k]/total for k in keys])
                        # draw shots samples
                        samp = np.random.choice(keys, size=shots, p=probs)
                        counts_samp = Counter(samp)
                        F = fidelity_from_counts(counts_samp, ideal_probs)
                        fidelities[cond].append(F)
                    else:
                        # if counts fewer than shots, use scaling (fall back to proportional)
                        counts_scaled = {k: int(v * (shots/total)) for k,v in counts.items()}
                        F = fidelity_from_counts(counts_scaled, ideal_probs)
                        fidelities[cond].append(F)
                elif 'quasi_dist' in r:
                    quasi = r['quasi_dist']
                    # perform multiple independent samplings to estimate variability
                    for rep in range(n_repeats_sampling):
                        counts_samp = sample_counts_from_quasi(quasi, shots, rng=np.random.default_rng())
                        F = fidelity_from_counts(counts_samp, ideal_probs)
                        fidelities[cond].append(F)
                else:
                    raise ValueError("Run must contain 'counts' or 'quasi_dist'")

        # compute stats per condition
        cond_stats = {}
        for cond, vals in fidelities.items():
            arr = np.array(vals)
            mean, lo, hi = bootstrap_mean(arr, nboot=nboot)
            cond_stats[cond] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr, ddof=1)), 'n': len(arr),
                                'boot_mean': mean, 'boot_lo': lo, 'boot_hi': hi}
        # paired tests: try to pair by order if counts same number for with/without
        with_vals = np.array(fidelities.get('with', []))
        without_vals = np.array(fidelities.get('without', []))
        pair_t, pair_p = (None, None)
        if len(with_vals) and len(without_vals) and len(with_vals)==len(without_vals):
            pair_t, pair_p = paired_test(with_vals, without_vals)
        # compute delta mean
        delta_mean = None
        if 'with' in cond_stats and 'without' in cond_stats:
            delta_mean = cond_stats['with']['mean'] - cond_stats['without']['mean']
        # save summary row
        summary_rows.append({
            'shots': shots,
            'with_mean': cond_stats.get('with', {}).get('mean', None),
            'with_std': cond_stats.get('with', {}).get('std', None),
            'with_n': cond_stats.get('with', {}).get('n', 0),
            'with_boot_lo': cond_stats.get('with', {}).get('boot_lo', None),
            'with_boot_hi': cond_stats.get('with', {}).get('boot_hi', None),
            'without_mean': cond_stats.get('without', {}).get('mean', None),
            'without_std': cond_stats.get('without', {}).get('std', None),
            'without_n': cond_stats.get('without', {}).get('n', 0),
            'without_boot_lo': cond_stats.get('without', {}).get('boot_lo', None),
            'without_boot_hi': cond_stats.get('without', {}).get('boot_hi', None),
            'delta_mean': delta_mean,
            'paired_t': pair_t,
            'paired_p': pair_p
        })

    # write CSV
    keys = ['shots','with_mean','with_std','with_n','with_boot_lo','with_boot_hi',
            'without_mean','without_std','without_n','without_boot_lo','without_boot_hi',
            'delta_mean','paired_t','paired_p']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in keys})
    print(f"Summary written to {out_csv}")
    return pd.DataFrame(summary_rows)

# -------------------------
# Example loader for JSON file that contains an array of runs
# -------------------------
def load_runs_from_json(path: str) -> List[Dict]:
    with open(path,'r') as f:
        data = json.load(f)
    return data

# -------------------------
# Example usage in __main__
# -------------------------
if __name__ == "__main__":
    # Example: prepare a JSON file "runs.json" with two runs (with/without),
    # each providing 'quasi_dist' or 'counts' and n_qubits.
    if len(sys.argv) < 2:
        print("Usage: python analyze_fidelity_vs_shots.py runs.json")
        sys.exit(1)
    runs_json = sys.argv[1]
    if not os.path.exists(runs_json):
        print("File not found:", runs_json); sys.exit(1)
    runs = load_runs_from_json(runs_json)
    # define shots levels you tested
    shots_levels = [1024, 4096, 8192]
    df_summary = analyze_runs_by_shots(runs, shots_levels, n_repeats_sampling=30, nboot=2000, out_csv='shots_summary.csv')
    print(df_summary.to_string(index=False))
