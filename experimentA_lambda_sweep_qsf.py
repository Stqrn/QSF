#!/usr/bin/env python3
"""
experimentA_lambda_sweep_qsf.py

Experiment A:
λ-sweep under phase-damping noise with shallow, noise-aware Variational QSF.

Outputs:
 - lambda_sweep_summary.csv
 - fidelity_vs_lambda.png
 - prints bootstrap CI and paired-test summaries
"""

import math
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from collections import OrderedDict

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity
from qiskit_aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error, phase_damping_error, depolarizing_error
from scipy.optimize import minimize
from scipy import stats

# -----------------------
# Configuration
# -----------------------
N_QUBITS = 4                 # system qubits (use 4 to balance SNR and cost)
N_ANCILLA = 4                # ancilla count (one per system qubit)
TOTAL_Q = N_QUBITS + N_ANCILLA
SHOTS_EVAL = 2048            # for counts-like evaluation (optional)
USE_NOISY_SIM = True
VAR_LAYERS = 2               # shallow ansatz layers
VAR_MAXITER = 60             # optimization iters (COBYLA)
LAMBDA_VALUES = [0.05, 0.15, 0.30, 0.45, 0.60]  # sweep values (θ proxy for λ)
N_BOOT = 2000

CSV_OUT = "lambda_sweep_summary.csv"
PLOT_OUT = "fidelity_vs_lambda.png"

# -----------------------
# Noise model: phase-damping dominant
# -----------------------
def build_phase_damping_noise_model(phase_p=0.03, cx_dep_p=0.02) -> NoiseModel:
    nm = NoiseModel()
    # single-qubit phase damping on all single-qubit ops (approx)
    pd = phase_damping_error(phase_p)
    dep_cx = depolarizing_error(cx_dep_p, 2)
    # attach to common gate names
    nm.add_all_qubit_quantum_error(pd, ["u1", "u2", "u3", "rz", "rx", "ry", "x"])
    nm.add_all_qubit_quantum_error(dep_cx, ["cx", "cx"])
    return nm

# -----------------------
# Circuits: Bell-like / GHZ preparation on system
# -----------------------
def prepare_ghz_on_system(n_sys: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_sys)
    qc.h(0)
    for i in range(n_sys - 1):
        qc.cx(i, i+1)
    return qc

# -----------------------
# Observation unitary U_obs(λ)
# We build a practical entangling pattern mapping λ -> rotation angles
# -----------------------
def create_u_obs(n_sys: int, n_anc: int, lam: float) -> QuantumCircuit:
    total = n_sys + n_anc
    qc = QuantumCircuit(total)
    for i in range(n_sys):
        s = i
        a = n_sys + i
        # Controlled-RZ approximating e^{-i lam Zs Zo}
        qc.cp(lam * math.pi, s, a)
        # small ancilla rotation (weak measurement emulation)
        qc.ry(lam * math.pi / 6, a)
        # extra entangling cx
        qc.cx(s, a)
    return qc

# -----------------------
# Simple reversal U^\dagger
# -----------------------
def create_simple_reversal(n_sys: int, n_anc: int, lam: float) -> QuantumCircuit:
    return create_u_obs(n_sys, n_anc, lam).inverse()

# -----------------------
# Petz-inspired approximate recovery (practical)
# -----------------------
def create_petz_like(n_sys: int, n_anc: int, lam: float) -> QuantumCircuit:
    total = n_sys + n_anc
    qc = QuantumCircuit(total)
    for i in range(n_sys):
        s = i; a = n_sys + i
        qc.rz(-lam * math.pi / 2, a)
        qc.cx(s, a)
        qc.ry(-lam * math.pi / 4, a)
        qc.cp(-lam * math.pi * 0.9, s, a)
        qc.rx(lam * math.pi / 10, s)
    return qc

# -----------------------
# Shallow variational QSF template + binder
# - small hardware-efficient ansatz interleaving system-ancilla interactions
# -----------------------
def build_shallow_qsf_template(n_sys: int, n_anc: int, n_layers: int) -> Tuple[QuantumCircuit, int]:
    total = n_sys + n_anc
    qc = QuantumCircuit(total)
    # template structure: per layer apply RY/RZ on all qubits, then local entanglers s<->a
    for _ in range(n_layers):
        for q in range(total):
            qc.ry(0, q)
            qc.rz(0, q)
        for i in range(n_sys):
            qc.cx(i, n_sys + i)
    # param count estimate: 2 params per qubit per layer
    n_params = n_layers * total * 2
    return qc, n_params

def bind_params_to_template(template: QuantumCircuit, params: np.ndarray) -> QuantumCircuit:
    # rebuild structure substituting numeric angles sequentially
    total = template.num_qubits
    # approximate layers from param length
    # each layer uses 2*total params (RY,RZ) then none for cx
    params = params.flatten()
    ppp = 2 * total
    n_layers = max(1, len(params) // ppp)
    qc = QuantumCircuit(total)
    idx = 0
    for _ in range(n_layers):
        for q in range(total):
            ang1 = float(params[idx % len(params)]); idx += 1
            ang2 = float(params[idx % len(params)]); idx += 1
            qc.ry(ang1, q)
            qc.rz(ang2, q)
        for i in range(total//2):
            qc.cx(i, (total//2) + i)
    return qc

# -----------------------
# Utility: simulate full sequence and return system reduced state (DensityMatrix)
# -----------------------
def simulate_full_density(n_sys: int, n_anc: int,
                          prep_circ: QuantumCircuit,
                          u_obs: QuantumCircuit,
                          recovery: QuantumCircuit,
                          simulator: AerSimulator) -> DensityMatrix:
    total = n_sys + n_anc
    full = QuantumCircuit(total)
    # prepare on system
    full.compose(prep_circ, list(range(n_sys)), inplace=True)
    # apply u_obs on full registers
    full.compose(u_obs, list(range(total)), inplace=True)
    # apply recovery
    full.compose(recovery, list(range(total)), inplace=True)
    # simulate density
    circ_t = transpile(full, simulator)
    res = simulator.run(circ_t).result()
    # extract density_matrix or statevector
    if 'density_matrix' in res.data(0):
        dm = res.data(0)['density_matrix']
        rho = DensityMatrix(dm)
    else:
        sv = res.get_statevector()
        rho = DensityMatrix(sv)
    # partial trace ancilla (trace out ancilla)
    rho_sys = partial_trace(rho, list(range(n_sys, n_sys + n_anc)))
    return rho_sys

# -----------------------
# Cost / optimizer for shallow QSF (noise-aware)
# -----------------------
def train_variational_qsf(n_sys: int, n_anc: int, lam: float,
                          simulator: AerSimulator,
                          n_layers: int=VAR_LAYERS,
                          maxiter: int = VAR_MAXITER) -> Tuple[QuantumCircuit, np.ndarray, float]:
    print("Training shallow Variational QSF (noise-aware)...")
    template, n_params = build_shallow_qsf_template(n_sys, n_anc, n_layers)
    # pick param vector length proportional to n_params (use slightly larger to ensure coverage)
    param_len = max(n_params, 8)
    # target density matrix (GHZ/Bell on system)
    if n_sys == 1:
        raise ValueError("n_sys must be >=2")
    # build prep
    prep = prepare_ghz_on_system(n_sys)
    target_dm = DensityMatrix((Statevector.from_instruction(prep)).data)
    # observation unitary
    u_obs = create_u_obs(n_sys, n_anc, lam)

    def cost_fn(x):
        r_circ = bind_params_to_template(template, x)
        rho_sys = simulate_full_density(n_sys, n_anc, prep, u_obs, r_circ, simulator)
        F = state_fidelity(rho_sys, target_dm)
        # we minimize negative fidelity
        return -float(F)

    # initial random params
    x0 = np.random.uniform(0, 2*math.pi, size=param_len)
    res = minimize(cost_fn, x0, method='COBYLA', options={'maxiter': maxiter, 'tol':1e-3})
    x_opt = res.x
    r_opt = bind_params_to_template(template, x_opt)
    final_f = -res.fun
    print(f" Training done. achieved fidelity ≈ {final_f:.4f}")
    return r_opt, x_opt, float(final_f)

# -----------------------
# Bootstrap helper
# -----------------------
def bootstrap_mean_ci(data: List[float], nboot: int = N_BOOT, alpha=0.05) -> Tuple[float,float,float]:
    arr = np.array(data)
    rng = np.random.default_rng(0)
    boots = rng.choice(arr, size=(nboot, len(arr)), replace=True)
    means = boots.mean(axis=1)
    lo = np.percentile(means, 100*alpha/2)
    hi = np.percentile(means, 100*(1-alpha/2))
    return float(means.mean()), float(lo), float(hi)

# -----------------------
# Run λ-sweep: train QSF at each λ, evaluate strategies
# -----------------------
def run_lambda_sweep(lambdas: List[float], use_noisy_sim: bool = USE_NOISY_SIM,
                     shots_eval: int = SHOTS_EVAL) -> Dict[float, Dict[str, Any]]:
    results = OrderedDict()
    # prepare simulator
    if use_noisy_sim:
        noise = build_phase_damping_noise_model(phase_p=0.03, cx_dep_p=0.02)
        sim = AerSimulator(method='density_matrix', noise_model=noise)
        print("Using noisy simulator (phase-damping dominant).")
    else:
        sim = AerSimulator(method='statevector')
        print("Using ideal simulator.")
    # common prep
    prep = prepare_ghz_on_system(N_QUBITS)

    for lam in lambdas:
        print("\n" + "="*60)
        print(f"λ = {lam:.3f}  — building/evaluating strategies")
        u_obs = create_u_obs(N_QUBITS, N_ANCILLA, lam)

        # Strategy A: Simple U^dagger
        r_simple = create_simple_reversal(N_QUBITS, N_ANCILLA, lam)
        rho_simple = simulate_full_density(N_QUBITS, N_ANCILLA, prep, u_obs, r_simple, sim)
        F_simple = float(state_fidelity(rho_simple, DensityMatrix((Statevector.from_instruction(prep)).data)))

        # Strategy B: Petz-like
        r_petz = create_petz_like(N_QUBITS, N_ANCILLA, lam)
        rho_petz = simulate_full_density(N_QUBITS, N_ANCILLA, prep, u_obs, r_petz, sim)
        F_petz = float(state_fidelity(rho_petz, DensityMatrix((Statevector.from_instruction(prep)).data)))

        # Strategy C: Variational QSF (shallow) — train noise-aware
        r_var, params_var, f_train = train_variational_qsf(N_QUBITS, N_ANCILLA, lam, sim, n_layers=VAR_LAYERS, maxiter=VAR_MAXITER)
        rho_var = simulate_full_density(N_QUBITS, N_ANCILLA, prep, u_obs, r_var, sim)
        F_var = float(state_fidelity(rho_var, DensityMatrix((Statevector.from_instruction(prep)).data)))

        # store results
        results[lam] = {
            'F_simple': F_simple,
            'F_petz': F_petz,
            'F_var': F_var,
            'trained_estimate': f_train,
            'counts_eval': None
        }
        print(f"Results λ={lam:.3f}: Simple {F_simple:.4f}, Petz {F_petz:.4f}, VarQSF {F_var:.4f} (trained {f_train:.4f})")

    # optional: produce CSV + plot
    write_summary_csv(results, CSV_OUT)
    plot_results(results, PLOT_OUT)
    return results

# -----------------------
# CSV + Plot helpers
# -----------------------
def write_summary_csv(results: Dict[float, Dict[str, Any]], filename: str):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'F_simple', 'F_petz', 'F_var', 'trained_estimate'])
        for lam, data in results.items():
            writer.writerow([lam, data['F_simple'], data['F_petz'], data['F_var'], data['trained_estimate']])
    print(f"Summary CSV written to {filename}")

def plot_results(results: Dict[float, Dict[str, Any]], out_png: str):
    lambdas = list(results.keys())
    Fs = np.array([[results[l]['F_simple'], results[l]['F_petz'], results[l]['F_var']] for l in lambdas])
    plt.figure(figsize=(8,5))
    plt.plot(lambdas, Fs[:,0], '-o', label='Simple U†')
    plt.plot(lambdas, Fs[:,1], '-s', label='Petz-like')
    plt.plot(lambdas, Fs[:,2], '-^', label='Variational QSF (shallow)')
    plt.xlabel('λ (coupling proxy)')
    plt.ylabel('Recovered Fidelity (system vs target)')
    plt.title('λ-sweep under phase-damping noise')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_png}")
    plt.show()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    start = time.time()
    res = run_lambda_sweep(LAMBDA_VALUES, use_noisy_sim=USE_NOISY_SIM, shots_eval=SHOTS_EVAL)
    elapsed = time.time() - start
    print("\nExperiment A completed in %.1f s" % elapsed)
    # print summary table
    print("\nLambda sweep summary:")
    for lam, d in res.items():
        print(f"λ={lam:.3f}  Simple={d['F_simple']:.4f}  Petz={d['F_petz']:.4f}  VarQSF={d['F_var']:.4f}  (trained {d['trained_estimate']:.4f})")
