#!/usr/bin/env python3
"""
process_fidelity_1q.py
Compute average process fidelity for QSF vs baselines on single-qubit subsystem.

Procedure:
  - Prepare basis states {|0>, |1>, |+>, |+i>}
  - Apply U_obs on (sys+ancilla), apply recovery, trace out ancilla
  - Compare output state with ideal (no observe) via state_fidelity
"""
import os, numpy as np, pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, partial_trace
from qiskit_aer import AerSimulator

# Config
N_SYS = 1
N_ANC = 1
LAMBDA = 0.8
SHOTS = 8192
RESULTS_DIR = "qsf_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# builders (simple versions)
def prep_state(name):
    qc = QuantumCircuit(N_SYS)
    if name == "0":
        pass
    elif name == "1":
        qc.x(0)
    elif name == "+":
        qc.h(0)
    elif name == "+i":
        qc.sdg(0); qc.h(0)
    return qc

def u_obs(n_sys, n_anc, lam):
    qc = QuantumCircuit(n_sys + n_anc)
    qc.cp(lam * np.pi, 0, 1)
    qc.ry(lam * np.pi / 6, 1)
    qc.cx(0,1)
    return qc

def recovery_qsf(n_sys, n_anc, lam):
    qc = QuantumCircuit(n_sys + n_anc)
    qc.cp(-lam * np.pi/6, 1, 0)
    qc.ry(-lam * np.pi/8, 1)
    qc.cx(0,1)
    return qc

def baseline_udag(n_sys, n_anc, lam):
    return u_obs(n_sys,n_anc,lam).inverse()

def run_process_fidelity():
    simulator = AerSimulator(method="density_matrix")
    states = ["0","1","+","+i"]
    target_contribs = []
    for s in states:
        prep = prep_state(s)
        target_sv = Statevector(prep).data
        target_dm = DensityMatrix(target_sv)
        # build full circuit: prep (on sys) + u_obs + recovery -> get reduced sys state
        u = u_obs(N_SYS, N_ANC, LAMBDA)
        r_qsf = recovery_qsf(N_SYS, N_ANC, LAMBDA)
        full = QuantumCircuit(N_SYS + N_ANC)
        full.compose(prep, [0], inplace=True)
        full.compose(u, [0,1], inplace=True)
        full.compose(r_qsf, [0,1], inplace=True)
        circ_t = transpile(full, simulator)
        res = simulator.run(circ_t).result()
        dm = DensityMatrix(res.data(0)['density_matrix'])
        rho_sys = partial_trace(dm, [1])
        f = state_fidelity(rho_sys, target_dm)
        target_contribs.append(f)
    avg_f_qsf = float(np.mean(target_contribs))

    # baseline Udag
    target_contribs_b = []
    for s in states:
        prep = prep_state(s)
        target_dm = DensityMatrix(Statevector(prep).data)
        full = QuantumCircuit(N_SYS + N_ANC)
        full.compose(prep,[0], inplace=True)
        full.compose(u_obs(N_SYS,N_ANC,LAMBDA), [0,1], inplace=True)
        full.compose(baseline_udag(N_SYS,N_ANC,LAMBDA), [0,1], inplace=True)
        res = simulator.run(transpile(full, simulator)).result()
        dm = DensityMatrix(res.data(0)['density_matrix'])
        rho_sys = partial_trace(dm, [1])
        target_contribs_b.append(float(state_fidelity(rho_sys, target_dm)))
    avg_f_udag = float(np.mean(target_contribs_b))

    print("Process avg fidelity (QSF) =", avg_f_qsf)
    print("Process avg fidelity (Udag) =", avg_f_udag)
    pd.DataFrame([{"method":"qsf","proc_f":avg_f_qsf},{"method":"udag","proc_f":avg_f_udag}]).to_csv(os.path.join(RESULTS_DIR,"process_fidelity_1q.csv"), index=False)

if __name__ == "__main__":
    run_process_fidelity()
