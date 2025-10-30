# qsf_validation.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity, DensityMatrix, entropy
from qiskit_aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error, depolarizing_error, amplitude_damping_error
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# --------------------------
# Helpers
# --------------------------
def prep_bell(qc: QuantumCircuit, q0: int, q1: int):
    qc.h(q0)
    qc.cx(q0, q1)

def U_obs_param(qc: QuantumCircuit, q_sys: Tuple[int,int], anc: int, theta: float):
    # implement controlled rotations from each system qubit to ancilla
    # using CRX(angle) or decomposed version - CRX is supported in Aer
    qc.crx(2*theta, q_sys[0], anc)
    qc.crx(2*theta, q_sys[1], anc)

def U_obs_dag_param(qc: QuantumCircuit, q_sys: Tuple[int,int], anc: int, theta: float):
    qc.crx(-2*theta, q_sys[1], anc)
    qc.crx(-2*theta, q_sys[0], anc)

def counts_to_probs(counts: Dict[str,int], shots:int):
    return {k: v/shots for k,v in counts.items()}

# compute mutual information I(A:B) from density matrix rho_AB
def mutual_information(rho_ab: DensityMatrix, dims=[2,2]) -> float:
    # rho_ab is DensityMatrix object; partial traces:
    rho_a = partial_trace(rho_ab, [1])
    rho_b = partial_trace(rho_ab, [0])
    Sa = entropy(rho_a, base=2)
    Sb = entropy(rho_b, base=2)
    Sab = entropy(rho_ab, base=2)
    return Sa + Sb - Sab

# --------------------------
# Run simulation for a given theta
# --------------------------
def run_simulation(theta: float, with_recovery: bool = True, ancilla_accessible: bool = True,
                   noise_model: NoiseModel = None) -> Dict:
    # qubit indices: 0,1 system ; 2 ancilla
    n = 3
    qr = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qr)
    # prepare target on q0,q1
    prep_bell(qc, 0, 1)
    # entangle with ancilla via U_obs
    U_obs_param(qc, (0,1), 2, theta)
    # optionally simulate ancilla lost (measure ancilla now -> inaccessible)
    if not ancilla_accessible:
        qc.measure_all()  # we'll just project everything; but we'll do statevector branching below
        # For simulator branch we will handle without measure; here we assume decoherence emulated differently
    # apply recovery
    if with_recovery and ancilla_accessible:
        U_obs_dag_param(qc, (0,1), 2, theta)
    # Note: measurement not added here; we use statevector/density
    # run via AerSimulator (statevector or density matrix)
    backend = AerSimulator(method='density_matrix') if noise_model else AerSimulator(method='statevector')
    if noise_model:
        sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    else:
        sim = AerSimulator(method='statevector')
    # transpile and run
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    psi = result.get_statevector(t_qc) if not noise_model else result.data(0)['density_matrix']
    # convert to density matrix object
    if not noise_model:
        rho_total = DensityMatrix(psi)
    else:
        rho_total = DensityMatrix(psi)  # already density matrix
    # partial trace ancilla to get system reduced state
    rho_sys = partial_trace(rho_total, [2])
    # ideal target state density
    sv_target = Statevector.from_label('00').tensor(Statevector.from_label('00')) # temporary
    # build ideal Bell for 2 qubits:
    bell = (Statevector.from_label('00') + Statevector.from_label('11')).normalize()
    rho_target = DensityMatrix(bell)
    # fidelity between rho_sys and bell target
    F = state_fidelity(rho_sys, rho_target)
    # mutual information between system (q0q1) and ancilla (q2) if density available
    # note: for mutual info we need partition (system:ancilla). system is dimension 4, ancilla dim 2
    try:
        MI = mutual_information(rho_total, dims=[4,2])
    except Exception:
        MI = None
    return {'theta': theta, 'fidelity': F, 'mutual_info': MI, 'rho_sys': rho_sys, 'rho_total': rho_total}

# --------------------------
# Noise model helper (example)
# --------------------------
def example_noise_model() -> NoiseModel:
    nm = NoiseModel()
    # add depolarizing on 2-qubit gates (CNOT) and amplitude damping on single qubit
    dep = depolarizing_error(0.02, 2)
    amp = amplitude_damping_error(0.01)
    nm.add_all_qubit_quantum_error(dep, ['cx'])
    nm.add_all_qubit_quantum_error(amp, ['rx','crx','x','u3','u2','u1'])
    return nm

# --------------------------
# λ-sweep and plotting
# --------------------------
def lambda_sweep_test(thetas=[0.05, 0.24, 0.43, 0.62, 0.81, 1.0, 1.5, 2.0], do_noise=False):
    results = []
    nm = example_noise_model() if do_noise else None
    for th in thetas:
        # Accessible ancilla + recovery
        r1 = run_simulation(th, with_recovery=True, ancilla_accessible=True, noise_model=nm)
        # No recovery (just apply U_obs and measure sys)
        r2 = run_simulation(th, with_recovery=False, ancilla_accessible=True, noise_model=nm)
        # Ancilla inaccessible: simulate by tracing out ancilla before attempting recovery
        # (We emulate by simulating and then removing ancilla info and then not applying R)
        r3 = run_simulation(th, with_recovery=False, ancilla_accessible=False, noise_model=nm)
        results.append({'theta': th, 'with_recovery': r1, 'without_recovery': r2, 'ancilla_lost': r3})
        print(f"theta={th:.3f} -> F_with={r1['fidelity']:.4f}, F_noR={r2['fidelity']:.4f}, F_lost={r3['fidelity']:.4f}")
    # plotting
    thetas = [x['theta'] for x in results]
    F_with = [x['with_recovery']['fidelity'] for x in results]
    F_no = [x['without_recovery']['fidelity'] for x in results]
    F_lost = [x['ancilla_lost']['fidelity'] for x in results]
    plt.figure(figsize=(7,4))
    plt.plot(thetas, F_with, '-o', label='With Recovery')
    plt.plot(thetas, F_no, '-s', label='No Recovery')
    plt.plot(thetas, F_lost, '-^', label='Ancilla Lost')
    plt.xlabel('Coupling parameter theta (proxy for λ)')
    plt.ylabel('Fidelity (system vs Bell)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0,1.02)
    plt.title('λ-sweep (simulation)')
    plt.show()
    return results

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print("Running QSF validation simulation (noise-free)...")
    res_clean = lambda_sweep_t test(thetas=[0.05,0.24,0.43,0.81,1.5], do_noise=False)  # small set
    print("Running QSF validation simulation (with noise model)...")
    res_noise = lambda_sweep_test(thetas=[0.05,0.24,0.43,0.81,1.5], do_noise=True)
