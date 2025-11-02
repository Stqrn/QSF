from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import state_fidelity, DensityMatrix
import numpy as np

def amplitude_damping_channel(circuit, qubits, gamma):
    """Implements amplitude damping via Kraus ops using ancilla."""
    for q in qubits:
        circuit.ry(2*np.arcsin(np.sqrt(gamma)), q)
    return circuit

def qsf_recovery_conditional(circuit, ancilla, target, clbit):
    """Conditional recovery controlled by measured ancilla."""
    # Example: if ancilla == 1, apply X to target
    circuit.x(target).c_if(clbit, 1)
    return circuit

def qsf_recovery_unconditional(circuit, target):
    """Unconditional (no ancilla info) recovery."""
    circuit.x(target)
    return circuit

def build_midcircuit_experiment(measure=True, gamma=0.15):
    qc = QuantumCircuit(2, 1)
    # Step 1: create entanglement
    qc.h(0)
    qc.cx(0, 1)

    # Step 2: noise (simulate system–env coupling)
    amplitude_damping_channel(qc, [1], gamma)

    if measure:
        # Step 3: measure ancilla mid-circuit
        qc.measure(0, 0)
        # Step 4: conditional recovery based on measurement
        qsf_recovery_conditional(qc, 0, 1, 0)
    else:
        # destroy ancilla info (simulate decoherence)
        qc.reset(0)
        # unconditional recovery (no info)
        qsf_recovery_unconditional(qc, 1)

    # Step 5: measure final qubit to check fidelity
    qc.save_statevector()
    return qc

# Run both experiments
shots = 8192
backend = Aer.get_backend('aer_simulator')

circuits = {
    "conditional": build_midcircuit_experiment(measure=True),
    "no_info": build_midcircuit_experiment(measure=False)
}

results = {}
for label, qc in circuits.items():
    transpiled = transpile(qc, backend)
    job = execute(transpiled, backend, shots=shots)
    result = job.result()
    state = result.data(0)['statevector']
    results[label] = DensityMatrix(state)

# Compute fidelity with ideal Bell state
ideal = DensityMatrix.from_instruction(QuantumCircuit(2).h(0).cx(0, 1))
f_cond = state_fidelity(results["conditional"], ideal)
f_no = state_fidelity(results["no_info"], ideal)

print(f"Conditional recovery fidelity: {f_cond:.4f}")
print(f"No-info recovery fidelity:      {f_no:.4f}")
