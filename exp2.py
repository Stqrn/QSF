"""
Baseline VQE (Variational Quantum Eigensolver) Experiment
==========================================================
Run VQE WITHOUT QSF for comparison with QSF-enhanced VQE
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimator
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ============================================
# Configuration
# ============================================
BACKEND_NAME = 'ibm_torino'
NUM_ITERATIONS = 50  # VQE optimization iterations

# ============================================
# 1. VQE Ansatz (Variational Form)
# ============================================

def create_vqe_ansatz(n_qubits: int, depth: int = 2) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    Create a hardware-efficient VQE ansatz.
    
    Args:
        n_qubits: Number of qubits
        depth: Number of ansatz layers
    
    Returns:
        Tuple of (circuit, parameters)
    """
    qc = QuantumCircuit(n_qubits)
    params = []
    
    # Initial layer of RY rotations
    for i in range(n_qubits):
        param = Parameter(f'θ_init_{i}')
        params.append(param)
        qc.ry(param, i)
    
    # Repeated layers
    for d in range(depth):
        # Entangling layer (circular CNOTs)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)  # Close the loop
        
        # Rotation layer
        for i in range(n_qubits):
            param = Parameter(f'θ_d{d}_q{i}')
            params.append(param)
            qc.ry(param, i)
    
    return qc, params


# ============================================
# 2. Hamiltonian Definition
# ============================================

def create_test_hamiltonian(n_qubits: int) -> SparsePauliOp:
    """
    Create a simple test Hamiltonian (e.g., Ising model).
    
    H = Σ Z_i Z_{i+1} + Σ X_i
    
    Args:
        n_qubits: Number of qubits
    
    Returns:
        SparsePauliOp representing the Hamiltonian
    """
    pauli_strings = []
    coeffs = []
    
    # ZZ interactions
    for i in range(n_qubits - 1):
        pauli_str = 'I' * i + 'ZZ' + 'I' * (n_qubits - i - 2)
        pauli_strings.append(pauli_str)
        coeffs.append(1.0)
    
    # X fields
    for i in range(n_qubits):
        pauli_str = 'I' * i + 'X' + 'I' * (n_qubits - i - 1)
        pauli_strings.append(pauli_str)
        coeffs.append(0.5)
    
    return SparsePauliOp(pauli_strings, coeffs)


# ============================================
# 3. VQE Optimization
# ============================================

class VQERunner:
    """VQE runner for baseline experiments."""
    
    def __init__(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp,
                 estimator: Estimator):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.estimator = estimator
        self.energies = []
        self.iteration = 0
        
    def cost_function(self, params: np.ndarray) -> float:
        """
        Cost function for VQE optimization.
        
        Args:
            params: Circuit parameters
        
        Returns:
            Energy expectation value
        """
        # Bind parameters to circuit
        bound_circuit = self.ansatz.assign_parameters(params)
        
        # Run estimator
        job = self.estimator.run(bound_circuit, self.hamiltonian)
        result = job.result()
        energy = result.values[0]
        
        # Store for tracking
        self.energies.append(energy)
        self.iteration += 1
        
        if self.iteration % 10 == 0:
            print(f"   Iteration {self.iteration}: Energy = {energy:.4f}")
        
        return energy
    
    def run(self, initial_params: np.ndarray = None) -> Dict:
        """
        Run VQE optimization.
        
        Args:
            initial_params: Initial parameter values
        
        Returns:
            Dictionary with optimization results
        """
        if initial_params is None:
            initial_params = np.random.rand(len(self.ansatz.parameters)) * 2 * np.pi
        
        print(f"🔬 Starting VQE optimization...")
        print(f"   Parameters: {len(initial_params)}")
        print(f"   Initial energy: {self.cost_function(initial_params):.4f}")
        
        # Optimize
        result = minimize(
            self.cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': NUM_ITERATIONS, 'disp': False}
        )
        
        return {
            'optimal_energy': result.fun,
            'optimal_params': result.x,
            'success': result.success,
            'iterations': self.iteration,
            'energy_history': self.energies
        }


# ============================================
# 4. Run Baseline VQE Experiments
# ============================================

def run_baseline_vqe(backend_name: str = BACKEND_NAME) -> Dict:
    """
    Run baseline VQE experiments.
    
    Args:
        backend_name: IBM backend name
    
    Returns:
        Results dictionary
    """
    print(f"🚀 Starting Baseline VQE Experiment on {backend_name}")
    print("=" * 50)
    
    # Connect to backend
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"✅ Connected to {backend_name}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}
    
    results = {}
    
    # Test different qubit counts
    for n_qubits in [2, 3, 4]:
        print(f"\n{'='*50}")
        print(f"🔬 Running {n_qubits}-qubit VQE (Baseline)...")
        
        try:
            # Create ansatz and Hamiltonian
            ansatz, params = create_vqe_ansatz(n_qubits, depth=2)
            hamiltonian = create_test_hamiltonian(n_qubits)
            
            print(f"   Ansatz depth: {ansatz.depth()}")
            print(f"   Parameters: {len(params)}")
            print(f"   Hamiltonian terms: {len(hamiltonian)}")
            
            # Create estimator
            with Session(backend=backend) as session:
                estimator = Estimator(session=session)
                
                # Run VQE
                vqe = VQERunner(ansatz, hamiltonian, estimator)
                result = vqe.run()
                
                # Calculate theoretical ground state (for comparison)
                # For small systems, can diagonalize exactly
                if n_qubits <= 4:
                    import scipy.sparse.linalg as spla
                    H_matrix = hamiltonian.to_matrix()
                    eigenvalues, _ = spla.eigsh(H_matrix, k=1, which='SA')
                    theoretical_ground = eigenvalues[0]
                else:
                    theoretical_ground = None
                
                # Calculate approximation ratio
                if theoretical_ground is not None:
                    approx_ratio = result['optimal_energy'] / theoretical_ground
                else:
                    approx_ratio = None
                
                # Store results
                results[n_qubits] = {
                    'optimal_energy': result['optimal_energy'],
                    'optimal_params': result['optimal_params'],
                    'energy_history': result['energy_history'],
                    'iterations': result['iterations'],
                    'success': result['success'],
                    'theoretical_ground': theoretical_ground,
                    'approximation_ratio': approx_ratio,
                    'ansatz_depth': ansatz.depth()
                }
                
                print(f"   ✅ Optimal Energy: {result['optimal_energy']:.4f}")
                if theoretical_ground:
                    print(f"   Theoretical Ground: {theoretical_ground:.4f}")
                    print(f"   Approximation Ratio: {approx_ratio:.4f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[n_qubits] = {'error': str(e)}
    
    print(f"\n{'='*50}")
    print("✅ Baseline VQE experiments completed!")
    
    return results


# ============================================
# 5. Analysis and Visualization
# ============================================

def analyze_vqe_results(results: Dict) -> None:
    """
    Analyze and visualize VQE results.
    
    Args:
        results: Results from run_baseline_vqe
    """
    print("\n📊 VQE BASELINE RESULTS SUMMARY")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Final energies
    qubits = []
    energies = []
    theoretical = []
    
    for n, data in sorted(results.items()):
        if 'error' not in data:
            qubits.append(n)
            energies.append(data['optimal_energy'])
            if data['theoretical_ground']:
                theoretical.append(data['theoretical_ground'])
            
            print(f"\n{n} Qubits:")
            print(f"  Optimal Energy: {data['optimal_energy']:.4f}")
            if data['theoretical_ground']:
                print(f"  Theoretical: {data['theoretical_ground']:.4f}")
                print(f"  Approximation Ratio: {data['approximation_ratio']:.4f}")
    
    ax1 = axes[0]
    x_pos = np.arange(len(qubits))
    width = 0.35
    
    ax1.bar(x_pos - width/2, energies, width, label='VQE (Baseline)', color='blue', alpha=0.7)
    if theoretical:
        ax1.bar(x_pos + width/2, theoretical, width, label='Theoretical', color='green', alpha=0.7)
    
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Ground State Energy', fontsize=12)
    ax1.set_title('VQE Baseline: Energy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(qubits)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Convergence histories
    ax2 = axes[1]
    colors = ['blue', 'orange', 'green']
    
    for i, (n, data) in enumerate(sorted(results.items())):
        if 'error' not in data and 'energy_history' in data:
            ax2.plot(data['energy_history'], label=f'{n} qubits', 
                    color=colors[i % len(colors)], linewidth=2)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('VQE Baseline: Convergence', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_vqe_results.png', dpi=300, bbox_inches='tight')
    print("\n📈 Plots saved as 'baseline_vqe_results.png'")
    plt.show()


# ============================================
# 6. Main Execution
# ============================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   BASELINE VQE EXPERIMENTS                   ║
    ║   (Without QSF Framework)                    ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Run experiments
    results = run_baseline_vqe()
    
    # Analyze
    if results:
        analyze_vqe_results(results)
        
        print("\n" + "="*50)
        print("✅ VQE BASELINE EXPERIMENTS COMPLETED!")
        print("="*50)
