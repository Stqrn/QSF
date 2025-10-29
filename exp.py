"""
Baseline Bell State Experiment (Without QSF)
=============================================
This code runs Bell state generation WITHOUT QSF framework
for direct comparison with QSF results.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit.quantum_info import state_fidelity, Statevector
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ============================================
# Configuration
# ============================================
BACKEND_NAME = 'ibm_torino'  # or your available backend
SHOTS = 8192  # Maximum shots for better statistics

# ============================================
# 1. Bell State Circuits (Baseline - No QSF)
# ============================================

def create_bell_state_baseline(n_qubits: int = 2) -> QuantumCircuit:
    """
    Create a simple Bell state without QSF protection.
    
    Args:
        n_qubits: Number of qubits (2, 3, or 4)
    
    Returns:
        QuantumCircuit: Bell state circuit
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Create Bell/GHZ state
    qc.h(0)  # Superposition on first qubit
    
    # Entangle all qubits
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    # Measure all
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc


def create_bell_states_all_sizes() -> Dict[int, QuantumCircuit]:
    """
    Create Bell states for 2, 3, and 4 qubits.
    
    Returns:
        Dict mapping n_qubits to circuit
    """
    circuits = {}
    for n in [2, 3, 4]:
        circuits[n] = create_bell_state_baseline(n)
    return circuits


# ============================================
# 2. Fidelity Calculation
# ============================================

def calculate_bell_fidelity(counts: Dict[str, int], n_qubits: int) -> float:
    """
    Calculate fidelity of Bell/GHZ state from measurement counts.
    
    For ideal Bell/GHZ state, we expect:
    - 50% |00...0⟩
    - 50% |11...1⟩
    
    Args:
        counts: Measurement counts
        n_qubits: Number of qubits
    
    Returns:
        Fidelity value (0 to 1)
    """
    total_shots = sum(counts.values())
    
    # Expected outcomes for Bell/GHZ state
    all_zeros = '0' * n_qubits
    all_ones = '1' * n_qubits
    
    # Get counts for expected outcomes
    count_zeros = counts.get(all_zeros, 0)
    count_ones = counts.get(all_ones, 0)
    
    # Fidelity = probability of correct outcomes
    fidelity = (count_zeros + count_ones) / total_shots
    
    return fidelity


def calculate_state_fidelity_from_counts(counts: Dict[str, int], 
                                         ideal_state: Statevector) -> float:
    """
    Alternative: Calculate fidelity using Qiskit's state_fidelity.
    
    Args:
        counts: Measurement counts
        ideal_state: Ideal target state
    
    Returns:
        Fidelity value
    """
    # Convert counts to probabilities
    total = sum(counts.values())
    probs = {k: v/total for k, v in counts.items()}
    
    # Reconstruct density matrix (simplified)
    # For more accurate: use quantum tomography
    return calculate_bell_fidelity(counts, len(list(counts.keys())[0]))


# ============================================
# 3. Run Experiments on Real Hardware
# ============================================

def run_baseline_experiments(backend_name: str = BACKEND_NAME,
                            shots: int = SHOTS) -> Dict:
    """
    Run baseline Bell state experiments on IBM Quantum hardware.
    
    Args:
        backend_name: Name of IBM backend
        shots: Number of measurement shots
    
    Returns:
        Dictionary with results for each qubit count
    """
    print(f"🚀 Starting Baseline Experiments on {backend_name}")
    print(f"📊 Shots: {shots}")
    print("=" * 50)
    
    # Initialize IBM Quantum service
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        print(f"✅ Connected to {backend_name}")
        print(f"   Qubits: {backend.num_qubits}")
        print(f"   Status: {backend.status().status_msg}")
    except Exception as e:
        print(f"❌ Error connecting to backend: {e}")
        return {}
    
    # Create circuits
    circuits = create_bell_states_all_sizes()
    results = {}
    
    # Run each circuit
    for n_qubits, qc in circuits.items():
        print(f"\n{'='*50}")
        print(f"🔬 Running {n_qubits}-qubit Bell state...")
        
        try:
            # Transpile for hardware
            transpiled = transpile(qc, backend=backend, optimization_level=3)
            print(f"   Depth after transpilation: {transpiled.depth()}")
            print(f"   Gates: {transpiled.count_ops()}")
            
            # Run with Session
            with Session(backend=backend) as session:
                sampler = Sampler(session=session)
                job = sampler.run(transpiled, shots=shots)
                
                print(f"   Job ID: {job.job_id()}")
                print(f"   Status: {job.status()}")
                
                # Wait for result
                result = job.result()
                
                # Get counts (quasi_dists to counts)
                quasi_dist = result.quasi_dists[0]
                
                # Convert quasi_dist to counts format
                counts = {}
                for key, prob in quasi_dist.items():
                    # Convert integer key to binary string
                    bitstring = format(key, f'0{n_qubits}b')
                    counts[bitstring] = int(prob * shots)
                
                # Calculate fidelity
                fidelity = calculate_bell_fidelity(counts, n_qubits)
                
                # Store results
                results[n_qubits] = {
                    'circuit': qc,
                    'transpiled': transpiled,
                    'counts': counts,
                    'fidelity': fidelity,
                    'depth': transpiled.depth(),
                    'gates': dict(transpiled.count_ops()),
                    'job_id': job.job_id()
                }
                
                print(f"   ✅ Fidelity: {fidelity:.4f}")
                print(f"   Top outcomes: {sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]}")
                
        except Exception as e:
            print(f"   ❌ Error running {n_qubits}-qubit circuit: {e}")
            results[n_qubits] = {'error': str(e)}
    
    print(f"\n{'='*50}")
    print("✅ Baseline experiments completed!")
    
    return results


# ============================================
# 4. Analysis and Visualization
# ============================================

def analyze_baseline_results(results: Dict) -> None:
    """
    Analyze and visualize baseline results.
    
    Args:
        results: Results dictionary from run_baseline_experiments
    """
    print("\n📊 BASELINE RESULTS SUMMARY")
    print("=" * 50)
    
    qubits_list = []
    fidelities = []
    depths = []
    
    for n_qubits, data in sorted(results.items()):
        if 'error' not in data:
            qubits_list.append(n_qubits)
            fidelities.append(data['fidelity'])
            depths.append(data['depth'])
            
            print(f"\n{n_qubits} Qubits:")
            print(f"  Fidelity: {data['fidelity']:.4f}")
            print(f"  Circuit Depth: {data['depth']}")
            print(f"  Gates: {data['gates']}")
    
    # Create visualization
    if qubits_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Fidelity plot
        ax1.plot(qubits_list, fidelities, 'o-', linewidth=2, markersize=10, color='blue')
        ax1.set_xlabel('Number of Qubits', fontsize=12)
        ax1.set_ylabel('Bell State Fidelity', fontsize=12)
        ax1.set_title('Baseline: Fidelity vs System Size', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Depth plot
        ax2.bar(qubits_list, depths, color='orange', alpha=0.7)
        ax2.set_xlabel('Number of Qubits', fontsize=12)
        ax2.set_ylabel('Circuit Depth', fontsize=12)
        ax2.set_title('Baseline: Circuit Complexity', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('baseline_results.png', dpi=300, bbox_inches='tight')
        print("\n📈 Plots saved as 'baseline_results.png'")
        plt.show()


def save_results_to_file(results: Dict, filename: str = 'baseline_results.txt') -> None:
    """
    Save results to text file.
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("BASELINE EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for n_qubits, data in sorted(results.items()):
            f.write(f"{n_qubits} Qubits:\n")
            if 'error' in data:
                f.write(f"  Error: {data['error']}\n")
            else:
                f.write(f"  Fidelity: {data['fidelity']:.4f}\n")
                f.write(f"  Depth: {data['depth']}\n")
                f.write(f"  Job ID: {data['job_id']}\n")
                f.write(f"  Top counts: {sorted(data['counts'].items(), key=lambda x: x[1], reverse=True)[:5]}\n")
            f.write("\n")
    
    print(f"💾 Results saved to '{filename}'")


# ============================================
# 5. Main Execution
# ============================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   BASELINE BELL STATE EXPERIMENTS            ║
    ║   (Without QSF Framework)                    ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Run experiments
    results = run_baseline_experiments()
    
    # Analyze results
    if results:
        analyze_baseline_results(results)
        save_results_to_file(results)
        
        print("\n" + "="*50)
        print("✅ ALL BASELINE EXPERIMENTS COMPLETED!")
        print("="*50)
        print("\nNext Steps:")
        print("1. Compare these results with your QSF results")
        print("2. Calculate improvement percentages")
        print("3. Run statistical significance tests")
    else:
        print("\n❌ No results obtained. Check your IBM Quantum access.")
