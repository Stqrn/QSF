"""
Same-Device Baseline Comparison on ibm_brisbane
================================================
CRITICAL EXPERIMENT: Fair comparison QSF vs Baseline

This resolves the cross-backend comparison issue and provides
the foundation for all paper claims!
"""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================
# CRITICAL CONFIGURATION
# ============================================
BACKEND_NAME = 'ibm_brisbane'  # MUST match your QSF experiments!
SHOTS = 8192  # Match your QSF shots
OPTIMIZATION_LEVEL = 3  # Match your QSF optimization
N_QUBITS_LIST = [2, 3, 4]  # Test all sizes

# Coupling strengths from your QSF experiments
# Use the ones that showed positive results
COUPLING_VALUES = [0.1, 0.3, 0.8, 1.0]  # From your plots

# ============================================
# 1. Baseline Circuit (NO QSF)
# ============================================

def create_baseline_bell_state(n_qubits: int) -> QuantumCircuit:
    """
    Create simple Bell/GHZ state WITHOUT any QSF components.
    
    This is the PURE baseline for comparison.
    
    Args:
        n_qubits: Number of system qubits
    
    Returns:
        QuantumCircuit without QSF
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Standard Bell/GHZ state preparation
    qc.h(0)  # Hadamard on first qubit
    
    # Chain of CNOTs to create entanglement
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc


def create_baseline_circuits() -> Dict[int, QuantumCircuit]:
    """
    Create baseline circuits for all qubit counts.
    
    Returns:
        Dictionary mapping n_qubits to circuit
    """
    circuits = {}
    for n in N_QUBITS_LIST:
        circuits[n] = create_baseline_bell_state(n)
    return circuits


# ============================================
# 2. Fidelity Calculation
# ============================================

def calculate_bell_fidelity(counts: Dict[str, int], n_qubits: int) -> float:
    """
    Calculate fidelity for Bell/GHZ state.
    
    Ideal state: (|00...0⟩ + |11...1⟩) / √2
    Fidelity = P(00...0) + P(11...1)
    
    Args:
        counts: Measurement counts dictionary
        n_qubits: Number of qubits
    
    Returns:
        Fidelity value [0, 1]
    """
    total_shots = sum(counts.values())
    
    # Expected outcomes
    all_zeros = '0' * n_qubits
    all_ones = '1' * n_qubits
    
    # Count correct outcomes
    correct_shots = counts.get(all_zeros, 0) + counts.get(all_ones, 0)
    
    # Fidelity
    fidelity = correct_shots / total_shots
    
    return fidelity


def calculate_fidelity_with_error(counts: Dict[str, int], 
                                   n_qubits: int,
                                   shots: int) -> Tuple[float, float]:
    """
    Calculate fidelity with statistical error estimate.
    
    Args:
        counts: Measurement counts
        n_qubits: Number of qubits
        shots: Total shots
    
    Returns:
        (fidelity, error_estimate)
    """
    fidelity = calculate_bell_fidelity(counts, n_qubits)
    
    # Statistical error: √(F(1-F)/N)
    error = np.sqrt(fidelity * (1 - fidelity) / shots)
    
    return fidelity, error


# ============================================
# 3. Run Baseline Experiments
# ============================================

def run_baseline_on_brisbane(shots: int = SHOTS,
                             optimization_level: int = OPTIMIZATION_LEVEL) -> Dict:
    """
    Run baseline experiments on ibm_brisbane.
    
    THIS IS THE CRITICAL EXPERIMENT!
    
    Args:
        shots: Number of measurement shots
        optimization_level: Transpiler optimization level
    
    Returns:
        Complete results dictionary
    """
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   🎯 CRITICAL BASELINE EXPERIMENT                    ║
    ║   Same-Device Comparison on ibm_brisbane             ║
    ║                                                       ║
    ║   This will enable fair QSF vs Baseline comparison!  ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Backend: {BACKEND_NAME}")
    print(f"   Shots: {shots}")
    print(f"   Optimization: Level {optimization_level}")
    print(f"   Qubits: {N_QUBITS_LIST}")
    print("\n" + "="*60)
    
    # Initialize service
    try:
        service = QiskitRuntimeService()
        backend = service.backend(BACKEND_NAME)
        print(f"\n✅ Connected to {BACKEND_NAME}")
        print(f"   Status: {backend.status().status_msg}")
        print(f"   Queue: {backend.status().pending_jobs} jobs")
    except Exception as e:
        print(f"\n❌ Error connecting to backend: {e}")
        return {}
    
    # Create circuits
    circuits = create_baseline_circuits()
    
    # Results storage
    results = {
        'backend': BACKEND_NAME,
        'timestamp': datetime.now().isoformat(),
        'shots': shots,
        'optimization_level': optimization_level,
        'experiments': {}
    }
    
    # Run each circuit
    for n_qubits, qc in circuits.items():
        print(f"\n{'='*60}")
        print(f"🔬 Running BASELINE for {n_qubits} qubits...")
        print('='*60)
        
        try:
            # Transpile
            print(f"\n📐 Transpiling circuit...")
            qc_transpiled = transpile(
                qc, 
                backend=backend, 
                optimization_level=optimization_level
            )
            
            print(f"   Original depth: {qc.depth()}")
            print(f"   Transpiled depth: {qc_transpiled.depth()}")
            print(f"   Gate counts: {dict(qc_transpiled.count_ops())}")
            
            # Run on hardware
            print(f"\n🚀 Submitting to {BACKEND_NAME}...")
            
            with Session(backend=backend) as session:
                sampler = Sampler(session=session)
                job = sampler.run(qc_transpiled, shots=shots)
                
                job_id = job.job_id()
                print(f"   Job ID: {job_id}")
                print(f"   Status: {job.status()}")
                
                # Wait for result
                print(f"\n⏳ Waiting for results...")
                result = job.result()
                print(f"   ✅ Job completed!")
                
                # Extract counts
                quasi_dist = result.quasi_dists[0]
                counts = {}
                for key, prob in quasi_dist.items():
                    bitstring = format(key, f'0{n_qubits}b')
                    counts[bitstring] = int(prob * shots)
                
                # Calculate fidelity
                fidelity, error = calculate_fidelity_with_error(
                    counts, n_qubits, shots
                )
                
                # Store results
                results['experiments'][n_qubits] = {
                    'fidelity': fidelity,
                    'fidelity_error': error,
                    'counts': counts,
                    'depth': qc_transpiled.depth(),
                    'gate_counts': dict(qc_transpiled.count_ops()),
                    'job_id': job_id,
                    'success': True
                }
                
                # Display results
                print(f"\n📊 Results:")
                print(f"   Fidelity: {fidelity:.4f} ± {error:.4f}")
                print(f"   Depth: {qc_transpiled.depth()}")
                
                # Top 5 outcomes
                sorted_counts = sorted(
                    counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                print(f"\n   Top outcomes:")
                for bitstring, count in sorted_counts:
                    prob = count / shots
                    print(f"      |{bitstring}⟩: {count:5d} ({prob:.3f})")
                
        except Exception as e:
            print(f"\n❌ Error for {n_qubits} qubits: {e}")
            results['experiments'][n_qubits] = {
                'success': False,
                'error': str(e)
            }
    
    print(f"\n{'='*60}")
    print("✅ BASELINE EXPERIMENTS COMPLETED!")
    print('='*60)
    
    return results


# ============================================
# 4. Save Results
# ============================================

def save_results(results: Dict, filename: str = None):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        filename: Output filename (auto-generated if None)
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_brisbane_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Also save human-readable summary
    txt_filename = filename.replace('.json', '_summary.txt')
    with open(txt_filename, 'w') as f:
        f.write("BASELINE RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Backend: {results['backend']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Shots: {results['shots']}\n")
        f.write(f"Optimization: Level {results['optimization_level']}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*60 + "\n")
        for n_qubits, data in sorted(results['experiments'].items()):
            if data['success']:
                f.write(f"\n{n_qubits} Qubits:\n")
                f.write(f"  Fidelity: {data['fidelity']:.4f} ± {data['fidelity_error']:.4f}\n")
                f.write(f"  Depth: {data['depth']}\n")
                f.write(f"  Job ID: {data['job_id']}\n")
            else:
                f.write(f"\n{n_qubits} Qubits: FAILED\n")
                f.write(f"  Error: {data['error']}\n")
    
    print(f"📄 Summary saved to: {txt_filename}")


# ============================================
# 5. Quick Visualization
# ============================================

def plot_baseline_results(results: Dict):
    """
    Create quick visualization of baseline results.
    
    Args:
        results: Results dictionary
    """
    # Extract data
    qubits = []
    fidelities = []
    errors = []
    depths = []
    
    for n, data in sorted(results['experiments'].items()):
        if data['success']:
            qubits.append(n)
            fidelities.append(data['fidelity'])
            errors.append(data['fidelity_error'])
            depths.append(data['depth'])
    
    if not qubits:
        print("⚠️  No successful results to plot!")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Fidelity plot
    ax1.errorbar(qubits, fidelities, yerr=errors, 
                fmt='o-', capsize=5, linewidth=2, markersize=10,
                color='blue', label='Baseline (ibm_brisbane)')
    ax1.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bell State Fidelity', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline Fidelity vs System Size', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add values on plot
    for x, y, err in zip(qubits, fidelities, errors):
        ax1.text(x, y + 0.05, f'{y:.3f}±{err:.3f}', 
                ha='center', fontsize=9)
    
    # Depth plot
    ax2.bar(qubits, depths, color='orange', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
    ax2.set_title('Baseline Circuit Complexity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for x, y in zip(qubits, depths):
        ax2.text(x, y + 0.5, str(y), ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_brisbane_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved as: {filename}")
    
    plt.show()


# ============================================
# 6. Comparison with QSF (if data available)
# ============================================

def compare_with_qsf(baseline_results: Dict, 
                     qsf_results: Dict = None):
    """
    Compare baseline results with QSF results.
    
    Args:
        baseline_results: Baseline results from this experiment
        qsf_results: QSF results (if available)
    """
    print("\n" + "="*60)
    print("📊 BASELINE vs QSF COMPARISON")
    print("="*60)
    
    if qsf_results is None:
        print("\n⚠️  QSF results not provided.")
        print("📝 TODO: Load your QSF results and compare!")
        print("\nExpected format:")
        print("qsf_results = {")
        print("  2: {'fidelity': 0.XX, 'coupling': 0.3},")
        print("  3: {'fidelity': 0.XX, 'coupling': 0.3},")
        print("  4: {'fidelity': 0.XX, 'coupling': 0.3}")
        print("}")
        return
    
    # Comparison table
    print("\nCOMPARISON TABLE:")
    print("-"*60)
    print(f"{'Qubits':<8} {'Baseline':<12} {'QSF':<12} {'ΔF':<12} {'Improvement'}")
    print("-"*60)
    
    for n in sorted(baseline_results['experiments'].keys()):
        if baseline_results['experiments'][n]['success']:
            base_f = baseline_results['experiments'][n]['fidelity']
            
            if n in qsf_results:
                qsf_f = qsf_results[n]['fidelity']
                delta_f = qsf_f - base_f
                improvement = (delta_f / base_f) * 100
                
                symbol = "✅" if delta_f > 0 else "❌"
                
                print(f"{n:<8} {base_f:.4f}      {qsf_f:.4f}      "
                      f"{delta_f:+.4f}      {improvement:+.2f}% {symbol}")
    
    print("-"*60)


# ============================================
# 7. Main Execution
# ============================================

def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("STARTING CRITICAL BASELINE EXPERIMENT")
    print("="*60)
    print("\n⚠️  IMPORTANT:")
    print("   This experiment provides the fair comparison")
    print("   needed for the paper!")
    print("\n   Estimated time: 30-60 minutes")
    print("   (depending on queue)")
    print("\n" + "="*60)
    
    # Confirmation
    response = input("\nProceed with experiment? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Experiment cancelled.")
        return
    
    # Run experiments
    results = run_baseline_on_brisbane()
    
    if results and results['experiments']:
        # Save results
        save_results(results)
        
        # Visualize
        plot_baseline_results(results)
        
        # Compare with QSF (if available)
        print("\n" + "="*60)
        print("NEXT STEP: Compare with your QSF results!")
        print("="*60)
        print("\nLoad your QSF results and run:")
        print("compare_with_qsf(baseline_results, qsf_results)")
        
    else:
        print("\n❌ Experiment failed. Check errors above.")


if __name__ == "__main__":
    main()


# ============================================
# 8. Helper: Load and Compare Later
# ============================================

def load_and_compare(baseline_file: str, qsf_file: str = None):
    """
    Helper function to load results and compare later.
    
    Args:
        baseline_file: Path to baseline results JSON
        qsf_file: Path to QSF results JSON (optional)
    """
    # Load baseline
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    print("✅ Loaded baseline results")
    
    # Load QSF if provided
    if qsf_file:
        with open(qsf_file, 'r') as f:
            qsf = json.load(f)
        print("✅ Loaded QSF results")
        
        # Compare
        compare_with_qsf(baseline, qsf)
    else:
        print("⚠️  No QSF file provided")
        print("   Showing baseline results only:")
        plot_baseline_results(baseline)


# ============================================
# Usage Examples
# ============================================

"""
USAGE:

1. Run the experiment:
   python this_script.py
   
2. Or in Python:
   results = run_baseline_on_brisbane()
   save_results(results)
   plot_baseline_results(results)

3. Later, compare with QSF:
   load_and_compare('baseline_brisbane_20241030.json', 
                    'qsf_results.json')

4. Or manually:
   qsf_results = {
       2: {'fidelity': 0.86, 'coupling': 0.3},
       3: {'fidelity': 0.73, 'coupling': 0.3},
       4: {'fidelity': 0.85, 'coupling': 0.3}
   }
   compare_with_qsf(baseline_results, qsf_results)
"""
