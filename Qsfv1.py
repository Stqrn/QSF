"""
Quantum Image Recovery with QSF
================================
Simplified version for IBM Quantum hardware
Encodes a small image, applies noise, and recovers with QSF
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from datetime import datetime
import json

# Configuration
IBM_QUANTUM_TOKEN = "MqKZLaGHJWs0-CEVoXexzaxGfLE1-1gn8DgNJSltwfhh"
BACKEND_NAME = "ibm_torino"
SHOTS = 8192
OPTIMIZATION_LEVEL = 3

# Simple 2x2 image (4 pixels)
SAMPLE_IMAGE = np.array([
    [0.8, 0.6],
    [0.4, 0.2]
])

print("="*60)
print("QUANTUM IMAGE RECOVERY WITH QSF")
print("="*60)

# ============================================
# Step 1: Encode Image as Quantum State
# ============================================

def image_to_amplitudes(image_matrix):
    """Convert image to normalized quantum amplitudes.

    Fix: image contains intensities (non-negative). We convert intensities -> amplitudes
    by taking the sqrt of intensities and then normalizing the amplitude vector.
    """
    print("\n📊 Encoding image as quantum state...")

    # Flatten image
    flattened = np.asarray(image_matrix).flatten()

    if np.any(flattened < 0):
        raise ValueError("Image intensities must be non-negative.")

    # Convert intensities -> amplitudes via sqrt
    amplitudes = np.sqrt(flattened)

    # Normalize to unit vector (quantum amplitudes)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        normalized = amplitudes / norm
    else:
        normalized = amplitudes  # all zeros case (shouldn't happen for a real image)

    # Calculate number of qubits needed
    n_qubits = int(np.ceil(np.log2(len(normalized))))

    # Pad to power of 2
    target_length = 2**n_qubits
    if len(normalized) < target_length:
        padded = np.zeros(target_length)
        padded[:len(normalized)] = normalized
        normalized = padded

    print(f"   Image shape: {image_matrix.shape}")
    print(f"   Qubits needed: {n_qubits}")
    print(f"   Amplitudes: {normalized}")

    return normalized, n_qubits

def create_encoding_circuit(amplitudes):
    """Create circuit to encode amplitudes."""
    n_qubits = int(np.log2(len(amplitudes)))
    qc = QuantumCircuit(n_qubits)

    # Initialize with amplitudes
    qc.initialize(amplitudes, range(n_qubits))

    return qc

# ============================================
# Step 2: Apply Noise (Simulated Attack)
# ============================================

def apply_noise(circuit, noise_strength=0.3):
    """Apply noise to simulate quantum attack."""
    print(f"\n⚠️  Applying noise (strength: {noise_strength})...")

    noisy_circuit = circuit.copy()
    n_qubits = circuit.num_qubits

    for qubit in range(n_qubits):
        # Phase noise
        noisy_circuit.rz(noise_strength * np.pi, qubit)
        # Amplitude noise
        noisy_circuit.rx(noise_strength * 0.5, qubit)

    print("   ✅ Noise applied")
    return noisy_circuit

# ============================================
# Step 3: QSF Recovery
# ============================================

def create_qsf_recovery(circuit, lambda_strength=0.5):
    """Apply QSF recovery protocol."""
    print(f"\n🔧 Applying QSF recovery (λ={lambda_strength})...")

    n_sys = circuit.num_qubits
    n_total = n_sys + 1  # Add 1 ancilla

    # Create new circuit with ancilla
    qc = QuantumCircuit(n_total, n_sys)

    # Copy original (noisy) circuit
    qc.compose(circuit, range(n_sys), inplace=True)
    qc.barrier()

    # Synergic coupling with ancilla
    for qubit in range(n_sys):
        qc.cx(qubit, n_sys)  # Entangle with ancilla
        qc.ry(lambda_strength, n_sys)
        qc.cx(qubit, n_sys)

    qc.barrier()

    # Recovery operations
    for qubit in range(n_sys):
        qc.ry(-lambda_strength * 0.8, qubit)
        qc.rz(-lambda_strength * 0.5, qubit)

    qc.barrier()

    # Measure system qubits
    qc.measure(range(n_sys), range(n_sys))

    print("   ✅ QSF recovery applied")
    return qc

# ============================================
# Step 4: Run on IBM Quantum
# ============================================

def run_on_ibm_quantum(image_matrix, noise_strength=0.3, lambda_strength=0.5):
    """Run complete pipeline on IBM Quantum."""

    print(f"\n{'='*60}")
    print("RUNNING ON IBM QUANTUM HARDWARE")
    print('='*60)

    # Connect to backend
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=IBM_QUANTUM_TOKEN
    )

    try:
        backend = service.backend(BACKEND_NAME)
        print(f"✅ Using: {BACKEND_NAME}")
    except:
        backend = service.least_busy(operational=True, simulator=False)
        print(f"✅ Using: {backend.name}")

    # Step 1: Encode image
    amplitudes, n_qubits = image_to_amplitudes(image_matrix)
    original_circuit = create_encoding_circuit(amplitudes)

    # Step 2: Apply noise
    noisy_circuit = apply_noise(original_circuit, noise_strength)

    # Step 3: Create recovery circuits
    # Without QSF (just measure noisy state)
    noisy_measured = noisy_circuit.copy()
    noisy_measured.measure_all()

    # With QSF recovery
    recovered_circuit = create_qsf_recovery(noisy_circuit, lambda_strength)

    # Transpile
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=OPTIMIZATION_LEVEL
    )

    isa_noisy = pm.run(noisy_measured)
    isa_recovered = pm.run(recovered_circuit)

    print(f"\n📋 Circuit depths:")
    print(f"   Without QSF: {isa_noisy.depth()}")
    print(f"   With QSF: {isa_recovered.depth()}")

    # Run on hardware
    print(f"\n🚀 Submitting to {backend.name}...")
    sampler = Sampler(mode=backend)

    job = sampler.run([isa_noisy, isa_recovered], shots=SHOTS)
    print(f"   Job ID: {job.job_id()}")
    print(f"   Waiting for results...")

    result = job.result()

    # Extract counts
    # Note: keeping original structure but ensure conversion to dict for safety
    try:
        counts_noisy = result[0].data.meas.get_counts()
    except Exception:
        # Fallback: try common alternative access patterns
        try:
            counts_noisy = result[0].data.get('counts', {})
        except Exception:
            counts_noisy = {}

    try:
        counts_recovered = result[1].data.c.get_counts()
    except Exception:
        try:
            counts_recovered = result[1].data.get('counts', {})
        except Exception:
            counts_recovered = {}

    # ensure plain dicts
    counts_noisy = dict(counts_noisy) if counts_noisy is not None else {}
    counts_recovered = dict(counts_recovered) if counts_recovered is not None else {}

    print("\n✅ Results received!")

    return {
        'original_amplitudes': amplitudes,
        'counts_noisy': dict(counts_noisy),
        'counts_recovered': dict(counts_recovered),
        'backend': backend.name,
        'noise_strength': noise_strength,
        'lambda_strength': lambda_strength,
        'n_qubits': n_qubits
    }

# ============================================
# Step 5: Reconstruct and Compare
# ============================================

def counts_to_amplitudes(counts, n_amplitudes, bitstring_order='little'):
    """Reconstruct amplitudes from measurement counts.

    Fixes:
    - Probability = counts/total, amplitude = sqrt(probability)
    - Normalize amplitude vector after sqrt
    - Handles common bitstring ordering (Qiskit may return MSB-first)
    """
    total = sum(counts.values()) if len(counts) > 0 else 1
    amplitudes = np.zeros(n_amplitudes, dtype=float)

    for bitstring, count in counts.items():
        if bitstring_order == 'little':
            # treat bitstring as least-significant-bit first (reverse for index)
            idx = int(bitstring[::-1], 2)
        else:
            idx = int(bitstring, 2)

        if idx < n_amplitudes:
            prob = count / total
            amplitudes[idx] = np.sqrt(prob)

    # After taking sqrt(prob) vector might not be unit norm due to sampling noise — normalize it
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes = amplitudes / norm

    return amplitudes

def reconstruct_image(amplitudes, original_shape):
    """Reconstruct image from amplitudes.

    Fix: intensities = |amplitude|^2, then reshape and normalize for display.
    """
    n_pixels = np.prod(original_shape)

    # Take first n_pixels amplitudes
    image_values = np.abs(amplitudes[:n_pixels]) ** 2  # Convert to intensities (probabilities)

    # Reshape
    reconstructed = image_values.reshape(original_shape)

    # Normalize for display (optional)
    if np.max(reconstructed) > 0:
        reconstructed = reconstructed / np.max(reconstructed)

    return reconstructed

def calculate_fidelity(original, reconstructed):
    """Calculate image fidelity (overlap) correctly.

    Behavior:
    - original: original image intensities (matrix)
    - reconstructed: reconstructed image intensities (matrix)
    We convert both to amplitude vectors (sqrt of intensities) and compute
    fidelity for pure states: |<psi|phi>|^2.
    """
    # Flatten and turn intensities -> amplitudes
    orig_flat = np.asarray(original).flatten()
    rec_flat = np.asarray(reconstructed).flatten()

    # Convert intensities -> amplitudes (sqrt) and normalize
    orig_amps = np.sqrt(orig_flat)
    rec_amps = np.sqrt(rec_flat)

    # Normalize
    if np.linalg.norm(orig_amps) > 0:
        orig_amps = orig_amps / np.linalg.norm(orig_amps)
    if np.linalg.norm(rec_amps) > 0:
        rec_amps = rec_amps / np.linalg.norm(rec_amps)

    # compute overlap and fidelity
    overlap = np.vdot(orig_amps, rec_amps)  # conjugate(orig) · rec
    fidelity = np.abs(overlap) ** 2

    return float(fidelity)

# ============================================
# Step 6: Visualization
# ============================================

def plot_results(original_img, noisy_img, recovered_img, results):
    """Create comprehensive visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Images
    ax1, ax2, ax3 = axes[0]

    im1 = ax1.imshow(original_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('After Noise (No Recovery)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    im3 = ax3.imshow(recovered_img, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Recovered with QSF', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: Analysis
    ax4, ax5, ax6 = axes[1]

    # Pixel comparison
    pixels = range(original_img.size)
    orig_flat = original_img.flatten()
    noisy_flat = noisy_img.flatten()
    recovered_flat = recovered_img.flatten()

    ax4.plot(pixels, orig_flat, 'bo-', label='Original', linewidth=2, markersize=8)
    ax4.plot(pixels, noisy_flat, 'ro-', label='Noisy', linewidth=2, markersize=8)
    ax4.plot(pixels, recovered_flat, 'go-', label='Recovered', linewidth=2, markersize=8)
    ax4.set_xlabel('Pixel Index', fontweight='bold')
    ax4.set_ylabel('Intensity', fontweight='bold')
    ax4.set_title('Pixel Values', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Measurement distributions
    states_noisy = list(results['counts_noisy'].keys())
    values_noisy = list(results['counts_noisy'].values())

    ax5.bar(states_noisy, values_noisy, color='red', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('State', fontweight='bold')
    ax5.set_ylabel('Counts', fontweight='bold')
    ax5.set_title('Noisy Measurements', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)

    states_rec = list(results['counts_recovered'].keys())
    values_rec = list(results['counts_recovered'].values())

    ax6.bar(states_rec, values_rec, color='green', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('State', fontweight='bold')
    ax6.set_ylabel('Counts', fontweight='bold')
    ax6.set_title('Recovered Measurements', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)

    plt.suptitle(f'Quantum Image Recovery on {results["backend"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'image_recovery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: {filename}")
    plt.show()

# ============================================
# Main Execution
# ============================================

def main():
    """Run complete image recovery experiment."""

    print(f"\nOriginal Image:")
    print(SAMPLE_IMAGE)

    # Run on IBM Quantum
    results = run_on_ibm_quantum(
        SAMPLE_IMAGE,
        noise_strength=0.3,
        lambda_strength=0.7
    )

    # Reconstruct images
    print("\n📊 Reconstructing images from measurements...")

    n_amplitudes = len(results['original_amplitudes'])

    noisy_amps = counts_to_amplitudes(results['counts_noisy'], n_amplitudes)
    recovered_amps = counts_to_amplitudes(results['counts_recovered'], n_amplitudes)

    noisy_img = reconstruct_image(noisy_amps, SAMPLE_IMAGE.shape)
    recovered_img = reconstruct_image(recovered_amps, SAMPLE_IMAGE.shape)

    print(f"\nReconstructed Noisy Image:")
    print(noisy_img)
    print(f"\nReconstructed Recovered Image:")
    print(recovered_img)

    # Calculate fidelities
    fid_noisy = calculate_fidelity(SAMPLE_IMAGE, noisy_img)
    fid_recovered = calculate_fidelity(SAMPLE_IMAGE, recovered_img)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Noisy fidelity:     {fid_noisy:.4f}")
    print(f"Recovered fidelity: {fid_recovered:.4f}")
    print(f"Improvement:        {fid_recovered - fid_noisy:+.4f}")
    print("="*60)

    # Visualize
    plot_results(SAMPLE_IMAGE, noisy_img, recovered_img, results)

    # Save results
    results['fidelity_noisy'] = float(fid_noisy)
    results['fidelity_recovered'] = float(fid_recovered)
    results['improvement'] = float(fid_recovered - fid_noisy)

    filename = f'image_recovery_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert numpy arrays to lists for JSON
    results_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in results.items()}

    with open(filename, 'w') as f:
        json.dump(results_save, f, indent=2)

    print(f"💾 Results saved: {filename}")
    print("\n✅ Quantum image recovery experiment complete!")

if __name__ == "__main__":
    main()
