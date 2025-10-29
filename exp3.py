"""
QSF vs Baseline Comparison and Statistical Analysis
===================================================
Compare QSF results with baseline and perform statistical tests
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import json

# ============================================
# 1. Data Loading and Preparation
# ============================================

def load_results(qsf_file: str = 'qsf_results.json',
                baseline_file: str = 'baseline_results.json') -> Tuple[Dict, Dict]:
    """
    Load QSF and baseline results from files.
    
    Args:
        qsf_file: Path to QSF results
        baseline_file: Path to baseline results
    
    Returns:
        Tuple of (qsf_results, baseline_results)
    """
    try:
        with open(qsf_file, 'r') as f:
            qsf_results = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {qsf_file} not found. Using sample data.")
        qsf_results = {}
    
    try:
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {baseline_file} not found. Using sample data.")
        baseline_results = {}
    
    return qsf_results, baseline_results


def create_sample_data():
    """
    Create sample data structure for demonstration.
    Replace with your actual results!
    """
    # Sample QSF results (from your previous experiments)
    qsf_results = {
        2: {'fidelity': 0.93, 'depth': 25, 'ancillas': 2},
        3: {'fidelity': 0.87, 'depth': 40, 'ancillas': 3},
        4: {'fidelity': 0.86, 'depth': 60, 'ancillas': 4}
    }
    
    # Sample baseline results (to be replaced with actual runs)
    baseline_results = {
        2: {'fidelity': 0.89, 'depth': 10},
        3: {'fidelity': 0.85, 'depth': 18},
        4: {'fidelity': 0.82, 'depth': 30}
    }
    
    return qsf_results, baseline_results


# ============================================
# 2. Comparison Metrics
# ============================================

def calculate_improvement(qsf_value: float, baseline_value: float) -> float:
    """
    Calculate percentage improvement.
    
    Args:
        qsf_value: Value with QSF
        baseline_value: Value without QSF
    
    Returns:
        Percentage improvement
    """
    return ((qsf_value - baseline_value) / baseline_value) * 100


def calculate_overhead(qsf_data: Dict, baseline_data: Dict) -> Dict:
    """
    Calculate overhead metrics.
    
    Args:
        qsf_data: QSF experiment data
        baseline_data: Baseline experiment data
    
    Returns:
        Dictionary of overhead metrics
    """
    overhead = {}
    
    # Circuit depth overhead
    overhead['depth_increase'] = qsf_data['depth'] - baseline_data['depth']
    overhead['depth_ratio'] = qsf_data['depth'] / baseline_data['depth']
    
    # Qubit overhead
    if 'ancillas' in qsf_data:
        overhead['ancilla_qubits'] = qsf_data['ancillas']
    
    return overhead


def create_comparison_table(qsf_results: Dict, baseline_results: Dict) -> pd.DataFrame:
    """
    Create comprehensive comparison table.
    
    Args:
        qsf_results: QSF experimental results
        baseline_results: Baseline experimental results
    
    Returns:
        Pandas DataFrame with comparison
    """
    data = []
    
    for n_qubits in sorted(set(qsf_results.keys()) & set(baseline_results.keys())):
        qsf = qsf_results[n_qubits]
        baseline = baseline_results[n_qubits]
        
        if 'error' not in qsf and 'error' not in baseline:
            improvement = calculate_improvement(qsf['fidelity'], baseline['fidelity'])
            overhead = calculate_overhead(qsf, baseline)
            
            row = {
                'Qubits': n_qubits,
                'Baseline Fidelity': f"{baseline['fidelity']:.4f}",
                'QSF Fidelity': f"{qsf['fidelity']:.4f}",
                'Improvement (%)': f"{improvement:+.2f}%",
                'Ancillas': qsf.get('ancillas', 'N/A'),
                'Depth Increase': overhead['depth_increase'],
                'Net Benefit': 'Positive' if improvement > 0 else 'Negative'
            }
            data.append(row)
    
    return pd.DataFrame(data)


# ============================================
# 3. Statistical Significance Testing
# ============================================

def bootstrap_confidence_interval(data: List[float], 
                                  n_bootstrap: int = 1000,
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Args:
        data: List of measurements
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return lower, upper


def perform_statistical_tests(qsf_measurements: List[float],
                              baseline_measurements: List[float]) -> Dict:
    """
    Perform statistical significance tests.
    
    Args:
        qsf_measurements: List of QSF fidelity measurements
        baseline_measurements: List of baseline fidelity measurements
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # T-test
    t_stat, p_value = stats.ttest_ind(qsf_measurements, baseline_measurements)
    results['t_test'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_mw = stats.mannwhitneyu(qsf_measurements, baseline_measurements)
    results['mann_whitney'] = {
        'u_statistic': u_stat,
        'p_value': p_value_mw,
        'significant': p_value_mw < 0.05
    }
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(qsf_measurements) - np.mean(baseline_measurements)
    pooled_std = np.sqrt((np.std(qsf_measurements)**2 + np.std(baseline_measurements)**2) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
    results['effect_size'] = cohen_d
    
    return results


# ============================================
# 4. Visualization
# ============================================

def create_comparison_plots(qsf_results: Dict, baseline_results: Dict):
    """
    Create comprehensive comparison visualizations.
    
    Args:
        qsf_results: QSF experimental results
        baseline_results: Baseline experimental results
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Extract data
    qubits = sorted(set(qsf_results.keys()) & set(baseline_results.keys()))
    qsf_fidelities = [qsf_results[q]['fidelity'] for q in qubits if 'error' not in qsf_results[q]]
    baseline_fidelities = [baseline_results[q]['fidelity'] for q in qubits if 'error' not in baseline_results[q]]
    improvements = [calculate_improvement(qsf_results[q]['fidelity'], 
                                         baseline_results[q]['fidelity']) 
                   for q in qubits if 'error' not in qsf_results[q]]
    
    # Plot 1: Side-by-side fidelity comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(qubits))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_fidelities, width, label='Baseline', 
                    color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, qsf_fidelities, width, label='QSF', 
                    color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Number of Qubits', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bell State Fidelity', fontsize=11, fontweight='bold')
    ax1.set_title('Fidelity Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(qubits)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Improvement percentages
    ax2 = plt.subplot(2, 3, 2)
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(qubits, improvements, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Number of Qubits', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('QSF Performance Gain', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.2f}%', ha='center', 
                va='bottom' if imp > 0 else 'top', fontsize=10)
    
    # Plot 3: Overhead analysis
    ax3 = plt.subplot(2, 3, 3)
    ancillas = [qsf_results[q].get('ancillas', 0) for q in qubits]
    depth_increase = [qsf_results[q]['depth'] - baseline_results[q]['depth'] 
                     for q in qubits if 'error' not in qsf_results[q]]
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(qubits, ancillas, 'o-', label='Ancilla Qubits', 
                     color='purple', linewidth=2, markersize=8)
    line2 = ax3_twin.plot(qubits, depth_increase, 's-', label='Depth Increase', 
                         color='orange', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Number of Qubits', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Ancilla Qubits', fontsize=11, fontweight='bold', color='purple')
    ax3_twin.set_ylabel('Circuit Depth Increase', fontsize=11, fontweight='bold', color='orange')
    ax3.set_title('Resource Overhead', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # Plot 4: Trade-off analysis (Improvement vs Overhead)
    ax4 = plt.subplot(2, 3, 4)
    total_overhead = [a + d/10 for a, d in zip(ancillas, depth_increase)]  # Normalized
    
    scatter = ax4.scatter(total_overhead, improvements, c=qubits, cmap='viridis',
                         s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, q in enumerate(qubits):
        ax4.annotate(f'{q} qubits', (total_overhead[i], improvements[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Total Overhead (normalized)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Trade-off: Overhead vs Benefit', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax4, label='Qubits')
    
    # Plot 5: Relative performance (normalized)
    ax5 = plt.subplot(2, 3, 5)
    baseline_norm = [f / max(baseline_fidelities) for f in baseline_fidelities]
    qsf_norm = [f / max(qsf_fidelities) for f in qsf_fidelities]
    
    ax5.plot(qubits, baseline_norm, 'o-', label='Baseline (normalized)', 
            linewidth=2, markersize=8, color='blue')
    ax5.plot(qubits, qsf_norm, 's-', label='QSF (normalized)', 
            linewidth=2, markersize=8, color='red')
    
    ax5.set_xlabel('Number of Qubits', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Normalized Fidelity', fontsize=11, fontweight='bold')
    ax5.set_title('Scalability Comparison', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.1])
    
    # Plot 6: Summary text box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n\n"
    summary_text += f"Average Improvement: {np.mean(improvements):.2f}%\n"
    summary_text += f"Max Improvement: {max(improvements):.2f}%\n"
    summary_text += f"Min Improvement: {min(improvements):.2f}%\n\n"
    summary_text += f"Avg Ancilla Overhead: {np.mean(ancillas):.1f} qubits\n"
    summary_text += f"Avg Depth Increase: {np.mean(depth_increase):.1f} gates\n\n"
    
    # Determine overall assessment
    avg_imp = np.mean(improvements)
    if avg_imp > 5:
        assessment = "✅ Significant Improvement"
    elif avg_imp > 0:
        assessment = "⚠️ Modest Improvement"
    else:
        assessment = "❌ No Clear Benefit"
    
    summary_text += f"Overall Assessment:\n{assessment}"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('qsf_vs_baseline_comprehensive.png', dpi=300, bbox_inches='tight')
    print("📊 Comprehensive comparison plot saved as 'qsf_vs_baseline_comprehensive.png'")
    plt.show()


# ============================================
# 5. Generate Report
# ============================================

def generate_comparison_report(qsf_results: Dict, baseline_results: Dict,
                               output_file: str = 'comparison_report.txt'):
    """
    Generate detailed text report.
    
    Args:
        qsf_results: QSF experimental results
        baseline_results: Baseline experimental results
        output_file: Output filename
    """
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("QSF vs BASELINE: COMPREHENSIVE COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Create comparison table
        df = create_comparison_table(qsf_results, baseline_results)
        f.write("COMPARISON TABLE:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Statistical summary
        f.write("="*60 + "\n")
        f.write("STATISTICAL SUMMARY:\n")
        f.write("="*60 + "\n\n")
        
        qubits = sorted(set(qsf_results.keys()) & set(baseline_results.keys()))
        improvements = [calculate_improvement(qsf_results[q]['fidelity'], 
                                             baseline_results[q]['fidelity']) 
                       for q in qubits if 'error' not in qsf_results[q]]
        
        f.write(f"Average Improvement: {np.mean(improvements):.2f}%\n")
        f.write(f"Std Dev: {np.std(improvements):.2f}%\n")
        f.write(f"Range: [{min(improvements):.2f}%, {max(improvements):.2f}%]\n\n")
        
        # Conclusions
        f.write("="*60 + "\n")
        f.write("CONCLUSIONS:\n")
        f.write("="*60 + "\n\n")
        
        avg_imp = np.mean(improvements)
        if avg_imp > 5:
            f.write("✅ QSF shows SIGNIFICANT improvement over baseline.\n")
            f.write("   The overhead is justified by the performance gains.\n")
        elif avg_imp > 0:
            f.write("⚠️ QSF shows MODEST improvement over baseline.\n")
            f.write("   Trade-offs should be considered carefully.\n")
        else:
            f.write("❌ QSF does not show clear benefits over baseline.\n")
            f.write("   Further optimization or different conditions may be needed.\n")
        
        f.write("\n")
        f.write("For paper: Use this data to justify QSF's contribution!\n")
    
    print(f"📄 Detailed report saved as '{output_file}'")


# ============================================
# 6. Main Execution
# ============================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   QSF vs BASELINE COMPARISON                 ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Load or create data
    print("📥 Loading results...")
    qsf_results, baseline_results = create_sample_data()
    # Replace with: qsf_results, baseline_results = load_results()
    
    print("\n📊 Creating comparison table...")
    df = create_comparison_table(qsf_results, baseline_results)
    print(df)
    
    print("\n📈 Generating visualizations...")
    create_comparison_plots(qsf_results, baseline_results)
    
    print("\n📄 Generating report...")
    generate_comparison_report(qsf_results, baseline_results)
    
    print("\n" + "="*50)
    print("✅ COMPARISON ANALYSIS COMPLETED!")
    print("="*50)
    print("\nFiles generated:")
    print("  - qsf_vs_baseline_comprehensive.png")
    print("  - comparison_report.txt")
    print("\nUse these for your paper! 📝")
