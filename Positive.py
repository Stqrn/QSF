import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import IBMQ
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# تحميل الحساب من IBM Quantum (إذا لم تكن محملاً مسبقاً)
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

class QSFBellExperiment:
    def __init__(self, backend_name=None):
        self.backend_name = backend_name or 'aer_simulator'
        if 'simulator' in self.backend_name:
            self.backend = Aer.get_backend(self.backend_name)
        else:
            provider = IBMQ.get_provider(hub='ibm-q')
            self.backend = provider.get_backend(backend_name)
        
    def create_bell_circuit(self, apply_measurement=True, lambda_param=0.5, apply_recovery=True):
        """
        ينشئ دائرة تجربة Bell مع/بدون استرجاع
        """
        qc = QuantumCircuit(2, 2)
        
        # 1. تحضير حالة Bell |Φ⁺⟩
        qc.h(0)
        qc.cx(0, 1)
        
        if apply_measurement:
            # 2. قياس مدمر على الكيوبت الأول
            qc.barrier()
            qc.measure(0, 0)
            
            if apply_recovery:
                # 3. عملية الاسترجاع التآزري
                qc.barrier()
                theta = lambda_param * np.pi / 2
                
                # عملية استرجاع تعتمد على λ
                qc.ry(theta, 0)
                qc.ry(theta, 1)
                qc.cx(0, 1)
                qc.ry(-theta, 1)
        
        # 4. قياس نهائي للتحقق من الfidelity
        qc.barrier()
        # تحويل ل basis Bell
        qc.h(0)
        qc.cx(1, 0)
        qc.measure([0, 1], [0, 1])
        
        return qc
    
    def calculate_bell_fidelity(self, counts, total_shots):
        """
        يحسب Bell State Fidelity من النتائج
        """
        # حالات Bell المثالية |Φ⁺⟩ و |Φ⁻⟩
        bell_plus = counts.get('00', 0) + counts.get('11', 0)
        bell_minus = counts.get('01', 0) + counts.get('10', 0)
        
        # Fidelity بالنسبة لـ |Φ⁺⟩
        fidelity = bell_plus / total_shots
        return fidelity
    
    def run_experiment(self, lambda_params, shots=8192):
        """
        يشغل التجربة لمدى من قيم λ
        """
        results = {
            'with_recovery': [],
            'without_recovery': [],
            'no_measurement': []
        }
        
        circuits = []
        
        for lam in lambda_params:
            # مع استرجاع
            qc_recovery = self.create_bell_circuit(
                apply_measurement=True, 
                lambda_param=lam, 
                apply_recovery=True
            )
            qc_recovery.name = f"recovery_lambda_{lam}"
            circuits.append(qc_recovery)
            
            # بدون استرجاع
            qc_no_recovery = self.create_bell_circuit(
                apply_measurement=True, 
                lambda_param=lam, 
                apply_recovery=False
            )
            qc_no_recovery.name = f"no_recovery_lambda_{lam}"
            circuits.append(qc_no_recovery)
            
            # بدون قياس (حالة مرجعية)
            qc_reference = self.create_bell_circuit(
                apply_measurement=False, 
                lambda_param=lam, 
                apply_recovery=False
            )
            qc_reference.name = f"reference_lambda_{lam}"
            circuits.append(qc_reference)
        
        # تشغيل الدوائر
        if 'simulator' in self.backend_name:
            job = execute(circuits, self.backend, shots=shots)
        else:
            # للجهاز الحقيقي: تحويل الدوائر
            transpiled_circuits = transpile(circuits, self.backend)
            job = self.backend.run(transpiled_circuits, shots=shots)
            job_monitor(job)
        
        result = job.result()
        
        # تحليل النتائج
        for i, lam in enumerate(lambda_params):
            idx_recovery = i * 3
            idx_no_recovery = i * 3 + 1
            idx_reference = i * 3 + 2
            
            counts_recovery = result.get_counts(idx_recovery)
            counts_no_recovery = result.get_counts(idx_no_recovery)
            counts_reference = result.get_counts(idx_reference)
            
            fid_recovery = self.calculate_bell_fidelity(counts_recovery, shots)
            fid_no_recovery = self.calculate_bell_fidelity(counts_no_recovery, shots)
            fid_reference = self.calculate_bell_fidelity(counts_reference, shots)
            
            results['with_recovery'].append(fid_recovery)
            results['without_recovery'].append(fid_no_recovery)
            results['no_measurement'].append(fid_reference)
        
        return results, lambda_params
    
    def plot_results(self, results, lambda_params):
        """
        يرسم نتائج التجربة
        """
        plt.figure(figsize=(12, 8))
        
        # رسم الfidelity
        plt.subplot(2, 2, 1)
        plt.plot(lambda_params, results['with_recovery'], 'bo-', label='مع الاسترجاع', linewidth=2)
        plt.plot(lambda_params, results['without_recovery'], 'ro-', label='بدون استرجاع', linewidth=2)
        plt.plot(lambda_params, results['no_measurement'], 'g--', label='بدون قياس (مرجعي)', alpha=0.7)
        plt.xlabel('معامل الاقتران λ')
        plt.ylabel('Bell State Fidelity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('تأثير الاسترجاع التآزري على الFidelity')
        
        # رسم ΔF (التحسن)
        plt.subplot(2, 2, 2)
        delta_f = np.array(results['with_recovery']) - np.array(results['without_recovery'])
        plt.plot(lambda_params, delta_f, 's-', color='purple', linewidth=2)
        plt.xlabel('معامل الاقتران λ')
        plt.ylabel('ΔF = F(مع) - F(بدون)')
        plt.grid(True, alpha=0.3)
        plt.title('التحسن في الFidelity بسبب الاسترجاع')
        
        # رسم نسبة التحسن
        plt.subplot(2, 2, 3)
        improvement_ratio = delta_f / (1 - np.array(results['without_recovery']))
        plt.plot(lambda_params, improvement_ratio, 'o-', color='orange', linewidth=2)
        plt.xlabel('معامل الاقتران λ')
        plt.ylabel('نسبة التحسن')
        plt.grid(True, alpha=0.3)
        plt.title('نسبة الاسترجاع من المعلومات المفقودة')
        
        plt.tight_layout()
        plt.show()
        
        return delta_f, improvement_ratio

# التجربة الرئيسية
if __name__ == "__main__":
    # 1. اختر backend (غير إلى جهاز حقيقي عندما تكون جاهزاً)
    # backend = 'ibm_torino'  # أو 'ibm_brisbane'
    backend = 'aer_simulator'  # للمحاكاة أولاً
    
    # 2. قيم λ التي ستختبرها
    lambda_params = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    
    # 3. إنشاء وتشغيل التجربة
    experiment = QSFBellExperiment(backend_name=backend)
    results, lambda_vals = experiment.run_experiment(lambda_params, shots=8192)
    
    # 4. رسم وتحليل النتائج
    delta_f, improvement_ratio = experiment.plot_results(results, lambda_vals)
    
    # 5. طباعة النتائج الرقمية
    print("\n" + "="*50)
    print("النتائج التجريبية:")
    print("="*50)
    for i, lam in enumerate(lambda_vals):
        print(f"λ = {lam:.2f}:")
        print(f"  - Fidelity مع الاسترجاع: {results['with_recovery'][i]:.4f}")
        print(f"  - Fidelity بدون استرجاع: {results['without_recovery'][i]:.4f}")
        print(f"  - ΔF (التحسن): {delta_f[i]:.4f}")
        print(f"  - نسبة التحسن: {improvement_ratio[i]:.4f}")
        print()
    
    # 6. البحث عن أفضل قيمة لـ λ
    best_idx = np.argmax(delta_f)
    best_lambda = lambda_vals[best_idx]
    best_improvement = delta_f[best_idx]
    
    print(f"أفضل معامل اقتران: λ = {best_lambda}")
    print(f"أقصى تحسن في الFidelity: ΔF = {best_improvement:.4f}")
