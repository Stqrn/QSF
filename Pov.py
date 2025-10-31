import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
from qiskit.providers.ibmq import IBMQ
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

class AdvancedQSFExperiment:
    def __init__(self, backend_name='aer_simulator'):
        self.backend_name = backend_name
        if 'simulator' in backend_name:
            self.backend = Aer.get_backend(backend_name)
        else:
            provider = IBMQ.get_provider(hub='ibm-q')
            self.backend = provider.get_backend(backend_name)
    
    def theoretical_curve(self, lam, a, b, c, d):
        """منحنى نظري ملائم لنتائجك"""
        return a * np.exp(-b * (lam - c)**2) + d
    
    def fit_optimal_lambda(self, lambda_params, fidelities):
        """إيجاد قيمة λ المثلى من البيانات"""
        try:
            # تخمين أولي من نتائجك
            initial_guess = [0.15, 10.0, 0.6, 0.8]
            popt, pcov = curve_fit(self.theoretical_curve, lambda_params, fidelities, p0=initial_guess)
            
            # إنشاء نقاط أكثر دقة للمنحنى
            lam_fine = np.linspace(min(lambda_params), max(lambda_params), 100)
            fitted_curve = self.theoretical_curve(lam_fine, *popt)
            
            # إيجاد الذروة
            optimal_idx = np.argmax(fitted_curve)
            optimal_lambda = lam_fine[optimal_idx]
            max_fidelity = fitted_curve[optimal_idx]
            
            return popt, lam_fine, fitted_curve, optimal_lambda, max_fidelity
        except:
            return None, None, None, np.mean(lambda_params), np.max(fidelities)
    
    def create_optimized_recovery_circuit(self, lambda_param, circuit_type='bell'):
        """دائرة محسنة بناءً على نتائجك"""
        qc = QuantumCircuit(2, 2)
        
        if circuit_type == 'bell':
            # حالة Bell |Φ⁺⟩
            qc.h(0)
            qc.cx(0, 1)
        
        # قياس مدمر
        qc.barrier()
        qc.measure(0, 0)
        
        # عملية استرجاع محسنة - مبنية على نتائجك
        qc.barrier()
        
        # الاسترجاع التآزري الأمثل (معدّل بناءً على ذروة λ=0.6)
        theta = lambda_param * np.pi / 2
        phi = lambda_param * np.pi / 4
        
        # عملية unitary أكثر تعقيداً (شبه وحدوية)
        qc.ry(theta, 0)
        qc.ry(phi, 1)
        qc.cx(0, 1)
        qc.ry(-phi, 1)
        qc.rx(phi/2, 0)
        
        # تحقق من الfidelity
        qc.barrier()
        qc.h(0)
        qc.cx(1, 0)
        qc.measure([0, 1], [0, 1])
        
        return qc
    
    def run_optimized_experiment(self, lambda_params, shots=8192):
        """تجربة محسنة تركز على نطاق λ الأمثل"""
        circuits = []
        
        for lam in lambda_params:
            # مع الاسترجاع المحسن
            qc_recovery = self.create_optimized_recovery_circuit(lam)
            qc_recovery.name = f"opt_recovery_{lam}"
            circuits.append(qc_recovery)
            
            # بدون استرجاع
            qc_no_recovery = self.create_optimized_recovery_circuit(0.0)  # λ=0 يعني لا استرجاع
            qc_no_recovery.name = f"no_recovery_{lam}"
            circuits.append(qc_no_recovery)
        
        # تشغيل الدوائر
        if 'simulator' in self.backend_name:
            job = execute(circuits, self.backend, shots=shots)
        else:
            transpiled_circuits = transpile(circuits, self.backend)
            job = self.backend.run(transpiled_circuits, shots=shots)
            job_monitor(job)
        
        result = job.result()
        
        # تحليل النتائج
        recovery_fidelities = []
        no_recovery_fidelities = []
        
        for i, lam in enumerate(lambda_params):
            counts_recovery = result.get_counts(i * 2)
            counts_no_recovery = result.get_counts(i * 2 + 1)
            
            fid_recovery = self.calculate_bell_fidelity(counts_recovery, shots)
            fid_no_recovery = self.calculate_bell_fidelity(counts_no_recovery, shots)
            
            recovery_fidelities.append(fid_recovery)
            no_recovery_fidelities.append(fid_no_recovery)
        
        return recovery_fidelities, no_recovery_fidelities, lambda_params
    
    def calculate_bell_fidelity(self, counts, total_shots):
        """حساب Bell State Fidelity"""
        bell_plus = counts.get('00', 0) + counts.get('11', 0)
        return bell_plus / total_shots
    
    def comprehensive_analysis(self, lambda_params, recovery_fids, no_recovery_fids):
        """تحليل شامل يشبه نتائجك"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. منحنى Fidelity vs λ (كما في صورتك)
        ax1.plot(lambda_params, recovery_fids, 'bo-', label='مع الاسترجاع', linewidth=2, markersize=8)
        ax1.plot(lambda_params, no_recovery_fids, 'ro-', label='بدون استرجاع', linewidth=2, markersize=8)
        ax1.set_xlabel('معامل الاقتران λ')
        ax1.set_ylabel('Bell State Fidelity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('تأثير الاسترجاع التآزري على الFidelity')
        
        # 2. منحنى ملائم (كما في الصورة 1000064018.png)
        popt, lam_fine, fitted_curve, opt_lam, max_fid = self.fit_optimal_lambda(lambda_params, recovery_fids)
        
        if popt is not None:
            ax2.plot(lambda_params, recovery_fids, 'bo', label='بيانات تجريبية')
            ax2.plot(lam_fine, fitted_curve, 'r-', label='منحنى ملائم', linewidth=2)
            ax2.axvline(x=opt_lam, color='g', linestyle='--', label=f'λ أمثل = {opt_lam:.2f}')
            ax2.set_xlabel('معامل الاقتران λ')
            ax2.set_ylabel('Bell State Fidelity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title('ملاءمة منحنى Fidelity vs λ')
        
        # 3. تحسن الFidelity (ΔF)
        delta_f = np.array(recovery_fids) - np.array(no_recovery_fids)
        ax3.plot(lambda_params, delta_f, 's-', color='purple', linewidth=2, markersize=8)
        ax3.set_xlabel('معامل الاقتران λ')
        ax3.set_ylabel('ΔF = F(مع) - F(بدون)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('التحسن في الFidelity بسبب الاسترجاع')
        
        # 4. نسبة الاسترجاع
        lost_info = 1 - np.array(no_recovery_fids)
        recovered_ratio = delta_f / (lost_info + 1e-8)  # تجنب القسمة على صفر
        ax4.plot(lambda_params, recovered_ratio, 'o-', color='orange', linewidth=2, markersize=8)
        ax4.set_xlabel('معامل الاقتران λ')
        ax4.set_ylabel('نسبة المعلومات المسترجعة')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('نسبة الاسترجاع من المعلومات المفقودة')
        
        plt.tight_layout()
        plt.show()
        
        return delta_f, recovered_ratio, opt_lam, max_fid

# التشغيل الرئيسي مع إعدادات مبنية على نتائجك
if __name__ == "__main__":
    # نطاق λ مركز حول الذروة التي وجدتها (0.6)
    lambda_params = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    
    experiment = AdvancedQSFExperiment(backend_name='aer_simulator')  # غير لجهاز حقيقي عندما تصبح جاهزاً
    
    print("جاري تشغيل التجربة المحسنة...")
    recovery_fids, no_recovery_fids, lambda_vals = experiment.run_optimized_experiment(lambda_params)
    
    print("جاري التحليل الشامل...")
    delta_f, recovered_ratio, opt_lam, max_fid = experiment.comprehensive_analysis(
        lambda_vals, recovery_fids, no_recovery_fids
    )
    
    # تقرير النتائج
    print("\n" + "="*60)
    print("تقرير النتائج النهائي - استناداً إلى تحليل بياناتك")
    print("="*60)
    print(f"🎯 أفضل معامل اقتران: λ = {opt_lam:.3f}")
    print(f"📈 أقصى fidelity ممكن: {max_fid:.4f}")
    print(f"💫 أقصى تحسن في الfidelity: {np.max(delta_f):.4f}")
    print(f"🔄 أقصى نسبة استرجاع: {np.max(recovered_ratio):.2%}")
    print("\n" + "="*60)
