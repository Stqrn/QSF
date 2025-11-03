# ==============================
# التجربة الكاملة: تحويل الصورة إلى حالة كمومية واستعادتها باستخدام QSF
# ==============================

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("تجربة تحويل الصورة إلى حالة كمومية واستعادتها باستخدام QSF")
print("=" * 60)

# ==============================
# الخطوة 1: تحويل الصورة إلى حالة كمومية
# ==============================

def image_to_quantum_state(image_matrix):
    """
    تحويل مصفوفة الصورة إلى حالة كمومية
    """
    print("🔹 جاري تحويل الصورة إلى حالة كمومية...")
    
    # تسطيح المصفوفة وتحويلها إلى متجه
    flattened = image_matrix.flatten()
    
    # التأكد من أن طول المتجه مناسب لعدد كيوبيتات
    n_elements = len(flattened)
    n_qubits = int(np.ceil(np.log2(n_elements)))
    target_length = 2**n_qubits
    
    # إذا كان المتجه قصيرًا، نضيف أصفار
    if len(flattened) < target_length:
        padded = np.zeros(target_length)
        padded[:len(flattened)] = flattened
        flattened = padded
    
    # تطبيع القيم لتكون سعات كمومية صحيحة
    norm = np.linalg.norm(flattened)
    if norm > 0:
        normalized = flattened / norm
    else:
        normalized = flattened
    
    # إنشاء الحالة الكمومية
    quantum_state = Statevector(normalized)
    
    print(f"✅ تم التحويل: {image_matrix.shape} → {n_qubits} كيوبيت")
    print(f"   السعات: {normalized[:4]}...")
    
    return quantum_state, normalized, n_qubits

def create_encoding_circuit(amplitudes, n_qubits):
    """
    إنشاء دارة لترميز السعات في حالة كمومية
    """
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # تطبيق البوابات لتحقيق السعات المطلوبة
    qc.initialize(amplitudes, qr)
    
    return qc

# مثال: مصفوفة صورة 2x2 بسيطة تمثل صورة رمادية
print("\n📊 إنشاء بيانات الصورة الأصلية...")
sample_image = np.array([
    [0.8, 0.6, 0.9],
    [0.4, 0.2, 0.7],
    [0.3, 0.5, 0.1]
])

print("الصورة الأصلية:")
print(sample_image)

# تحويل الصورة إلى حالة كمومية
original_state, original_amplitudes, n_qubits = image_to_quantum_state(sample_image)
original_circuit = create_encoding_circuit(original_amplitudes, n_qubits)

print(f"\nالدائرة الأصلية:")
print(original_circuit.draw(output='text'))

# ==============================
# الخطوة 2: تطبيق هجوم كمومي (ضجيج)
# ==============================

def apply_quantum_attack(circuit, attack_strength=0.3):
    """
    تطبيق هجوم كمومي (ضجيج) على الدارة
    """
    print(f"\n🔹 جاري تطبيق هجوم كمومي (شدة الهجوم: {attack_strength})...")
    
    n_qubits = circuit.num_qubits
    attacked_circuit = circuit.copy()
    
    for qubit in range(n_qubits):
        # تطبيق ضجيج طوري
        attacked_circuit.rz(attack_strength * np.pi, qubit)
        # تطبيق ضجيج سعوي
        attacked_circuit.rx(attack_strength * 0.5, qubit)
    
    print("✅ تم تطبيق الهجوم الكمومي")
    return attacked_circuit

# تطبيق الهجوم
attack_strength = 0.4
attacked_circuit = apply_quantum_attack(original_circuit, attack_strength)

# ==============================
# الخطوة 3: تطبيق إطار العمل QSF للاستعادة
# ==============================

def create_qsf_recovery(circuit, lambda_strength=0.5):
    """
    تطبيق إطار العمل QSF للاستعادة
    """
    print(f"\n🔹 جاري تطبيق QSF للاستعادة (قوة التآزر: {lambda_strength})...")
    
    n_system_qubits = circuit.num_qubits
    
    # إنشاء دارة جديدة مع إضافة كيوبيت المراقب
    qr_system = QuantumRegister(n_system_qubits, 'system')
    qr_observer = QuantumRegister(1, 'observer')
    cr = ClassicalRegister(n_system_qubits, 'c')
    
    recovery_circuit = QuantumCircuit(qr_system, qr_observer, cr)
    
    # دمج الدارة الأصلية مع الدارة الجديدة
    recovery_circuit.compose(circuit, qubits=range(n_system_qubits), inplace=True)
    
    # تطبيق عملية التآزر (Synergic Operator)
    for qubit in range(n_system_qubits):
        # تطبيق بوابة تحكمية بين النظام والمراقب
        recovery_circuit.cx(qubit, n_system_qubits)
        recovery_circuit.ry(lambda_strength, n_system_qubits)
        recovery_circuit.cx(qubit, n_system_qubits)
    
    print("✅ تم تطبيق عملية التآزر بين النظام والمراقب")
    return recovery_circuit, n_system_qubits

def apply_variational_recovery(circuit, system_qubits, theta_params):
    """
    تطبيق خريطة الاستعادة المتغيرة
    """
    print("🔹 جاري تطبيق خريطة الاستعادة المتغيرة...")
    
    # تطبيق بوابات متغيرة على كيوبيتات النظام
    for i, theta in enumerate(theta_params):
        qubit = i % system_qubits
        circuit.ry(theta, qubit)
        circuit.rz(theta * 0.7, qubit)
    
    # قياس كيوبيتات النظام
    for qubit in range(system_qubits):
        circuit.measure(qubit, qubit)
    
    print("✅ تم تطبيق خريطة الاستعادة")
    return circuit

# تطبيق QSF للاستعادة
lambda_strength = 0.6
recovery_circuit, system_qubits = create_qsf_recovery(attacked_circuit, lambda_strength)

# معلمات الاستعادة (يمكن تحسينها باستخدام التحسين)
theta_params = [0.15, 0.25, 0.1, 0.3, 0.2, 0.35]
final_circuit = apply_variational_recovery(recovery_circuit, system_qubits, theta_params)

print(f"\nالدائرة النهائية بعد QSF:")
print(final_circuit.draw(output='text', fold=-1))

# ==============================
# الخطوة 4: محاكاة وقياس النتائج
# ==============================

def simulate_experiment(original_circuit, attacked_circuit, recovered_circuit, shots=8192):
    """
    محاكاة الدارات ومقارنة النتائج
    """
    print(f"\n🔹 جاري محاكاة التجربة ({shots} shot)...")
    
    # المحاكاة للحصول على الحالات
    backend_statevector = Aer.get_backend('statevector_simulator')
    
    # الحصول على الحالات الكمومية
    original_state = Statevector.from_instruction(original_circuit)
    attacked_state = Statevector.from_instruction(attacked_circuit)
    
    # للحصول على الحالة المستعادة، نحتاج لمحاكاة بدون قياس أولاً
    recovered_circuit_no_measure = recovered_circuit.copy()
    recovered_circuit_no_measure.remove_final_measurements()
    recovered_state = Statevector.from_instruction(recovered_circuit_no_measure)
    
    # حساب الأمانة (Fidelity)
    fidelity_attack = state_fidelity(original_state, attacked_state)
    fidelity_recovery = state_fidelity(original_state, recovered_state)
    
    print("📈 نتائج الأمانة (Fidelity):")
    print(f"   • بعد الهجوم: {fidelity_attack:.4f}")
    print(f"   • بعد الاستعادة بـ QSF: {fidelity_recovery:.4f}")
    print(f"   • التحسن: {fidelity_recovery - fidelity_attack:+.4f}")
    
    # محاكاة القياسات
    backend_qasm = Aer.get_backend('qasm_simulator')
    job_original = execute(original_circuit, backend_qasm, shots=shots)
    job_recovered = execute(recovered_circuit, backend_qasm, shots=shots)
    
    counts_original = job_original.result().get_counts()
    counts_recovered = job_recovered.result().get_counts()
    
    return original_state, attacked_state, recovered_state, fidelity_attack, fidelity_recovery, counts_original, counts_recovered

# تشغيل المحاكاة
original_state, attacked_state, recovered_state, fid_attack, fid_recovery, counts_orig, counts_rec = simulate_experiment(
    original_circuit, attacked_circuit, final_circuit
)

# ==============================
# الخطوة 5: استخراج واستعادة قيم الصورة
# ==============================

def extract_image_values(statevector, original_shape):
    """
    استخراج قيم الصورة من الحالة الكمومية
    """
    # الحصول على السعات
    amplitudes = statevector.data
    n_elements = np.prod(original_shape)
    
    # أخذ العناصر الأولى فقط (حسب حجم الصورة الأصلية)
    image_values = np.abs(amplitudes[:n_elements]) ** 2
    
    # إعادة التشكيل إلى شكل الصورة الأصلية
    recovered_image = image_values.reshape(original_shape)
    
    # إعادة القياس إلى المدى الأصلي
    if np.max(recovered_image) > 0:
        recovered_image = recovered_image / np.max(recovered_image)
    
    return recovered_image

print("\n🔹 جاري استخراج قيم الصورة المستعادة...")

# استخراج الصورة المستعادة
recovered_image = extract_image_values(recovered_state, sample_image.shape)

print("✅ تم استخراج الصورة المستعادة")

# ==============================
# الخطوة 6: عرض النتائج والمقارنة
# ==============================

def plot_comprehensive_results(original_img, recovered_img, original_state, attacked_state, recovered_state, counts_orig, counts_rec):
    """
    رسم نتائج شاملة للمقارنة
    """
    print("\n📊 جاري إنشاء الرسوم البيانية...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. مقارنة الصور
    plt.subplot(3, 4, 1)
    plt.imshow(original_img, cmap='gray', vmin=0, vmax=1)
    plt.title('الصورة الأصلية')
    plt.colorbar()
    
    plt.subplot(3, 4, 2)
    attacked_img = extract_image_values(attacked_state, original_img.shape)
    plt.imshow(attacked_img, cmap='gray', vmin=0, vmax=1)
    plt.title('الصورة بعد الهجوم')
    plt.colorbar()
    
    plt.subplot(3, 4, 3)
    plt.imshow(recovered_img, cmap='gray', vmin=0, vmax=1)
    plt.title('الصورة المستعادة بـ QSF')
    plt.colorbar()
    
    plt.subplot(3, 4, 4)
    difference = np.abs(original_img - recovered_img)
    plt.imshow(difference, cmap='hot')
    plt.title('الفرق بين الأصل والمستعاد')
    plt.colorbar()
    
    # 2. مقارنة السعات الكمومية
    plt.subplot(3, 4, 5)
    n_show = 8
    original_amps = np.abs(original_state.data[:n_show])**2
    plt.bar(range(n_show), original_amps, alpha=0.7, label='أصلية')
    plt.title('السعات الكمومية الأصلية')
    plt.xticks(range(n_show))
    
    plt.subplot(3, 4, 6)
    attacked_amps = np.abs(attacked_state.data[:n_show])**2
    plt.bar(range(n_show), attacked_amps, alpha=0.7, color='red', label='بعد الهجوم')
    plt.title('السعات بعد الهجوم')
    plt.xticks(range(n_show))
    
    plt.subplot(3, 4, 7)
    recovered_amps = np.abs(recovered_state.data[:n_show])**2
    plt.bar(range(n_show), recovered_amps, alpha=0.7, color='green', label='مستعادة')
    plt.title('السعات المستعادة')
    plt.xticks(range(n_show))
    
    plt.subplot(3, 4, 8)
    width = 0.25
    x = np.arange(n_show)
    plt.bar(x - width, original_amps, width, label='أصلية', alpha=0.7)
    plt.bar(x, attacked_amps, width, label='بعد الهجوم', alpha=0.7)
    plt.bar(x + width, recovered_amps, width, label='مستعادة', alpha=0.7)
    plt.title('مقارنة السعات')
    plt.xticks(x)
    plt.legend()
    
    # 3. توزيع القياسات
    plt.subplot(3, 4, 9)
    plot_histogram(counts_orig, ax=plt.gca(), color='blue', alpha=0.7)
    plt.title('توزيع القياسات - الأصلية')
    
    plt.subplot(3, 4, 10)
    plot_histogram(counts_rec, ax=plt.gca(), color='green', alpha=0.7)
    plt.title('توزيع القياسات - المستعادة')
    
    # 4. مقاييس الأداء
    plt.subplot(3, 4, 11)
    metrics = ['الأمانة بعد الهجوم', 'الأمانة بعد الاستعادة', 'التحسن']
    values = [fid_attack, fid_recovery, fid_recovery - fid_attack]
    colors = ['red', 'green', 'blue']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('مقاييس الأداء')
    plt.ylim(0, 1)
    
    # إضافة القيم على الأعمدة
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 5. خطأ الاستعادة لكل بكسل
    plt.subplot(3, 4, 12)
    error_per_pixel = np.abs(original_img - recovered_img).flatten()
    plt.plot(error_per_pixel, 'ro-', alpha=0.7)
    plt.title('خطأ الاستعادة لكل بكسل')
    plt.xlabel('رقم البكسل')
    plt.ylabel('قيمة الخطأ')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return difference

# رسم النتائج الشاملة
difference = plot_comprehensive_results(
    sample_image, recovered_image, 
    original_state, attacked_state, recovered_state,
    counts_orig, counts_rec
)

# ==============================
# الخطوة 7: تحليل النتائج النهائية
# ==============================

def print_final_analysis(original_img, recovered_img, difference, fid_attack, fid_recovery):
    """
    طباعة تحليل نهائي مفصل للنتائج
    """
    print("\n" + "=" * 60)
    print("📋 التحليل النهائي للنتائج")
    print("=" * 60)
    
    # حساب مقاييس الخطأ
    mse = np.mean(difference ** 2)
    mae = np.mean(difference)
    max_error = np.max(difference)
    
    # حساب تحسن الأمانة
    fidelity_improvement = fid_recovery - fid_attack
    improvement_percentage = (fidelity_improvement / fid_attack) * 100
    
    print(f"🎯 مقاييس الجودة:")
    print(f"   • الأمانة بعد الهجوم:        {fid_attack:.4f}")
    print(f"   • الأمانة بعد الاستعادة:     {fid_recovery:.4f}")
    print(f"   • التحسن في الأمانة:         {fidelity_improvement:+.4f} ({improvement_percentage:+.1f}%)")
    
    print(f"\n📊 مقاييس الخطأ في الصورة:")
    print(f"   • متوسط الخطأ التربيعي (MSE):  {mse:.6f}")
    print(f"   • متوسط الخطأ المطلق (MAE):   {mae:.6f}")
    print(f"   • أقصى خطأ:                  {max_error:.6f}")
    
    print(f"\n🖼️  مقارنة قيم البكسل:")
    print("   البكسل | الأصل | المستعاد | الفرق")
    print("   " + "-" * 30)
    for i in range(min(6, original_img.size)):
        orig_val = original_img.flatten()[i]
        rec_val = recovered_img.flatten()[i]
        diff_val = abs(orig_val - rec_val)
        print(f"   {i:6} | {orig_val:.3f} | {rec_val:.3f}    | {diff_val:.3f}")
    
    print(f"\n💡 الاستنتاج:")
    if fidelity_improvement > 0.1:
        print("   ✅ QSF حقق تحسنًا كبيرًا في استعادة الحالة الكمومية")
    elif fidelity_improvement > 0.05:
        print("   ✅ QSF حقق تحسنًا ملحوظًا في استعادة الحالة الكمومية")
    elif fidelity_improvement > 0:
        print("   ⚠️  QSF حقق تحسنًا طفيفًا في الاستعادة")
    else:
        print("   ❌ QSF لم يحقق تحسنًا في هذه التجربة")
    
    if mae < 0.1:
        print("   ✅ دقة استعادة الصورة ممتازة")
    elif mae < 0.2:
        print("   ✅ دقة استعادة الصورة جيدة")
    else:
        print("   ⚠️  هناك مجال لتحسين دقة الاستعادة")
    
    print("=" * 60)

# طباعة التحليل النهائي
print_final_analysis(sample_image, recovered_image, difference, fid_attack, fid_recovery)

# ==============================
# الخطوة 8: حفظ النتائج
# ==============================

def save_results(original_img, recovered_img, parameters):
    """
    حفظ النتائج والمعلمات
    """
    print("\n💾 جاري حفظ النتائج...")
    
    # حفظ الصور
    np.savetxt('original_image.txt', original_img, fmt='%.4f')
    np.savetxt('recovered_image.txt', recovered_img, fmt='%.4f')
    
    # حفظ المعلمات
    with open('experiment_parameters.txt', 'w') as f:
        f.write("معلمات تجربة QSF لاستعادة الصورة\n")
        f.write("=" * 40 + "\n")
        f.write(f"شدة الهجوم: {parameters['attack_strength']}\n")
        f.write(f"قوة التآزر (lambda): {parameters['lambda_strength']}\n")
        f.write(f"معلمات الثيتا: {parameters['theta_params']}\n")
        f.write(f"عدد الكيوبيتات: {parameters['n_qubits']}\n")
        f.write(f"الأمانة بعد الهجوم: {parameters['fid_attack']:.4f}\n")
        f.write(f"الأمانة بعد الاستعادة: {parameters['fid_recovery']:.4f}\n")
    
    print("✅ تم حفظ النتائج في الملفات:")
    print("   - original_image.txt")
    print("   - recovered_image.txt") 
    print("   - experiment_parameters.txt")

# حفظ النتائج
experiment_params = {
    'attack_strength': attack_strength,
    'lambda_strength': lambda_strength,
    'theta_params': theta_params,
    'n_qubits': n_qubits,
    'fid_attack': fid_attack,
    'fid_recovery': fid_recovery
}

save_results(sample_image, recovered_image, experiment_params)

print("\n🎉 تم الانتهاء من التجربة بنجاح!")
print("   يمكنك تعديل المعلمات وتحسين النتائج")
