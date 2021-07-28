from qcor import qjit, qalloc
from qiskit.circuit.library import QFT
n_qubits = 3

# Create the IQFT, reverse bits for qcor
iqft = QFT(n_qubits, inverse=True).reverse_bits()

# ---- Use the following qcor boilerplate to generate QIR ----#
# Creates a kernel parameterized on a qreg
qcor_kernel = qjit(iqft)

# Allocate the qreg
q = qalloc(n_qubits)

# Convert to QIR and write to file
qir = qcor_kernel.qir(q, add_entry_point=False,
                      qiskit_compat=True, kernel_name='py_qiskit_iqft')
with open('iqft.ll', 'w') as f:
    f.write(qir)


