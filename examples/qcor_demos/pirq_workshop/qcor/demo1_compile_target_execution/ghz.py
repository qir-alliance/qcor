# Demonstate the ability to interop with other quantum programming framework/IR
# "Run all that is written"

from qcor import qjit, qalloc

# Import from Qiskit qiskit_circuit
import qiskit

# Generate 3-qubit GHZ state with Qiskit
qiskit_circ = qiskit.QuantumCircuit(3)
qiskit_circ.h(0)
qiskit_circ.cx(0, 1)
qiskit_circ.cx(1, 2)
qiskit_circ.measure_all()

# Convert Qiskit circuit to QCOR IR
qcor_kernel = qjit(qiskit_circ)

# Allocate the qreg
q = qalloc(3)

# Examine the QCOR IR:
print('QCOR IR:')
qcor_kernel.print_kernel(q)

