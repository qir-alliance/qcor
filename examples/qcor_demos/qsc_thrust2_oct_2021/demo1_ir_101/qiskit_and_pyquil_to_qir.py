from qcor import qjit, qalloc
import qiskit

# Generate 3-qubit GHZ state with Qiskit
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()

# Creates a kernel parameterized on a qreg
qcor_kernel = qjit(circ)

# Allocate the qreg
q = qalloc(3)

# Convert to MLIR and print
mlir = qcor_kernel.mlir(q)
print(mlir)

# Convert to QIR and print
qir = qcor_kernel.qir(q, opt=3)
print(qir)

from pyquil import Program
from pyquil.gates import CNOT, H, MEASURE
  
p = Program()
p += H(0)
p += CNOT(0, 1)
ro = p.declare('ro', 'BIT', 2)
p += MEASURE(0, ro[0])
p += MEASURE(1, ro[1])

# This requires rigetti/quilc docker image
qcor_kernel_pyquil = qjit(p, opt=3)
r = qalloc(2)

# Convert to MLIR and print
mlir = qcor_kernel_pyquil.mlir(r)
print(mlir)

# Convert to QIR and print
qir = qcor_kernel_pyquil.qir(r)
print(qir)