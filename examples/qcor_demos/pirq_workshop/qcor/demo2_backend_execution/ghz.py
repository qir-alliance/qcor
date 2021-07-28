# Demonstate the ability to interop with other quantum programming framework/IR
# "Run all that is written"

from qcor import qjit, qalloc

# GHZ kernel expressed as QCOR's qjit
# from qcor import qreg, set_shots
# @qjit
# def ghz(q : qreg):
#   H(q[0])
#   for i in range(q.size() - 1):
#     CX(q[i], q[i + 1])
  
#   for i in range(q.size()):
#     Measure(q[i])

# # Allocate 3 qubits
# q = qalloc(3)

# # Run the bell experiment
# set_shots(1024)
# ghz(q)
# q.print()


# Or, import the IR from a Qiskit's QuantumCircuit
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

