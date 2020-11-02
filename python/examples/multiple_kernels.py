# Run this from the command line like this
#
# python3 multiple_kernels.py -shots 100

from qcor import qjit, qalloc, qreg

# To create QCOR quantum kernels in Python one 
# simply creates a Python function, writes Pythonic, 
# XASM-like quantum code, and annotates the kernel 
# to indicate it is meant for QCOR just in time compilation

# NOTE Programmers must type annotate their function arguments

@qjit
def measure_all_qubits(q : qreg):
    for i in range(q.size()):
        Measure(q[i])

# Define a Bell kernel
@qjit
def bell_test(q : qreg):
    H(q[0])
    CX(q[0], q[1])
    # Call other kernels
    measure_all_qubits(q)

# Allocate 2 qubits
q = qalloc(2)

# Inspect the IR
comp = bell_test.extract_composite(q)
print(comp.toString())

# Run the bell experiment
bell_test(q)
# Print the results
q.print()