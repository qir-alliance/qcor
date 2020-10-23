# Run this from the command line like this
#
# python3 bell.py -shots 100

from qcor import qjit, qalloc, qreg

# To create QCOR quantum kernels in Python one 
# simply creates a Python function, writes Pythonic, 
# XASM-like quantum code, and annotates the kernel 
# to indicate it is meant for QCOR just in time compilation

# NOTE Programmers must type annotate their function arguments

# Define a Bell kernel
@qjit
def bell(q : qreg):
    H(q[0])
    CX(q[0], q[1])
    for i in range(q.size()):
        Measure(q[i])

# Allocate 2 qubits
q = qalloc(2)

# Run the bell experiment
bell(q)

# Print the results
q.print()