# Run this from the command line like this
#
# python3 exp_i_theta.py -shots 100

from qcor import qjit, qalloc, qreg

# To create QCOR quantum kernels in Python one 
# simply creates a Python function, writes Pythonic, 
# XASM-like quantum code, and annotates the kernel 
# to indicate it is meant for QCOR just in time compilation

# NOTE Programmers must type annotate their function arguments

# Define a XASM kernel
@qjit
def exp_circuit(q : qreg, t0: float, t1: float):
    exponent_op1 = X(0) * Y(1) - Y(0) * X(1)
    exponent_op2 = X(0) * Z(1) * Y(2) -  X(2) * Z(1) * Y(0)
    X(q[0])
    exp_i_theta(q, t0, exponent_op1)
    exp_i_theta(q, t1, exponent_op2)
    
    for i in range(q.size()):
        Measure(q[i])

# Allocate 3 qubits
q = qalloc(3)

# Run the  experiment with some random angles
theta1 = 1.234
theta2 = 2.345

# Examine the circuit QASM
comp = exp_circuit.extract_composite(q, theta1, theta2)
print(comp.toString())

# Execute
exp_circuit(q, theta1, theta2)
# Print the results
q.print()