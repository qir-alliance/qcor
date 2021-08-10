from qcor import * 

# Print native code targetting a specific QPU backend

# NOTE Programmers must type annotate their function arguments
set_qpu('ibm:ibmq_manhattan')
# Define a Bell kernel
@qjit
def bell(q : qreg):
    H(q[0])
    CX(q[0], q[1])
    for i in range(q.size()):
        Measure(q[i])

# Allocate 2 qubits
q = qalloc(2)

print('XACC IR:')
bell.print_kernel(q)

# Default: QObj for IBM
print('Native QObj code:')
bell.print_native_code(q)
print('===========================')

# Can set the format with format key:
print('Native QASM code:')
bell.print_native_code(q, format='qasm')
