from qcor import *

@qjit
def bell(q : qreg):
    H(q[0])
    CNOT(q[0], q[1])        
    for i in range(q.size()):
        Measure(q[i])

q = qalloc(2)

print(bell.mlir(q))

print(bell.llvm_mlir(q))

print(bell.llvm_ir(q, add_entry_point=False))
