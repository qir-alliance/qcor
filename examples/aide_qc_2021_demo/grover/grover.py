from qcor import *

@qjit
def oracle_fn(q: qreg):
    CZ(q[0], q[2])
    CZ(q[1], q[2])
        
@qjit
def amplification(q: qreg):
    with compute:
        H(q)
        X(q)
    with action:
        Z.ctrl(q[0: q.size() - 1], q[q.size() - 1])
            
@qjit
def run_grover(q: qreg, oracle_var: KernelSignature(qreg), iterations: int):
    H(q)
    #Iteratively apply the oracle then reflect
    for i in range(iterations):
        oracle_var(q)
        amplification(q)
    # Measure all qubits
    Measure(q)

set_shots(1000)
q = qalloc(3)
run_grover(q, oracle_fn, 1)
counts = q.counts()
print(counts)