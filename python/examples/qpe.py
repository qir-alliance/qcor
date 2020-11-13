from qcor import *

@qjit
def iqft(q : qreg, startIdx : int, nbQubits : int):
    """
    Define an inverse quantum fourier transform kernel
    """
    for i in range(nbQubits/2):
        Swap(q[startIdx + i], q[startIdx + nbQubits - i - 1])
            
    for i in range(nbQubits-1):
        H(q[startIdx+i])
        j = i +1
        for y in range(i, -1, -1):
            theta = -MY_PI / 2**(j-y)
            CPhase(q[startIdx+j], q[startIdx + y], theta)
            
    H(q[startIdx+nbQubits-1])

@qjit
def oracle(q : qreg):
    """
    Define the oracle for our phase estimation algorithm,
    a T gate on the last qubit
    """
    bit = q.size()-1
    T(q[bit])

@qjit
def qpe(q : qreg):
    """
    Run the quantum phase estimation kernel using the 
    ctrl of the oracle kernel and the pre-defined inverse 
    fourier transform. 
    """
    nq = q.size()
    X(q[nq - 1])
    for i in range(q.size()-1):
        H(q[i])
            
    bitPrecision = nq-1
    for i in range(bitPrecision):
        nbCalls = 2**i
        for j in range(nbCalls):
            ctrl_bit = i
            oracle.ctrl(ctrl_bit, q)
            
    # Inverse QFT on the counting qubits
    iqft(q, 0, bitPrecision)
            
    for i in range(bitPrecision):
        Measure(q[i])

q = qalloc(4)
qpe(q)
print(q.counts())
