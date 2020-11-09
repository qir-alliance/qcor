# Using QCOR FTQC runtime, which supports 
# fast feedforward of measurement results.

from qcor import *
import math

# The Pauli basis is expressed as an integer:
# 0 = I, 1 = X, 2 = Y, 3 = Z
@qjit
def measure_basis(q: qreg, bases: List[int], out_parity: INT_REF):
    oneCount = 0
    for i in range(q.size()):
        pauliOp = bases[i]
        if pauliOp != 0:
            if pauliOp == 1:
                H(q[i])
            if pauliOp == 2:
                Rx(q[i], math.pi/2)
            if Measure(q[i]):
                # TODO: parse Python '+='
                oneCount = 1 + oneCount
    out_parity = oneCount - 2 * (oneCount / 2)

@qjit 
def ansatz(q: qreg, theta: float):
    X(q[0])
    Ry(q[1], theta)
    CX(q[1], q[0])

@qjit 
def reset_all(q: qreg):
    for i in range(q.size()):
        if Measure(q[i]):
            X(q[i])

@qjit 
def estimate_term_expectation(q: qreg, theta: float, bases: List[int], nSamples: int, out_energy: FLOAT_REF):
    sum = 0.0
    for i in range(nSamples):
        ansatz(q, theta)
        parity = 0
        measure_basis(q, bases, parity)
        if parity == 1:
            sum = sum - 1.0
        else:
            sum = sum + 1.0
        reset_all(q)
    out_energy = sum / nSamples

# Currently, we need a *global* variable to get FTQC results :( 
energy = 0.0
# Objective function:
def deuteron_cost_fn(angles: List[float]):
    theta = angles[0]
    H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1)
    # Offset
    total_energy = 5.907
    # Convert the term to bases (list of integers)
    for term in H.getNonIdentitySubTerms():
        term_bases = [0, 0]
        coeff = 0.0
        for op in term:
            coeff = op[1].coeff().real
            for qId in op[1].ops():
                op_name = op[1].ops()[qId]
                if op_name == 'X':
                   term_bases[qId] = 1
                if op_name == 'Y':
                   term_bases[qId] = 2
                if op_name == 'Z':
                    term_bases[qId] = 3
        global energy
        energy = 0.0
        num_samples = 1024
        q = qalloc(2)
        estimate_term_expectation(q, theta, term_bases, num_samples, energy)
        total_energy = total_energy + coeff * energy
    print("E(", theta, ") = ", total_energy)
    return total_energy

from scipy.optimize import minimize
result = minimize(deuteron_cost_fn, 0.0, method='COBYLA')
print(result)
