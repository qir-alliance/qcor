# Run this from the command line like this
#
# python3 qaoa_circuit.py 

from qcor import *
import numpy as np
# Define a Bell kernel
@qjit
def qaoa_circ(q: qreg, cost_ham: PauliOperator, nbSteps: int, theta: List[float], beta: List[float]):
    # Start off in the uniform superposition
    for i in range(q.size()):
        H(q[i])
    
    terms = cost_ham.getNonIdentitySubTerms()
    for step in range(nbSteps):
        # TODO: this looks weird (terms is a vector)
        # we need to support Pythonic len() method
        for i in range(terms.size()):
            exp_i_theta(q, theta[step], terms[i])

        # Reference Hamiltonian: 
        for i in range(q.size()):
            ref_ham_term = X(i)
            exp_i_theta(q, beta[step], ref_ham_term)
   
# Allocate 4 qubits
q = qalloc(2)
n_steps = 3
# Use the standard parameterization scheme: one param per step
theta_angle = np.random.rand(n_steps)
beta_angle = np.random.rand(n_steps)
H = -5.0 - 0.5 * (Z(0) - Z(3) - Z(1) * Z(2)) - Z(2) + 2 * Z(0) * Z(2) + 2.5 * Z(2) * Z(3)
comp = qaoa_circ.extract_composite(q, H, n_steps, theta_angle, beta_angle)
print(comp.toString())

