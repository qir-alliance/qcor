# Run this from the command line like this
#
# python3 qaoa_circuit.py 

from qcor import *
import numpy as np
from types import MethodType

# Define a QAOA kernel with variational parameters (theta and beta angles)
@qjit
def qaoa_circ(q: qreg, cost_ham: PauliOperator, nbSteps: int, theta: List[float], beta: List[float]):
    # Start off in the uniform superposition
    for i in range(q.size()):
        H(q[i])
    
    terms = cost_ham.getNonIdentitySubTerms()
    for step in range(nbSteps):
        for term in terms:
            exp_i_theta(q, theta[step], term)

        # Reference Hamiltonian: 
        for i in range(len(q)):
            ref_ham_term = X(i)
            exp_i_theta(q, beta[step], ref_ham_term)
   
# Allocate 4 qubits
q = qalloc(4)
n_steps = 3
# Hamiltonion:
H = -5.0 - 0.5 * (Z(0) - Z(3) - Z(1) * Z(2)) - Z(2) + 2 * Z(0) * Z(2) + 2.5 * Z(2) * Z(3)

# Custom arg_translator in a Pythonic way
def qaoa_translate(self, q: qreg, x: List[float]):
    ret_dict = {}    
    ret_dict["q"] = q
    ret_dict["cost_ham"] = H
    ret_dict["nbSteps"] = n_steps
    ret_dict["theta"] = x[:n_steps]
    ret_dict["beta"] = x[n_steps:]
    return ret_dict

# Rebind arg translate:
qaoa_circ.translate = MethodType(qaoa_translate, qjit)

# Use the standard parameterization scheme: 
# one theta + one beta per step
n_params = 2 * n_steps
obj = createObjectiveFunction(qaoa_circ, H, n_params)

# Run optimization
optimizer = createOptimizer('nlopt', {'initial-parameters': np.random.rand(n_params)})
results = optimizer.optimize(obj)
