# Run this from the command line like this
#
# python3 exp_fermion.py -shots 100

from qcor import *

@qjit
def ansatz(q : qreg, t0: float):
    exponent_op1 = adag(0) * a(1) - adag(1) * a(0)
    X(q[0])
    exp_i_theta(q, t0, exponent_op1)    

# Define the hamiltonian
H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

# Create the ObjectiveFunction, specify central finite diff gradient
obj = createObjectiveFunction(ansatz, H, 1, {'gradient-strategy':'central', 'step':1e-1})

# create the lbfgs optimizer
optimizer = createOptimizer('nlopt', {'algorithm':'l-bfgs', 'ftol':1e-3})

# Run VQE...
results = optimizer.optimize(obj)
print(results)