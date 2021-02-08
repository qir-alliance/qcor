from qcor import *

exponent_op = adag(0) * a(1) - adag(1) * a(0)

builder = KernelBuilder()

builder.x(0)
builder.exp('theta', exponent_op)
ansatz = builder.create()

# Define the hamiltonian
H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

# Create the ObjectiveFunction, specify central finite diff gradient
obj = createObjectiveFunction(ansatz, H, 1, {'gradient-strategy':'central', 'step':1e-1})

# create the lbfgs optimizer
optimizer = createOptimizer('nlopt', {'algorithm':'l-bfgs', 'ftol':1e-3})

# Run VQE...
results = optimizer.optimize(obj)
print(results)