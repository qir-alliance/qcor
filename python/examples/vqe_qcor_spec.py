from qcor import *

# Define the quantum kernel by providing a 
# python function that is annotated with qjit for 
# quantum just in time compilation
@qjit
def ansatz(q: qreg, theta: List[float]):
    X(q[0])
    Ry(q[1], theta[0])
    CX(q[1], q[0])

# Define the hamiltonian
H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

# Create the ObjectiveFunction, default is VQE
n_params = 1
obj = createObjectiveFunction(ansatz, H, n_params)

# evaluate at a concrete set of params
vqe_energy = obj([.59])
print(vqe_energy) 

# Run full optimization
optimizer = createOptimizer('nlopt')
results = optimizer.optimize(obj)
print(results)