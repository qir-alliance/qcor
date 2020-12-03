from qcor import *
from openfermion.ops import FermionOperator as FOp

# Define the quantum kernel by providing a 
# python function that is annotated with qjit for 
# quantum just in time compilation
@qjit
def ansatz(q: qreg, theta: List[float]):
    X(q[0])
    Ry(q[1], theta[0])
    CX(q[1], q[0])

# Define the hamiltonian
H = FOp('', 0.0002899) + FOp('0^ 0', -.43658) + \
    FOp('1 0^', 4.2866) + FOp('1^ 0', -4.2866) + FOp('1^ 1', 12.25) 

# Create the ObjectiveFunction, default is VQE
n_params = 1
obj = createObjectiveFunction(ansatz, H, n_params)

# Run full optimization
optimizer = createOptimizer('nlopt')
results = optimizer.optimize(obj)
print(results)