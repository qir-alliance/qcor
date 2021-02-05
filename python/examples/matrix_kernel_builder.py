from qcor import * 
import numpy as np 

set_qpu('qpp', {'shots':1024})

ccnot = np.eye(8)
ccnot[6,6] = 0.0
ccnot[7,7] = 0.0
ccnot[6,7] = 1.0
ccnot[7,6] = 1.0

# Synthesize the CCNOT kernel 
# using the KernelBuilder
builder = KernelBuilder()
[builder.x(i) for i in range(3)]
builder.synthesize(unitary=ccnot)
builder.measure(range(3))
ccnot_circuit = builder.create()

q = qalloc(3)
ccnot_circuit(q)
print(q.counts())

def generator(x : List[float]):
    # Must provide imports in the function!
    from scipy.sparse.linalg import expm
    from openfermion.ops import QubitOperator
    from openfermion.transforms import get_sparse_operator
    qop = QubitOperator('X0 Y1') - QubitOperator('Y0 X1')
    qubit_sparse = get_sparse_operator(qop)
    return expm(0.5j * x[0] * qubit_sparse).todense()

set_qpu('qpp')
b = KernelBuilder() # can pass this, kernel_args={'x':List[float]}), but will be inferred
b.x(0)
b.synthesize(unitary=generator, method='kak')
ansatz = b.create()

H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

# Create the VQE ObjectiveFunction, use a 
# central finite difference gradient strategy
vqe_obj = createObjectiveFunction(ansatz, H, 1, {'gradient-strategy':'central', 'step':1e-1})

# Create a gradient-based optimizer, l-bfgs
optimizer = createOptimizer('nlopt', {'initial-parameters':[.5], 'maxeval':10, 'algorithm':'l-bfgs'})

# Find the ground state via optimization
results = optimizer.optimize(vqe_obj)

# Print the results
print('Results:')
print(results)