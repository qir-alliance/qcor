from qcor import *

@qjit
def ansatz(q : qreg, x : List[float]):
    X(q[0])
    # The goal of 'with decompose(..) as MAT' is to define a 
    # the unitary matrix MAT within the scope of the 
    # with statement. Here we use SciPy and Openfermion 
    # to construct the X0Y1 - Y0X1 operator as a matrix. 
    # Programmers must provide MAT as a numpy.matrix, 
    # here todense() maps the sparse operator to a numpy.matrix.
    # Note that if your matrix is dependent on a kernel argument, 
    # you must define it in the depends_on=[..] decompose arg.
    with decompose(q, kak) as u:
        from scipy.sparse.linalg import expm
        from openfermion.ops import QubitOperator
        from openfermion.linalg import get_sparse_operator
        qop = QubitOperator('X0 Y1') - QubitOperator('Y0 X1')
        qubit_sparse = get_sparse_operator(qop)
        u = expm(0.5j * x[0] * qubit_sparse).todense()

# Define the Hamiltonain
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
print()

# See what the circuit decomposition was at the opt angle
ansatz.print_kernel(qalloc(2), results[1])