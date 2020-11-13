from qcor import *

@qjit
def ansatz(q : qreg, x : List[float]):
    X(q[0])
    with decompose(q, kak, depends_on=[x]) as u:
        from scipy.sparse.linalg import expm
        from openfermion.ops import QubitOperator
        from openfermion.transforms import get_sparse_operator
        qop = QubitOperator('X0 Y1') - QubitOperator('Y0 X1')
        qubit_sparse = get_sparse_operator(qop)
        u = expm(0.5j * x[0] * qubit_sparse).todense()

H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

vqe_obj = createObjectiveFunction(ansatz, H, 1, {'gradient-strategy':'central', 'step':1e-1})
optimizer = createOptimizer('nlopt', {'initial-parameters':[.5], 'maxeval':10, 'algorithm':'l-bfgs'})
results = optimizer.optimize(vqe_obj)

print('Results:')
print(results)
print()

ansatz.print_kernel(qalloc(2), results[1])