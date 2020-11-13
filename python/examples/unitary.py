from qcor import *

@qjit
def ccnot(q : qreg):
    # create 111
    for i in range(q.size()):
        X(q[i])
            
    with decompose(q) as ccnot:
        ccnot = np.eye(8)
        ccnot[6,6] = 0.0
        ccnot[7,7] = 0.0
        ccnot[6,7] = 1.0
        ccnot[7,6] = 1.0
    
    # CCNOT should produce 110 (lsb)
    for i in range(q.size()):
        Measure(q[i])

q = qalloc(3)
ccnot(q)
counts = q.counts()
print(counts)

# Show of numpy integration with language extension
@qjit
def all_x(q : qreg):
    with decompose(q) as x_kron:
        sx = np.array([[0, 1],[1, 0]])
        x_kron = np.kron(np.kron(sx,sx),sx)
            
    for i in range(q.size()):
        Measure(q[i])

       
q = qalloc(3)
all_x(q)
counts = q.counts()
print(counts)

        
@qjit
def random_1qbit(q : qreg):
    with decompose(q, z_y_z) as random_unitary:
        random_unitary, _ = np.linalg.qr(np.random.rand(2,2), mode='complete')
            
    for i in range(q.size()):
        Measure(q[i])
        
q = qalloc(2)
print(random_1qbit.extract_composite(q).toString())

# Show off how to create a parameterized circuit from 
# a unitary matrix. 

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
q = qalloc(2)
o = createObjectiveFunction(ansatz, H, 1)
opt = createOptimizer('nlopt', {'initial-parameters':[.5]})
results = opt.optimize(o)
print(results)
ansatz.print_kernel(q, results[1])