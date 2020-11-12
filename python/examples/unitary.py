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