from qcor import *

@qjit
def ccnot(q : qreg):
    # create 111
    for i in range(q.size()):
        X(q[i])
    # decompose with qfactor        
    with decompose(q, qfactor) as ccnot:
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
