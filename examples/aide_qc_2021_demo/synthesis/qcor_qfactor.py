from qcor import *

@qjit
def ccnot(q : qreg):
    # create 111
    X(q)

    # decompose with qfactor        
    with decompose(q, qfactor) as ccnot:
        ccnot = np.eye(8)
        ccnot[6,6] = 0.0
        ccnot[7,7] = 0.0
        ccnot[6,7] = 1.0
        ccnot[7,6] = 1.0
    
    # CCNOT should produce 110 (lsb)
    Measure(q)

q = qalloc(3)
ccnot(q)
counts = q.counts()
print(counts)

