
from qcor import qjit, qreg
import numpy as np
def decompose(q : qreg):
    return 

@qjit
def ccnot(q :qreg):
    for i in range(q.size()):
        X(q[i])

    with decompose as ccnot_mat:
        ccnot_mat = np.eye(8,8)
        ccnot_mat[6, 6] = 0.0
        ccnot_mat[7, 7] = 0.0
        ccnot_mat[6, 7] = 1.0
        ccnot_mat[7, 6] = 1.0

    for i in range(q.size()):
        Measure(q[i])