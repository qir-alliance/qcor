from qcor import *
import numpy as np

@qjit 
def hf(q :qreg):
    X(q[0])
    X(q[2])

@qjit
def ucc1(q : qreg, x : float):
    with compute:
        Rx(q[0], np.pi/2.)
        for i in range(3):
            H(q[i+1])
        for i in range(3):
            CX(q[i], q[i+1])
    with action:
        Rz(q[3], x)

@qjit 
def ansatz(q : qreg, x : float):
    hf(q)
    ucc1(q, x)

@qjit
def test_ctrl(q: qreg, d : float):
    ucc1.ctrl(q[4], q, d)

H = createOperator("pyscf", {"basis": "sto-3g", "geometry":'''H  0.000000   0.0      0.0
H   0.0        0.0  .7474'''})

objective = createObjectiveFunction(ansatz, H, 1)
optimizer = createOptimizer("nlopt", {"maxeval": 20})

optval, opt_params = optimizer.optimize(objective)

print("energy = {}".format(optval))

print('\nAnsatz:')
ansatz.print_kernel(qalloc(4), 2.2)

print('\nCtrl-UCC1:')
qq = qalloc(5)
test_ctrl.print_kernel(qq, 2.2)