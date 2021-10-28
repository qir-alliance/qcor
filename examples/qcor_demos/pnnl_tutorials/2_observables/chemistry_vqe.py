from qcor import *
import numpy as np

@qjit
def ansatz(q : qreg, theta : float):
    X(q[0])
    X(q[2])
    with compute:
        Rx(q[0], np.pi/2.)
        H(q.tail(3))
        for i in range(3):
            X.ctrl([q[i]], q[i+1])
    with action:
        Rz(q.tail(), theta)

h2_geom = '''H  0.000000   0.0      0.0
H   0.0        0.0  .7474'''
H = createOperator("pyscf", {"basis": "sto-3g", "geometry": h2_geom})
print("pyscf operator:")
print(H)

H = operatorTransform("jw", H)
print("JW transform:")
print(H)

H = operatorTransform('qubit-tapering', H)
print("Reduced with qubit tapering:")
print(H)
