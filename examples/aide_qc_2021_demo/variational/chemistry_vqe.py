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

H = operatorTransform("jw", H)

def opt_function(x : List[float]):
    return ansatz.observe(H, qalloc(4), x[0])

optimizer = createOptimizer("nlopt")
ground_energy, opt_params = optimizer.optimize(opt_function, 1)
print("Energy: ", ground_energy)

H = operatorTransform('qubit-tapering', H)
@qjit
def one_qubit_ansatz(qq:qreg, theta :float, phi:float):
  q = qq.head()
  Ry(q, theta)
  Rz(q, phi)


def opt_function2(x : List[float]):
    return one_qubit_ansatz.observe(H, qalloc(1), x[0], x[1])

ground_energy, opt_params = optimizer.optimize(opt_function2, 2)
print("Energy: ", ground_energy)
