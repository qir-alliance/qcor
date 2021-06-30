from qcor import *

H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

@qjit
def ansatz(q : qreg, theta : float):
    X(q[0])
    Ry(q[1], theta)
    CX(q[1], q[0])

target_energy = -1.74

def objective_function(x):
    q = qalloc(H.nBits())
    energy = ansatz.observe(H, q, x[0])
    return abs(target_energy - energy)

objective_function([2.2])
optimizer = createOptimizer('nlopt', {'nlopt-maxeval':20})
opt_val, opt_params = optimizer.optimize(objective_function, 1)

print(opt_val, opt_params)