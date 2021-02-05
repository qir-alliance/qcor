from qcor import * 

nq = 3
# We assume a default qreg...
builder = KernelBuilder()#kernel_args={'t':float}) 

builder.x(0)
builder.ry(1, ('t',0))
builder.cnot(1, 0) 

# builder.rx(1, ('x', 1))

#builder.measure_all()
ansatz = builder.create()

# q = qalloc(nq)
# ansatz(q, 2.2, [1.1,2.2])
# print(q.counts())
H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

n_params = 1
obj = createObjectiveFunction(ansatz, H, n_params)

# evaluate at a concrete set of params
vqe_energy = obj([.59])
print(vqe_energy) 

# Run full optimization
optimizer = createOptimizer('nlopt')
results = optimizer.optimize(obj)
print(results)