from qcor import * 

nq = 10
builder = KernelBuilder() 

builder.h(0)
for i in range(nq-1):
    builder.cnot(i, i+1)
builder.measure_all()
ghz = builder.create()

q = qalloc(nq)
ghz(q)
print(q.counts())
