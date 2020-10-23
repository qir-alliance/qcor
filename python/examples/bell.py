from qcor import qjit, qalloc
import xacc 

@qjit
def bell(q):
    H(q[0])
    CX(q[0], q[1])
    for i in range(q.size()):
        Measure(q[i])

q_reg = qalloc(2)

bell(q_reg)

print('hi')
q_reg.print()


