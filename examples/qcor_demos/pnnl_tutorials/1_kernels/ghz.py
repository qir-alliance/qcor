from qcor import *

@qjit
def ghz(q : qreg):
    H(q[0])
    for i in range(q.size()-1):
        X.ctrl([q[i]], q[i+1])

    Measure(q)

set_qpu('ionq')
set_shots(1024)
q = qalloc(6)

ghz(q)
q.print()
# ghz.print_kernel(q)
