from qcor import *

@qjit
def ghz(q : qreg):
    first = q.head()
    H(first)
    for i in range(q.size()-1):
        X.ctrl([q[i]], q[i+1])

    Measure(q)

set_qpu('ibm:ibmq_paris')
set_shots(100)
q = qalloc(6)

ghz(q)

counts = q.counts()
for bit, count in counts.items():
    print(bit, ": ", count)

ghz.print_kernel(q)
