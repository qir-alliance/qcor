from qcor import *


@qjit
def noisy_zero(q: qreg, cx_count: int):
  H(q)
  for i in range(cx_count):
    CX(q[0], q[1])
  H(q)
  Measure(q)


set_verbose(True)
set_shots(1024)
# Enable validation mode
set_validate(True)
# On the noisy simulator, the validation will be successful for 1 cycle
# but will probably fail when running with 10 cycles.
nb_cycles = 1
q = qalloc(2)
noisy_zero(q, nb_cycles)
q.print()