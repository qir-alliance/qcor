from qcor import *

@qjit
def nothing(q : qreg):
  # This is Z Z = I
  Z(q[0])
  H(q[0])
  X(q[0])
  H(q[0])

  H(q[1])
  Rx(q[1], 1.2345)
  H(q[1])

set_opt_level(2)
nothing.print_kernel(qalloc(2))