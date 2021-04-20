from qcor import *

@qjit
def nothing(qbits : qreg, n : int, x : float):
  q = qbits.head()
  r = qbits.tail()

  for i in range(n):
    H(q)


  # This is Z Z = I
  Z(q)
  H(q)
  X(q)
  H(q)

  # Should be I if Rx on 0.0
  H(q)
  X.ctrl(q, r)
  Rx(q, x)
  X.ctrl(q, r)
  H(q)

set_opt_level(2)

qbits = qalloc(2)

print(nothing.n_instructions(qbits, 100, 0.0))
